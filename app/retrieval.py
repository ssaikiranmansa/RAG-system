import os
import re
import json
import hashlib
import logging
from google import genai
from db import get_conn
from ingest import embed
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")

logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# ── answer cache ──────────────────────────────────────────────────────────────
# in-memory cache — returns identical queries instantly with 0 API calls
# resets on server restart — use Redis in production for persistence
_answer_cache: dict = {}


def _cache_key(query: str, top_k: int) -> str:
    normalized = query.strip().lower()
    return hashlib.md5(f"{normalized}:{top_k}".encode()).hexdigest()


# ── dynamic top-k ─────────────────────────────────────────────────────────────

def compute_dynamic_top_k(query: str, base_k: int = 5) -> int:
    """
    Dynamically adjusts how many chunks to retrieve based on query complexity.

    Simple factual questions (who, what, when, where) → smaller k (3)
    Complex questions (how, why, compare, explain, summarize) → larger k (10)
    Default → base_k (5)
    """
    query_lower = query.lower().strip()

    simple_signals = ["what is", "who is", "when did", "where is", "how many", "what are"]
    complex_signals = ["how does", "why does", "explain", "compare", "summarize",
                       "describe", "what are the differences", "in detail", "elaborate"]

    if any(query_lower.startswith(s) for s in simple_signals):
        k = max(3, base_k - 2)
        logger.info("Dynamic top-k: simple query → k=%d", k)
        return k

    if any(s in query_lower for s in complex_signals):
        k = min(10, base_k + 5)
        logger.info("Dynamic top-k: complex query → k=%d", k)
        return k

    logger.info("Dynamic top-k: default → k=%d", base_k)
    return base_k


# ── reranking ─────────────────────────────────────────────────────────────────

def rerank(query: str, chunks: list[dict]) -> list[dict]:
    """
    Reranks retrieved chunks using a single batched Gemini call.

    Previous approach: 1 API call per chunk → N API calls total.
    Current approach:  all chunks scored in 1 API call → always 1 call.

    Gemini scores all chunks 0-10 in one prompt returning a JSON array.
    Chunks are sorted by score descending.
    Chunks scoring below 3 are filtered out as irrelevant.
    """
    if not chunks:
        return chunks

    logger.info("Reranking %d chunks in a single API call...", len(chunks))

    # build numbered passage list — truncate each to 300 chars to keep prompt small
    passages = "\n\n".join(
        f"{i+1}. {c['content'][:300]}"
        for i, c in enumerate(chunks)
    )

    prompt = f"""Score each passage for relevance to the query.
Return ONLY a valid JSON array of integers, one score per passage, from 0 to 10.
Example for 3 passages: [7, 2, 9]
No explanation. No other text. Just the JSON array.

Query: {query}

Passages:
{passages}

Scores:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"temperature": 0.1}
        )
        raw = response.text.strip()

        # extract JSON array from response
        array_match = re.search(r'\[[\d,\s]+\]', raw)
        if array_match:
            import json
            scores = json.loads(array_match.group())
            scores = [max(0, min(10, int(s))) for s in scores]
        else:
            # fallback — try extracting all numbers in order
            numbers = re.findall(r'\d+', raw)
            scores = [max(0, min(10, int(n))) for n in numbers[:len(chunks)]]

        # pad with 0 if Gemini returned fewer scores than chunks
        while len(scores) < len(chunks):
            scores.append(0)

        logger.info("Batch rerank scores: %s", scores)

    except Exception as e:
        logger.warning("Batch reranking failed: %s — defaulting all scores to 0.", e)
        scores = [0] * len(chunks)

    # assign scores to chunks
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = score
        logger.debug("Chunk %d → rerank score: %d", chunk["chunk_index"], score)

    chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

    filtered = [c for c in chunks if c["rerank_score"] >= 3]

    if not filtered:
        logger.warning("All chunks scored below threshold — keeping top 2 anyway.")
        filtered = chunks[:2]

    logger.info("Reranking complete: %d/%d chunks kept.", len(filtered), len(chunks))
    return filtered


# ── search ────────────────────────────────────────────────────────────────────

def search(query: str, top_k: int = 5) -> list[dict]:
    """
    Embed the query → vector search top-k → rerank results.
    """
    k = compute_dynamic_top_k(query, base_k=top_k)

    logger.info("Embedding query...")
    try:
        query_embedding = embed(query)
    except Exception as e:
        logger.error("Query embedding failed: %s", e)
        raise

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT source, chunk_index, content,
                           1 - (embedding <=> %s::vector) AS similarity,
                           page_number, section_header, word_count
                    FROM chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, k)
                )
                rows = cur.fetchall()
    except Exception as e:
        logger.error("Vector search failed: %s", e)
        raise

    chunks = [
        {
            "source": r[0],
            "chunk_index": r[1],
            "content": r[2],
            "similarity": r[3],
            "page_number": r[4],
            "section_header": r[5],
            "word_count": r[6]
        }
        for r in rows
    ]

    logger.info("Vector search returned %d candidates. Reranking...", len(chunks))
    reranked = rerank(query, chunks)
    return reranked


# ── answer ────────────────────────────────────────────────────────────────────

def answer(query: str, top_k: int = 5) -> dict:
    """
    Full RAG pipeline:
    1. Check answer cache — return immediately if same query asked before
    2. Embed query
    3. Dynamic top-k vector search
    4. Rerank results
    5. Compose grounded answer with citations
    6. Cache answer for future identical queries
    """
    logger.info("Processing query: %s", query[:80])

    # step 1: check answer cache
    key = _cache_key(query, top_k)
    if key in _answer_cache:
        logger.info("Answer cache hit — returning instantly. 0 API calls made.")
        cached = _answer_cache[key].copy()
        cached["cached"] = True
        return cached

    chunks = search(query, top_k)

    context = "\n\n".join(
        f"[Source: {c['source']} | Page {c['page_number']} | Section: {c['section_header']}]\n{c['content']}"
        if c['page_number'] or c['section_header']
        else f"[Source: {c['source']}]\n{c['content']}"
        for c in chunks
    )

    prompt = f"""Answer the question using only the context below.
The context may contain tables represented as plain text with numbers and column headers.
Interpret the numbers in context to answer the question — do not ignore numeric data.
If the answer is not in the context, say "I don't know."
Where possible, cite the page number and section your answer comes from.

Context:
{context}

Question: {query}"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
    except Exception as e:
        logger.error("Answer generation failed: %s", e)
        raise

    logger.info("Answer generated successfully.")
    result = {"answer": response.text, "sources": chunks, "cached": False}
    _answer_cache[key] = result
    logger.info("Answer cached. Cache size: %d", len(_answer_cache))
    return result