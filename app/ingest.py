import os
import re
import json
import hashlib
import logging
from google import genai
from db import get_conn, init_db
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")

logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

CACHE_FILE = os.path.join(os.path.dirname(__file__), "embed_cache.json")


# ── embedding cache helpers ───────────────────────────────────────────────────

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# ── extraction ────────────────────────────────────────────────────────────────

def extract_structured_text(pdf_path: str) -> list[dict]:
    """
    Upload PDF to Gemini and extract structured text with page numbers
    and section headers. Returns a list of page dicts:
    [{"page": 1, "section": "Introduction", "text": "..."}, ...]
    """
    logger.info("Uploading PDF to Gemini: %s", os.path.basename(pdf_path))
    file = client.files.upload(file=pdf_path, config={"mime_type": "application/pdf"})
    logger.info("Extracting structured text from PDF...")

    prompt = """Read this entire PDF document page by page including all visual content.
For each page return the output in this exact format:

PAGE: <page_number>
SECTION: <section header or title on this page, or 'None' if not present>
TEXT:
<full text visible on the page including text in images, charts, tables, headers, footers and captions>
---

Important instructions:
- Read ALL text visible on every page — do not skip any page
- For tables: write a one line human readable summary before the raw table data
- For charts and figures: describe what the figure shows in one sentence then extract any visible numbers or labels
- For scanned or image based pages: read the text as it visually appears
- Preserve all content verbatim including numbers, names and dates
- Do not skip pages even if they appear to be mostly visual"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[file, prompt]
    )

    raw = response.text
    pages = []

    blocks = re.split(r"\n---\n", raw)
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        page_match = re.search(r"PAGE:\s*(\d+)", block)
        section_match = re.search(r"SECTION:\s*(.+)", block)
        text_match = re.search(r"TEXT:\s*\n([\s\S]+)", block)

        page_num = int(page_match.group(1)) if page_match else None
        section = section_match.group(1).strip() if section_match else None
        section = None if section and section.lower() == "none" else section
        text = text_match.group(1).strip() if text_match else block

        if text:
            pages.append({
                "page": page_num,
                "section": section,
                "text": text
            })

    if not pages:
        logger.warning("Structured parsing failed for %s — falling back to plain extraction.", os.path.basename(pdf_path))
        pages = [{"page": None, "section": None, "text": raw}]

    logger.info("Extracted %d pages from %s", len(pages), os.path.basename(pdf_path))
    return pages


# ── chunking ──────────────────────────────────────────────────────────────────

def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using punctuation boundaries.
    Handles common abbreviations to avoid false splits.
    """
    text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|Fig|eq|al)\.',
                  lambda m: m.group().replace('.', '<DOT>'), text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.replace('<DOT>', '.').strip() for s in sentences]
    return [s for s in sentences if s]


def semantic_chunk(pages: list[dict], max_words: int = 500, overlap_sentences: int = 2) -> list[dict]:
    """
    Sentence-based semantic chunking.
    Groups sentences into chunks up to max_words.
    Overlaps by carrying over the last overlap_sentences sentences into the next chunk.
    """
    chunks = []
    chunk_index = 0

    for page in pages:
        page_num = page["page"]
        section = page["section"]
        text = page["text"]

        sentences = split_into_sentences(text)
        if not sentences:
            continue

        current_sentences = []
        current_word_count = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_words = len(sentence.split())

            if current_word_count + sentence_words > max_words and current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append({
                    "chunk_index": chunk_index,
                    "content": chunk_text,
                    "page_number": page_num,
                    "section_header": section,
                    "word_count": current_word_count
                })
                chunk_index += 1

                overlap = current_sentences[-overlap_sentences:] if len(current_sentences) >= overlap_sentences else current_sentences[:]
                current_sentences = overlap
                current_word_count = sum(len(s.split()) for s in current_sentences)

            current_sentences.append(sentence)
            current_word_count += sentence_words
            i += 1

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append({
                "chunk_index": chunk_index,
                "content": chunk_text,
                "page_number": page_num,
                "section_header": section,
                "word_count": current_word_count
            })
            chunk_index += 1

    return chunks


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Original word-based chunking — kept for backward compatibility."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


# ── embedding ─────────────────────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return response.embeddings[0].values


def embed_batch(texts: list[str], batch_size: int = 20) -> list[list[float]]:
    """
    Batch embed with cache — skips API call for texts already embedded before.
    Cache persisted to embed_cache.json so re-ingesting the same doc
    skips all embedding API calls entirely.
    """
    cache = load_cache()
    results = [None] * len(texts)
    to_embed_indices = []
    to_embed_texts = []

    for i, text in enumerate(texts):
        key = hash_text(text)
        if key in cache:
            results[i] = cache[key]
            logger.debug("Cache hit for chunk %d", i)
        else:
            to_embed_indices.append(i)
            to_embed_texts.append(text)

    logger.info("%d chunks need embedding, %d served from cache.",
                len(to_embed_texts), len(texts) - len(to_embed_texts))

    for b in range(0, len(to_embed_texts), batch_size):
        batch = to_embed_texts[b:b + batch_size]
        batch_indices = to_embed_indices[b:b + batch_size]
        batch_num = b // batch_size + 1
        logger.info("Embedding batch %d (%d chunks)...", batch_num, len(batch))

        try:
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=batch,
            )
        except Exception as e:
            logger.error("Embedding batch %d failed: %s", batch_num, e)
            raise

        for idx, e, text in zip(batch_indices, response.embeddings, batch):
            key = hash_text(text)
            cache[key] = e.values
            results[idx] = e.values

    save_cache(cache)
    return results


# ── ingest ────────────────────────────────────────────────────────────────────

def ingest(pdf_path: str, original_name: str = None):
    init_db()
    source = original_name if original_name else os.path.basename(pdf_path)

    logger.info("Starting ingestion for: %s", source)

    # skip if this source already exists in the database
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks WHERE source = %s;", (source,))
            count = cur.fetchone()[0]
            if count > 0:
                logger.info("'%s' already ingested (%d chunks). Skipping.", source, count)
                return {"source": source, "chunks_ingested": count, "skipped": True}

    # step 1: extract structured text page by page
    try:
        pages = extract_structured_text(pdf_path)
    except Exception as e:
        logger.error("Text extraction failed for %s: %s", source, e)
        raise

    # step 2: semantic sentence-based chunking with metadata
    chunks = semantic_chunk(pages, max_words=500, overlap_sentences=2)
    logger.info("Created %d semantic chunks from %s", len(chunks), source)

    # step 3: batch embed with cache
    try:
        texts = [c["content"] for c in chunks]
        embeddings = embed_batch(texts)
    except Exception as e:
        logger.error("Embedding failed for %s: %s", source, e)
        raise

    # step 4: store in database with metadata
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                for chunk, embedding in zip(chunks, embeddings):
                    cur.execute(
                        """INSERT INTO chunks
                           (source, chunk_index, content, embedding, page_number, section_header, word_count)
                           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                        (
                            source,
                            chunk["chunk_index"],
                            chunk["content"],
                            embedding,
                            chunk["page_number"],
                            chunk["section_header"],
                            chunk["word_count"]
                        )
                    )
            conn.commit()
    except Exception as e:
        logger.error("Database insert failed for %s: %s", source, e)
        raise

    logger.info("Ingestion complete: %d chunks stored for %s", len(chunks), source)
    return {"source": source, "chunks_ingested": len(chunks), "skipped": False}