"""
Unit tests — no DB, no Gemini API calls.
Tests pure logic in ingest.py and retrieval.py.
"""
import json
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock


# ── ingest: split_into_sentences ──────────────────────────────────────────────

from ingest import split_into_sentences

def test_split_basic():
    text = "Hello world. This is a test. Done."
    sentences = split_into_sentences(text)
    assert len(sentences) == 3

def test_split_preserves_abbreviations():
    text = "Dr. Smith went to Washington. He met Mr. Jones."
    sentences = split_into_sentences(text)
    # "Dr." and "Mr." should NOT cause splits
    assert len(sentences) == 2

def test_split_exclamation_and_question():
    text = "Really? Yes! Absolutely."
    sentences = split_into_sentences(text)
    assert len(sentences) == 3

def test_split_empty():
    assert split_into_sentences("") == []

def test_split_single_sentence():
    result = split_into_sentences("Just one sentence here")
    assert len(result) == 1


# ── ingest: semantic_chunk ────────────────────────────────────────────────────

from ingest import semantic_chunk

def _make_pages(text, page=1, section="Intro"):
    return [{"page": page, "section": section, "text": text}]

def test_chunk_respects_max_words():
    # Sentence-based chunker splits on sentence boundaries, so we need
    # many sentences totalling > max_words. Each sentence is ~10 words.
    sentences = [f"This is sentence number {i} with some extra padding words here." for i in range(60)]
    long_text = " ".join(sentences)
    pages = _make_pages(long_text)
    chunks = semantic_chunk(pages, max_words=100, overlap_sentences=0)
    assert len(chunks) >= 2

def test_chunk_carries_metadata():
    pages = _make_pages("Short text here.", page=3, section="Results")
    chunks = semantic_chunk(pages, max_words=500)
    assert chunks[0]["page_number"] == 3
    assert chunks[0]["section_header"] == "Results"

def test_chunk_index_increments():
    pages = _make_pages(" ".join(["word"] * 1200) + ".")
    chunks = semantic_chunk(pages, max_words=500, overlap_sentences=0)
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(indices)))

def test_chunk_overlap_shares_sentences():
    # With overlap=2, last 2 sentences of chunk N appear at start of chunk N+1
    sentences = [f"Sentence number {i}." for i in range(40)]
    text = " ".join(sentences)
    pages = _make_pages(text)
    chunks = semantic_chunk(pages, max_words=50, overlap_sentences=2)
    assert len(chunks) >= 2
    # content of chunk 1 should contain text from the end of chunk 0
    last_words_of_chunk0 = chunks[0]["content"].split()[-5:]
    chunk1_content = chunks[1]["content"]
    assert any(w in chunk1_content for w in last_words_of_chunk0)

def test_chunk_word_count_recorded():
    pages = _make_pages("One two three four five.")
    chunks = semantic_chunk(pages, max_words=500)
    assert chunks[0]["word_count"] > 0

def test_chunk_empty_page():
    pages = [{"page": 1, "section": None, "text": ""}]
    chunks = semantic_chunk(pages)
    assert chunks == []

def test_chunk_multiple_pages():
    pages = [
        {"page": 1, "section": "Intro", "text": "First page content here."},
        {"page": 2, "section": "Body", "text": "Second page content here."},
    ]
    chunks = semantic_chunk(pages)
    assert len(chunks) == 2
    assert chunks[0]["page_number"] == 1
    assert chunks[1]["page_number"] == 2


# ── ingest: embedding cache ───────────────────────────────────────────────────

from ingest import hash_text, load_cache, save_cache

def test_hash_text_consistent():
    assert hash_text("hello") == hash_text("hello")

def test_hash_text_different():
    assert hash_text("hello") != hash_text("world")

def test_hash_text_returns_string():
    result = hash_text("test")
    assert isinstance(result, str)
    assert len(result) == 32  # MD5 hex digest

def test_cache_roundtrip():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        tmp_path = f.name
    try:
        import ingest as ingest_module
        original = ingest_module.CACHE_FILE
        ingest_module.CACHE_FILE = tmp_path

        data = {"abc123": [0.1, 0.2, 0.3]}
        save_cache(data)
        loaded = load_cache()
        assert loaded == data

        ingest_module.CACHE_FILE = original
    finally:
        os.unlink(tmp_path)

def test_load_cache_missing_file():
    import ingest as ingest_module
    original = ingest_module.CACHE_FILE
    ingest_module.CACHE_FILE = "/tmp/nonexistent_cache_xyz.json"
    result = load_cache()
    assert result == {}
    ingest_module.CACHE_FILE = original


# ── ingest: embed_batch uses cache ────────────────────────────────────────────

def test_embed_batch_uses_cache():
    """embed_batch should NOT call the API for texts already in cache."""
    import ingest as ingest_module

    text = "cached text"
    key = hash_text(text)
    fake_embedding = [0.1] * 3072

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({key: fake_embedding}, f)
        tmp_path = f.name

    original = ingest_module.CACHE_FILE
    ingest_module.CACHE_FILE = tmp_path

    try:
        with patch.object(ingest_module.client.models, 'embed_content') as mock_embed:
            results = ingest_module.embed_batch([text])
            mock_embed.assert_not_called()
            assert results[0] == fake_embedding
    finally:
        ingest_module.CACHE_FILE = original
        os.unlink(tmp_path)


# ── retrieval: dynamic top-k ──────────────────────────────────────────────────

from retrieval import compute_dynamic_top_k

def test_simple_query_reduces_k():
    k = compute_dynamic_top_k("what is RAG?", base_k=5)
    assert k < 5

def test_complex_query_increases_k():
    k = compute_dynamic_top_k("explain how the retrieval pipeline works in detail", base_k=5)
    assert k > 5

def test_default_query_returns_base_k():
    k = compute_dynamic_top_k("tell me about the document", base_k=5)
    assert k == 5

def test_who_is_simple():
    assert compute_dynamic_top_k("who is the author?", base_k=5) < 5

def test_compare_is_complex():
    assert compute_dynamic_top_k("compare the two approaches", base_k=5) > 5

def test_k_never_below_1():
    k = compute_dynamic_top_k("what is x?", base_k=1)
    assert k >= 1

def test_k_capped_at_10():
    k = compute_dynamic_top_k("explain everything in detail", base_k=10)
    assert k <= 10


# ── retrieval: cache key ──────────────────────────────────────────────────────

from retrieval import _cache_key

def test_cache_key_normalizes_case():
    assert _cache_key("Hello", 5) == _cache_key("hello", 5)

def test_cache_key_normalizes_whitespace():
    assert _cache_key("  hello  ", 5) == _cache_key("hello", 5)

def test_cache_key_differs_by_top_k():
    assert _cache_key("hello", 3) != _cache_key("hello", 7)

def test_cache_key_returns_string():
    assert isinstance(_cache_key("query", 5), str)
