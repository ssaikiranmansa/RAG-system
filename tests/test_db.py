"""
DB tests — require a real Postgres + pgvector instance.
Run with: pytest tests/test_db.py

These are skipped automatically in CI unless POSTGRES_HOST is set.
In CI, docker-compose spins up pgvector before these run.
"""
import os
import pytest
import psycopg2

# Skip entire module if no DB is available
pytestmark = pytest.mark.skipif(
    os.environ.get("POSTGRES_HOST") is None,
    reason="No POSTGRES_HOST set — skipping DB tests"
)


@pytest.fixture(scope="module")
def conn():
    """Real DB connection for the test module."""
    c = psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", 5432),
        dbname=os.environ.get("POSTGRES_DB", "ragdb"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
    )
    yield c
    c.close()


@pytest.fixture(autouse=True)
def clean_test_chunks(conn):
    """Delete any chunks with source='test_doc.pdf' before each test."""
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE source = 'test_doc.pdf';")
        conn.commit()
    except Exception:
        conn.rollback()
    yield
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE source = 'test_doc.pdf';")
        conn.commit()
    except Exception:
        conn.rollback()


# ── schema ────────────────────────────────────────────────────────────────────

def test_vector_extension_installed(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        result = cur.fetchone()
    assert result is not None, "pgvector extension not installed"

def test_chunks_table_exists(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_name = 'chunks';
        """)
        result = cur.fetchone()
    assert result is not None

def test_chunks_table_has_all_columns(conn):
    expected = {"id", "source", "chunk_index", "content", "embedding",
                "page_number", "section_header", "word_count"}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'chunks';
        """)
        actual = {row[0] for row in cur.fetchall()}
    assert expected.issubset(actual)

def test_embedding_column_is_vector(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT udt_name FROM information_schema.columns
            WHERE table_name = 'chunks' AND column_name = 'embedding';
        """)
        result = cur.fetchone()
    assert result is not None
    assert result[0] == "vector"


# ── insert ────────────────────────────────────────────────────────────────────

def _fake_embedding(dim=3072):
    return [0.01] * dim

def test_insert_chunk(conn):
    embedding = _fake_embedding()
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO chunks
               (source, chunk_index, content, embedding, page_number, section_header, word_count)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               RETURNING id;""",
            ("test_doc.pdf", 0, "Test content.", embedding, 1, "Intro", 2)
        )
        row_id = cur.fetchone()[0]
    conn.commit()
    assert row_id is not None

def test_insert_multiple_chunks(conn):
    embedding = _fake_embedding()
    with conn.cursor() as cur:
        for i in range(5):
            cur.execute(
                """INSERT INTO chunks
                   (source, chunk_index, content, embedding, page_number, section_header, word_count)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                ("test_doc.pdf", i, f"Chunk {i} content.", embedding, i + 1, "Section", 3)
            )
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks WHERE source = 'test_doc.pdf';")
        count = cur.fetchone()[0]
    assert count == 5

def test_chunk_metadata_stored_correctly(conn):
    embedding = _fake_embedding()
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO chunks
               (source, chunk_index, content, embedding, page_number, section_header, word_count)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            ("test_doc.pdf", 0, "Hello world.", embedding, 7, "Results", 2)
        )
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT page_number, section_header, word_count FROM chunks WHERE source = 'test_doc.pdf';"
        )
        row = cur.fetchone()
    assert row[0] == 7
    assert row[1] == "Results"
    assert row[2] == 2


# ── vector search ─────────────────────────────────────────────────────────────

def test_vector_search_returns_results(conn):
    embedding = _fake_embedding()
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO chunks
               (source, chunk_index, content, embedding, page_number, section_header, word_count)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            ("test_doc.pdf", 0, "Searchable content.", embedding, 1, "Test", 2)
        )
    conn.commit()

    query_embedding = _fake_embedding()
    with conn.cursor() as cur:
        cur.execute(
            """SELECT source, content, 1 - (embedding <=> %s::vector) AS similarity
               FROM chunks
               ORDER BY embedding <=> %s::vector
               LIMIT 5;""",
            (query_embedding, query_embedding)
        )
        rows = cur.fetchall()

    assert len(rows) >= 1
    assert rows[0][0] == "test_doc.pdf"

def test_vector_search_similarity_score(conn):
    """Identical vectors should have similarity ~1.0."""
    embedding = _fake_embedding()
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO chunks
               (source, chunk_index, content, embedding, page_number, section_header, word_count)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            ("test_doc.pdf", 0, "Content.", embedding, 1, None, 1)
        )
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            """SELECT 1 - (embedding <=> %s::vector) AS similarity
               FROM chunks WHERE source = 'test_doc.pdf'
               LIMIT 1;""",
            (embedding,)
        )
        similarity = cur.fetchone()[0]

    assert similarity > 0.99


# ── deduplication guard ───────────────────────────────────────────────────────

def test_source_count_check(conn):
    """Simulates the ingest skip logic — if count > 0, skip."""
    embedding = _fake_embedding()
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO chunks
               (source, chunk_index, content, embedding, page_number, section_header, word_count)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            ("test_doc.pdf", 0, "Already ingested.", embedding, 1, None, 2)
        )
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks WHERE source = %s;", ("test_doc.pdf",))
        count = cur.fetchone()[0]

    assert count > 0  # ingest would skip this source
