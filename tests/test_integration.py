"""
Integration tests — FastAPI endpoints via TestClient.
All Gemini API calls and DB calls are mocked.
No real API key or DB needed.
"""
import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_ingest():
    """Mock the ingest() function called by /ingest endpoint."""
    with patch("main.ingest") as mock:
        mock.return_value = {
            "source": "test.pdf",
            "chunks_ingested": 10,
            "skipped": False
        }
        yield mock


@pytest.fixture
def mock_answer():
    """Mock the answer() function called by /query endpoint."""
    with patch("main.answer") as mock:
        mock.return_value = {
            "answer": "This is a grounded answer from the document.",
            "sources": [
                {
                    "source": "test.pdf",
                    "chunk_index": 0,
                    "content": "Relevant chunk content.",
                    "similarity": 0.92,
                    "page_number": 3,
                    "section_header": "Results",
                    "word_count": 45,
                    "rerank_score": 9
                }
            ],
            "cached": False
        }
        yield mock


@pytest.fixture
def client(mock_ingest, mock_answer):
    """TestClient with all external calls mocked."""
    from main import app
    return TestClient(app)


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ── /ingest ───────────────────────────────────────────────────────────────────

def _pdf_file(name="test.pdf"):
    """Minimal fake PDF bytes for upload."""
    return (name, io.BytesIO(b"%PDF-1.4 fake content"), "application/pdf")


def test_ingest_success(client, mock_ingest):
    response = client.post(
        "/ingest",
        files={"files": _pdf_file()}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert data[0]["source"] == "test.pdf"
    assert data[0]["chunks_ingested"] == 10
    assert data[0]["skipped"] is False


def test_ingest_calls_ingest_function(client, mock_ingest):
    client.post("/ingest", files={"files": _pdf_file()})
    mock_ingest.assert_called_once()


def test_ingest_rejects_non_pdf(client, mock_ingest):
    response = client.post(
        "/ingest",
        files={"files": ("document.txt", io.BytesIO(b"plain text"), "text/plain")}
    )
    assert response.status_code == 400
    assert "not a PDF" in response.json()["detail"]


def test_ingest_multiple_files(client, mock_ingest):
    mock_ingest.side_effect = [
        {"source": "a.pdf", "chunks_ingested": 5, "skipped": False},
        {"source": "b.pdf", "chunks_ingested": 8, "skipped": False},
    ]
    response = client.post(
        "/ingest",
        files=[
            ("files", _pdf_file("a.pdf")),
            ("files", _pdf_file("b.pdf")),
        ]
    )
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_ingest_skipped_response(client, mock_ingest):
    mock_ingest.return_value = {"source": "test.pdf", "chunks_ingested": 10, "skipped": True}
    response = client.post("/ingest", files={"files": _pdf_file()})
    assert response.status_code == 200
    assert response.json()[0]["skipped"] is True


def test_ingest_500_on_exception(client, mock_ingest):
    mock_ingest.side_effect = Exception("Gemini extraction failed")
    response = client.post("/ingest", files={"files": _pdf_file()})
    assert response.status_code == 500
    assert "Ingestion failed" in response.json()["detail"]


# ── /query ────────────────────────────────────────────────────────────────────

def test_query_success(client, mock_answer):
    response = client.post(
        "/query",
        json={"query": "What is the main topic?", "top_k": 5}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)


def test_query_answer_content(client, mock_answer):
    response = client.post("/query", json={"query": "What is X?", "top_k": 5})
    assert response.json()["answer"] == "This is a grounded answer from the document."


def test_query_sources_have_metadata(client, mock_answer):
    response = client.post("/query", json={"query": "What is X?", "top_k": 5})
    source = response.json()["sources"][0]
    assert "page_number" in source
    assert "section_header" in source
    assert "rerank_score" in source


def test_query_default_top_k(client, mock_answer):
    """top_k should default to 5 if not provided."""
    response = client.post("/query", json={"query": "What is X?"})
    assert response.status_code == 200
    _, kwargs = mock_answer.call_args
    args = mock_answer.call_args[0]
    assert args[1] == 5  # default top_k


def test_query_custom_top_k(client, mock_answer):
    response = client.post("/query", json={"query": "What is X?", "top_k": 3})
    assert response.status_code == 200
    args = mock_answer.call_args[0]
    assert args[1] == 3


def test_query_cached_flag(client, mock_answer):
    mock_answer.return_value["cached"] = True
    response = client.post("/query", json={"query": "What is X?", "top_k": 5})
    assert response.json()["cached"] is True


def test_query_empty_string(client, mock_answer):
    """Empty query string should still hit the endpoint (validation is Gemini's job)."""
    response = client.post("/query", json={"query": "", "top_k": 5})
    assert response.status_code == 200


def test_query_500_on_exception(client, mock_answer):
    mock_answer.side_effect = Exception("Vector search failed")
    response = client.post("/query", json={"query": "What is X?", "top_k": 5})
    assert response.status_code == 500
    assert "Query failed" in response.json()["detail"]


def test_query_missing_body(client):
    response = client.post("/query", json={})
    assert response.status_code == 422  # Pydantic validation error
