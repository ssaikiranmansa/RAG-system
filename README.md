![CI](https://github.com/ssaikiranmansa/RAG-system/actions/workflows/ci.yml/badge.svg)

# RAG API

A production-ready Retrieval-Augmented Generation (RAG) system built with Gemini and pgvector. Ingests PDFs, extracts structured text page-by-page, chunks semantically, stores embeddings in Postgres, and answers questions grounded in the document with citations.

## Architecture

```
PDF Upload
    │
    ▼
Gemini 2.5 Flash — structured text extraction (page + section metadata)
    │
    ▼
Sentence-based semantic chunking (500 words, 2-sentence overlap)
    │
    ▼
Gemini Embedding 001 — batch embed with disk cache (embed_cache.json)
    │
    ▼
pgvector (Postgres) — vector storage + similarity search
    │
    ▼
Query → embed → dynamic top-k search → batch rerank → grounded answer
```

## Stack

| Layer | Choice | Why |
|---|---|---|
| PDF extraction | Gemini 2.5 Flash | Handles scanned/image-heavy PDFs; extracts tables, figures, captions |
| Embeddings | `gemini-embedding-001` (3072-dim) | High-quality, consistent dimensionality |
| Vector store | Postgres + pgvector | Simple ops, no extra infra, strong SQL ergonomics |
| API | FastAPI | Async, typed, auto-docs at `/docs` |

## Features

- **Structured extraction** — page numbers and section headers stored as metadata, enabling precise citations
- **Semantic chunking** — sentence-aware splitting with overlap (no mid-sentence cuts)
- **Embedding cache** — MD5-keyed disk cache; re-ingesting the same doc skips all API calls
- **Dynamic top-k** — adjusts retrieval count based on query complexity (simple → k=3, complex → k=10)
- **Batch reranking** — all chunks scored in a single Gemini call (not N calls)
- **Answer cache** — in-memory cache for identical queries; zero API calls on repeat

## Setup

### 1. Prerequisites

- Docker & Docker Compose
- Python 3.11+
- A Gemini API key → [aistudio.google.com](https://aistudio.google.com)

### 2. Start Postgres with pgvector

```bash
docker-compose up -d
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Create a `.env` file in the root directory with the following variables:

```
GEMINI_API_KEY=your_gemini_api_key_here
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=ragdb
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
```

### 5. Run the API

```bash
cd app
uvicorn main:app --reload
```

API is now live at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest` | Upload one or more PDFs to extract, chunk, embed, and store |
| `POST` | `/query` | Ask a question, get a grounded answer with sources |
| `GET` | `/health` | Health check |

### Ingest a PDF

```bash
curl -X POST http://localhost:8000/ingest \
  -F "files=@document.pdf"
```

Response:
```json
[{"source": "document.pdf", "chunks_ingested": 42, "skipped": false}]
```

Re-ingesting the same file is a no-op (detected by source filename):
```json
[{"source": "document.pdf", "chunks_ingested": 42, "skipped": true}]
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main findings?", "top_k": 5}'
```

Response:
```json
{
  "answer": "According to page 3 (Results section)...",
  "sources": [
    {
      "source": "document.pdf",
      "chunk_index": 7,
      "content": "...",
      "similarity": 0.91,
      "page_number": 3,
      "section_header": "Results",
      "word_count": 312,
      "rerank_score": 9
    }
  ],
  "cached": false
}
```

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI pipeline
├── app/
│   ├── main.py              # FastAPI endpoints (/ingest, /query, /health)
│   ├── ingest.py            # Extraction → chunking → embedding → storage
│   ├── retrieval.py         # Query embedding → search → rerank → answer
│   └── db.py                # Postgres connection + schema init
├── tests/
│   ├── conftest.py          # Shared fixtures and path setup
│   ├── test_unit.py         # Unit tests (chunking, cache, dynamic top-k)
│   ├── test_db.py           # DB tests (schema, insert, vector search)
│   └── test_integration.py  # API endpoint tests
├── docker-compose.yml       # Postgres + pgvector
├── requirements.txt
└── README.md
```