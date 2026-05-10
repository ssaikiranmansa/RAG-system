import os
import logging
import tempfile
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from ingest import ingest
from retrieval import answer
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")

# ── configure logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/ingest")
async def ingest_pdf(files: List[UploadFile] = File(...)):
    logger.info("Ingest request received for %d file(s).", len(files))
    results = []

    for file in files:
        if not file.filename.endswith(".pdf"):
            logger.warning("Rejected non-PDF file: %s", file.filename)
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            logger.info("Ingesting file: %s", file.filename)
            result = ingest(tmp_path)
            results.append(result)
            logger.info("Ingested %s — %d chunks (skipped=%s)",
                        file.filename, result["chunks_ingested"], result.get("skipped"))
        except Exception as e:
            logger.error("Ingestion failed for %s: %s", file.filename, e)
            raise HTTPException(
                status_code=500,
                detail=f"Ingestion failed for {file.filename}: {str(e)}"
            )
        finally:
            os.unlink(tmp_path)

    return results


@app.post("/query")
async def query(req: QueryRequest):
    logger.info("Query received: %s (top_k=%d)", req.query[:80], req.top_k)
    try:
        result = answer(req.query, req.top_k)
        logger.info("Query answered successfully.")
        return result
    except Exception as e:
        logger.error("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok"}