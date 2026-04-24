"""FastAPI application for the RAG Knowledge Assistant."""

import os
import shutil
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag_pipeline import query
from ingestion.ingest import ingest_file, ingest_url

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Knowledge Assistant",
    description="A multi-source RAG system that answers questions with citations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# --- Request/Response Models ---

class QueryRequest(BaseModel):
    question: str


class URLIngestRequest(BaseModel):
    url: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list
    num_sources: int


class IngestResponse(BaseModel):
    status: str
    file: str = ""
    url: str = ""
    pages_loaded: int = 0
    chunks_created: int = 0


# --- Endpoints ---

@app.get("/")
async def root():
    return {
        "service": "RAG Knowledge Assistant",
        "version": "1.0.0",
        "endpoints": {
            "POST /ingest/file": "Upload and ingest a document (PDF, CSV, TXT)",
            "POST /ingest/url": "Ingest content from a web URL",
            "POST /query": "Ask a question against the knowledge base",
            "GET /health": "Health check",
        },
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file_endpoint(file: UploadFile = File(...)):
    """Upload and ingest a document into the knowledge base."""
    # Validate file type
    allowed_extensions = {".pdf", ".csv", ".txt"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {allowed_extensions}",
        )

    # Save uploaded file
    file_path = UPLOAD_DIR / file.filename
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = ingest_file(str(file_path))
        return IngestResponse(**result)

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        file.file.close()


@app.post("/ingest/url", response_model=IngestResponse)
async def ingest_url_endpoint(request: URLIngestRequest):
    """Ingest content from a web URL into the knowledge base."""
    try:
        result = ingest_url(request.url)
        return IngestResponse(**result)
    except Exception as e:
        logger.error(f"URL ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Ask a question against the knowledge base."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = query(request.question)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
