"""Document ingestion pipeline: load, chunk, embed, and store documents."""

import os
import logging
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from app.config import settings

logger = logging.getLogger(__name__)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialize the HuggingFace embedding model (runs locally, free)."""
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store(
    embeddings: Optional[HuggingFaceEmbeddings] = None,
) -> Chroma:
    """Get or create the ChromaDB vector store."""
    if embeddings is None:
        embeddings = get_embeddings()

    return Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )


def load_document(file_path: str) -> list:
    """Load a document based on its file extension.

    Supports: .pdf, .csv, .txt
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".csv":
        loader = CSVLoader(file_path, encoding="utf-8")
    elif ext == ".txt":
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf, .csv, or .txt")

    docs = loader.load()
    logger.info(f"Loaded {len(docs)} pages/rows from {file_path}")
    return docs


def load_web_page(url: str) -> list:
    """Load content from a web URL."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} documents from {url}")
    return docs


def chunk_documents(
    documents: list,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list:
    """Split documents into chunks using recursive character splitting."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


def ingest_file(file_path: str) -> dict:
    """Full ingestion pipeline for a local file.

    Load → Chunk → Embed → Store in ChromaDB.
    Returns metadata about what was ingested.
    """
    # Load
    documents = load_document(file_path)

    # Chunk
    chunks = chunk_documents(documents)

    # Add source metadata
    for chunk in chunks:
        chunk.metadata["source_file"] = os.path.basename(file_path)

    # Embed and store
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)
    vector_store.add_documents(chunks)

    result = {
        "file": os.path.basename(file_path),
        "pages_loaded": len(documents),
        "chunks_created": len(chunks),
        "status": "success",
    }
    logger.info(f"Ingestion complete: {result}")
    return result


def ingest_url(url: str) -> dict:
    """Full ingestion pipeline for a web URL."""
    documents = load_web_page(url)
    chunks = chunk_documents(documents)

    for chunk in chunks:
        chunk.metadata["source_url"] = url

    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)
    vector_store.add_documents(chunks)

    result = {
        "url": url,
        "pages_loaded": len(documents),
        "chunks_created": len(chunks),
        "status": "success",
    }
    logger.info(f"Ingestion complete: {result}")
    return result
