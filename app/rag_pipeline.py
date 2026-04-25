"""RAG query pipeline: retrieve relevant chunks and generate answers with citations."""

import logging
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from ingestion.ingest import get_embeddings, get_vector_store

logger = logging.getLogger(__name__)

RAG_PROMPT_TEMPLATE = """You are a helpful knowledge assistant. Answer the user's question
based ONLY on the provided context. If the context does not contain enough information
to answer the question, say: "I don't have enough information in my knowledge base to
answer that."

Always cite which source document(s) your answer comes from.

Context:
{context}

Question: {question}

Answer (with source citations):"""

RAG_PROMPT = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


def get_llm() -> ChatGoogleGenerativeAI:
    """Initialize the Gemini LLM."""
    return ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=settings.LLM_TEMPERATURE,
    )


def get_retriever(top_k: Optional[int] = None):
    """Create a ChromaDB retriever."""
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    return vector_store.as_retriever(
        search_kwargs={
            "k": top_k or settings.TOP_K,
        },
    )


def query(question: str) -> dict:
    """Run a question through the RAG pipeline.

    Retrieves relevant chunks, sends them + question to Gemini,
    returns the answer with source documents.
    """
    retriever = get_retriever()
    llm = get_llm()

    # Step 1: Retrieve relevant documents
    source_documents = retriever.invoke(question)

    if not source_documents:
        return {
            "question": question,
            "answer": "I don't have enough information in my knowledge base to answer that.",
            "sources": [],
            "num_sources": 0,
        }

    # Step 2: Build context from retrieved chunks
    context = "\n\n---\n\n".join(doc.page_content for doc in source_documents)

    # Step 3: Generate answer using LLM
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    # Step 4: Format source info
    sources = []
    for doc in source_documents:
        source_info = {
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "metadata": doc.metadata,
        }
        sources.append(source_info)

    response = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "num_sources": len(sources),
    }

    logger.info(f"Query answered with {len(sources)} source chunks")
    return response