"""RAG query pipeline: retrieve relevant chunks and generate answers with citations."""

import logging
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

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

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def get_llm() -> ChatGoogleGenerativeAI:
    """Initialize the Gemini LLM."""
    return ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=settings.LLM_TEMPERATURE,
        convert_system_message_to_human=True,
    )


def get_retriever(top_k: Optional[int] = None):
    """Create a ChromaDB retriever with similarity score threshold."""
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": top_k or settings.TOP_K,
            "score_threshold": settings.SIMILARITY_THRESHOLD,
        },
    )


def build_rag_chain() -> RetrievalQA:
    """Build the full RAG chain: retriever → LLM with prompt."""
    llm = get_llm()
    retriever = get_retriever()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )
    return chain


def query(question: str) -> dict:
    """Run a question through the RAG pipeline.

    Returns the answer, source documents, and metadata.
    """
    chain = build_rag_chain()
    result = chain.invoke({"query": question})

    # Extract source info from retrieved documents
    sources = []
    for doc in result.get("source_documents", []):
        source_info = {
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "metadata": doc.metadata,
        }
        sources.append(source_info)

    response = {
        "question": question,
        "answer": result["result"],
        "sources": sources,
        "num_sources": len(sources),
    }

    logger.info(f"Query answered with {len(sources)} source chunks")
    return response
