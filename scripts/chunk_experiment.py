"""Experiment with different chunk sizes to find optimal retrieval quality.

This script tests chunk sizes of 300, 500, and 1000, measures retrieval
relevance scores, and logs results — the kind of engineering rigor
that sets your project apart from tutorial-level work.

Usage: python scripts/chunk_experiment.py
"""

import os
import sys
import json
import shutil
from datetime import datetime

# Ensure project root is on the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader

from app.config import settings


CHUNK_SIZES = [300, 500, 1000]
TEST_QUERIES = [
    "What is machine learning?",
    "How does RAG work?",
    "What are vector databases used for?",
    "Explain supervised vs unsupervised learning",
    "What is transfer learning?",
]
RESULTS_FILE = "evaluation/chunk_experiment_results.json"


def run_experiment():
    """Test different chunk sizes and measure retrieval quality."""
    print("=" * 60)
    print("CHUNK SIZE EXPERIMENT")
    print("=" * 60)

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Load sample documents
    loader = CSVLoader("sample_data/ai_knowledge_base.csv", encoding="utf-8")
    documents = loader.load()
    print(f"\nLoaded {len(documents)} documents")

    results = {}

    for chunk_size in CHUNK_SIZES:
        print(f"\n--- Testing chunk_size={chunk_size} ---")

        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50,
        )
        chunks = splitter.split_documents(documents)
        print(f"  Created {len(chunks)} chunks")

        # Create temporary vector store
        temp_dir = f"./chroma_experiment_{chunk_size}"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=temp_dir,
        )

        # Test queries — use similarity_search_with_score (raw L2 distance, lower = better)
        query_results = []
        for query_text in TEST_QUERIES:
            docs_with_scores = vector_store.similarity_search_with_score(
                query_text, k=3
            )
            # Convert distances to similarity: 1 / (1 + distance)
            similarities = [1 / (1 + dist) for _, dist in docs_with_scores]
            avg_sim = sum(similarities) / len(similarities) if similarities else 0

            query_results.append({
                "query": query_text,
                "avg_similarity": round(avg_sim, 4),
                "top_similarity": round(max(similarities), 4) if similarities else 0,
                "num_results": len(similarities),
            })
            print(f"  Query: '{query_text[:40]}...' → avg similarity: {avg_sim:.4f}")

        overall_avg = sum(r["avg_similarity"] for r in query_results) / len(query_results)
        results[chunk_size] = {
            "num_chunks": len(chunks),
            "queries": query_results,
            "overall_avg_similarity": round(overall_avg, 4),
        }

        # Close ChromaDB client before cleanup (Windows file lock fix)
        try:
            vector_store._client.close()
        except Exception:
            pass
        import gc
        gc.collect()
        import time
        time.sleep(0.5)
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Find best chunk size
    best_size = max(results, key=lambda s: results[s]["overall_avg_similarity"])

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for size, data in results.items():
        marker = " ← BEST" if size == best_size else ""
        print(f"  chunk_size={size}: avg_similarity={data['overall_avg_similarity']:.4f}, "
              f"chunks={data['num_chunks']}{marker}")

    print(f"\nRecommendation: Use chunk_size={best_size}")

    # Save results
    os.makedirs("evaluation", exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "chunk_sizes_tested": CHUNK_SIZES,
        "best_chunk_size": best_size,
        "results": results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    run_experiment()