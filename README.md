# RAG Knowledge Assistant

A production-grade, multi-source Retrieval-Augmented Generation (RAG) system that ingests PDFs, web pages, and CSVs, answers questions with source citations, evaluates itself for hallucinations, and deploys via CI/CD.

## Architecture

```
[PDF / CSV / Web] → [LangChain Loaders] → [Text Splitter]
       ↓
[HuggingFace Embeddings] → [ChromaDB Vector Store]
       ↓
[Query] → [Retriever (top-k)] → [Gemini 2.0 Flash] → [Answer + Sources]
       ↓
[DeepEval Harness] → [Hallucination Score] → [GitHub Actions gate]
```

## Tech Stack

| Layer | Tool | Cost |
|-------|------|------|
| LLM | Gemini 2.0 Flash | Free API |
| Embeddings | all-MiniLM-L6-v2 | Local, free |
| Vector DB | ChromaDB | Local, free |
| Framework | LangChain | Open source |
| API | FastAPI + Uvicorn | Open source |
| Evaluation | DeepEval | Open source |
| UI | Streamlit | Open source |
| CI/CD | GitHub Actions | Free tier |
| Deployment | Google Cloud Run | Free tier |

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/YOUR_USERNAME/rag-knowledge-assistant.git
cd rag-knowledge-assistant
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

Get your free Gemini API key at [ai.google.dev](https://ai.google.dev/).

### 3. Run the API

```bash
uvicorn main:app --reload --port 8000
```

### 4. Run the Streamlit UI

```bash
streamlit run streamlit_app/app.py
```

### 5. Or use Docker

```bash
docker-compose up --build
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check |
| `POST` | `/ingest/file` | Upload and ingest a document |
| `POST` | `/ingest/url` | Ingest from a web URL |
| `POST` | `/query` | Ask a question |

### Example Usage

```bash
# Ingest a PDF
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@sample_data/report.pdf"

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?"}'
```

## Evaluation

The project includes a DeepEval-based evaluation harness that measures:

- **Hallucination rate** — does the answer contradict retrieved context?
- **Faithfulness** — is the answer grounded in the retrieved docs?
- **Contextual relevancy** — were the retrieved chunks actually relevant?

```bash
# Run evaluation
deepeval test run tests/test_rag.py --verbose

# Check CI threshold
python scripts/check_eval_threshold.py
```

## CI/CD Pipeline

Every pull request triggers:
1. Evaluation suite runs against the test dataset
2. Hallucination threshold gate — merges blocked if rate > 10%
3. On merge to main — auto-deploy to Google Cloud Run

## Project Structure

```
rag-knowledge-assistant/
├── app/
│   ├── config.py          # Centralized settings
│   └── rag_pipeline.py    # RAG chain: retrieve + generate
├── ingestion/
│   └── ingest.py          # Document loading + chunking + embedding
├── evaluation/
│   └── eval_results.csv   # Tracked eval scores over time
├── scripts/
│   └── check_eval_threshold.py  # CI gate script
├── streamlit_app/
│   └── app.py             # Chat UI
├── tests/
│   ├── test_rag.py        # DeepEval test harness
│   └── test_dataset.json  # Ground-truth Q&A pairs
├── sample_data/           # Sample documents for demo
├── .github/workflows/
│   └── ci.yml             # GitHub Actions CI/CD
├── main.py                # FastAPI application
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## License

MIT
