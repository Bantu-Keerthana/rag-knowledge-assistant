"""Streamlit UI for the RAG Knowledge Assistant."""

import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG Knowledge Assistant")
st.caption("Upload documents and ask questions with cited answers.")

# --- Sidebar: Document Ingestion ---
with st.sidebar:
    st.header("📁 Ingest documents")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "csv", "txt"],
        help="Supported formats: PDF, CSV, TXT",
    )

    if uploaded_file and st.button("Ingest file", type="primary"):
        with st.spinner("Ingesting document..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                response = requests.post(f"{API_URL}/ingest/file", files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.success(
                        f"Ingested **{result['file']}**: "
                        f"{result['pages_loaded']} pages → {result['chunks_created']} chunks"
                    )
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            except requests.ConnectionError:
                st.error("Cannot connect to API. Is the server running?")

    st.divider()

    # URL ingestion
    url_input = st.text_input("Or ingest from URL", placeholder="https://example.com/article")
    if url_input and st.button("Ingest URL"):
        with st.spinner("Fetching and ingesting..."):
            try:
                response = requests.post(
                    f"{API_URL}/ingest/url",
                    json={"url": url_input},
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Ingested URL: {result['chunks_created']} chunks created")
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            except requests.ConnectionError:
                st.error("Cannot connect to API. Is the server running?")

# --- Main: Query Interface ---
st.header("💬 Ask a question")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"📄 {len(message['sources'])} source(s)"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:**")
                    st.text(source["content"])
                    st.json(source["metadata"])

# Chat input
if prompt := st.chat_input("Ask something about your documents..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from API
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": prompt},
                )
                if response.status_code == 200:
                    result = response.json()
                    st.markdown(result["answer"])

                    if result["sources"]:
                        with st.expander(f"📄 {result['num_sources']} source(s)"):
                            for i, source in enumerate(result["sources"], 1):
                                st.markdown(f"**Source {i}:**")
                                st.text(source["content"])
                                st.json(source["metadata"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })
                else:
                    error_msg = f"Error: {response.json().get('detail', 'Unknown error')}"
                    st.error(error_msg)

            except requests.ConnectionError:
                st.error("Cannot connect to API. Start the server with: `uvicorn main:app --reload`")
