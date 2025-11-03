from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.rag_pipeline import RagConfig, build_rag_chain


load_dotenv()

st.set_page_config(page_title="Sales Data RAG Assistant", layout="wide")

DATA_PATH = Path(__file__).parent / "src" / "sales_data.csv"

openai_api_key = os.getenv("OPENAI_API_KEY")
default_llm = os.getenv("LLM_MODEL", "gpt-4o-mini")
default_embedding = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

st.title("Sales Data RAG Assistant")
st.write(
    "Ask questions about the provided sales dataset. "
    "The app builds a small retrieval-augmented generation (RAG) pipeline on top of the CSV."
)

if not openai_api_key:
    st.error(
        "OpenAI API key not found. Please set `OPENAI_API_KEY` in your environment or .env file."
    )
    st.stop()


@st.cache_resource(show_spinner="Setting up the RAG pipeline...")
def load_pipeline(llm_model: str, embedding_model: str, top_k: int):
    config = RagConfig(
        llm_model=llm_model,
        embedding_model=embedding_model,
        retrieval_top_k=top_k,
    )
    chain, retriever, _ = build_rag_chain(DATA_PATH, config)
    return chain, retriever


with st.sidebar:
    st.header("Configuration")
    llm_model = st.text_input("LLM model", value=default_llm)
    embedding_model = st.text_input("Embedding model", value=default_embedding)
    top_k = st.slider("Documents to retrieve", min_value=1, max_value=10, value=4)
    st.caption(
        "Defaults are loaded from the `.env` file when available. "
        "Update the fields to experiment with different models."
    )

try:
    rag_chain, retriever = load_pipeline(llm_model, embedding_model, top_k)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # pragma: no cover - surface errors to the UI.
    st.error(f"Failed to initialize the RAG pipeline: {exc}")
    st.stop()

question = st.text_input("Ask a question about the sales data")

if question:
    with st.spinner("Thinking..."):
        try:
            answer = rag_chain.invoke(question)
            st.markdown("**Answer**")
            st.write(answer)
        except Exception as exc:  # pragma: no cover - streamlit feedback
            st.error(f"Failed to generate an answer: {exc}")
        else:
            with st.expander("Retrieved context"):
                docs = retriever.invoke(question)
                if not docs:
                    st.write("Retriever did not return any context.")
                else:
                    for idx, doc in enumerate(docs, start=1):
                        st.markdown(f"**Chunk {idx}**")
                        st.code(doc.page_content)

with st.expander("Preview sales data"):
    if DATA_PATH.exists():
        import pandas as pd

        df_preview = pd.read_csv(DATA_PATH)
        st.dataframe(df_preview.head(20))
    else:
        st.write("Data file not found.")
