"""Minimal RAG pipeline utilities for the Streamlit demo app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore


@dataclass
class RagConfig:
    """Runtime configuration for the retrieval augmented generation chain."""

    llm_model: str
    embedding_model: str = "text-embedding-3-small"
    retrieval_top_k: int = 5


def _format_row(row: dict) -> str:
    """Create a compact natural language snippet from a single CSV row."""
    parts = [f"{key}: {value}" for key, value in row.items()]
    return "\n".join(parts)


def _format_docs(docs: Sequence[Document]) -> str:
    """Concatenate retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def dataframe_to_documents(df: pd.DataFrame) -> List[Document]:
    """Convert a dataframe into LangChain documents."""
    documents: List[Document] = []
    for idx, row in enumerate(df.to_dict(orient="records")):
        filtered = {k: v for k, v in row.items() if pd.notna(v)}
        if not filtered:
            continue
        documents.append(
            Document(
                page_content=_format_row(filtered),
                metadata={"row_index": idx},
            )
        )
    if not documents:
        raise ValueError("No valid records found to build documents from the dataframe.")
    return documents


def build_vector_store(docs: Iterable[Document], config: RagConfig) -> InMemoryVectorStore:
    """Create an in-memory vector store from prepared documents."""
    embeddings = OpenAIEmbeddings(model=config.embedding_model)
    return InMemoryVectorStore.from_documents(documents=list(docs), embedding=embeddings)


def build_rag_chain(data_path: Path, config: RagConfig):
    """Build a simple RAG chain backed by the provided CSV file."""
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file at {data_path}")

    df = pd.read_csv(data_path)
    documents = dataframe_to_documents(df)

    vector_store = build_vector_store(documents, config)
    retriever = vector_store.as_retriever(search_kwargs={"k": config.retrieval_top_k})

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an assistant that answers questions using the provided sales data. "
                    "Use only the supplied context when crafting your answers. "
                    "If the context does not contain the answer, say you do not know."
                ),
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )

    llm = ChatOpenAI(model=config.llm_model, temperature=0)

    rag_chain = (
        {
            "context": retriever | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever, vector_store


__all__ = [
    "RagConfig",
    "build_rag_chain",
    "build_vector_store",
    "dataframe_to_documents",
]
