"""LangChain RAG layer backed by Chroma.

The corpus is converted to LangChain Document objects and searched through
LangChain's Chroma vector store. Chroma's built-in embedding function is
wrapped in the LangChain Embeddings interface to keep the local MVP lightweight
and avoid introducing a second model download just for embeddings.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "knowledge.jsonl"


class ChromaDefaultEmbeddings(Embeddings):
    """Adapt Chroma's local default embedding function to LangChain."""

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._embedding_fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(value) for value in vector]
            for vector in self._embedding_fn(texts)
        ]

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embedding_fn([text])
        return [float(value) for value in vectors[0]]


def _load_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with DATA_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _documents() -> list[Document]:
    docs: list[Document] = []
    for record in _load_records():
        docs.append(
            Document(
                page_content=record["text"],
                metadata={
                    "vehicle_type": record["vehicle_type"],
                    "topic": record["topic"],
                    "title": record["title"],
                    "source_name": record["source_name"],
                    "source_url": record["source_url"],
                },
            )
        )
    return docs


@lru_cache(maxsize=1)
def _build_vector_store() -> Chroma:
    client = chromadb.Client()
    collection_name = "autosage_knowledge"
    embeddings = ChromaDefaultEmbeddings()
    vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    # Rebuild only when this in-memory collection is empty.
    collection = client.get_or_create_collection(collection_name)
    if collection.count() == 0:
        docs = _documents()
        ids = [f"autosage-{i}" for i in range(len(docs))]
        vector_store.add_documents(documents=docs, ids=ids)
    return vector_store


def retrieve(query: str, vehicle_type: str, n_results: int = 5) -> list[dict[str, Any]]:
    """Retrieve supporting evidence using LangChain's Chroma integration."""
    vector_store = _build_vector_store()
    count = len(_load_records())
    if count == 0:
        return []

    where = {"$or": [{"vehicle_type": vehicle_type}, {"vehicle_type": "both"}]}
    results = vector_store.similarity_search_with_score(
        query,
        k=min(n_results, count),
        filter=where,
    )

    docs: list[dict[str, Any]] = []
    for document, score in results:
        docs.append(
            {
                "text": document.page_content,
                "metadata": document.metadata or {},
                "distance": float(score),
            }
        )
    return docs
