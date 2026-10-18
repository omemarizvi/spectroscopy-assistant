"""Vector store wrapper around ChromaDB with local sentence-transformers.

This module owns everything about turning documents into searchable chunks:
  * chunking text into overlapping windows
  * embedding chunks locally (free, no API)
  * persisting them in ChromaDB
  * retrieving the most relevant chunks for a question

Keeping it in one place makes the RAG pipeline easy to follow and audit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions

from app import config


@dataclass
class RetrievedChunk:
    """A single chunk returned from a similarity search."""

    text: str
    source: str      # filename the chunk came from
    distance: float  # cosine distance (lower = more relevant)


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping character windows.

    Overlap preserves sentences that would otherwise be cut in half at a
    chunk boundary, which improves retrieval quality.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


class VectorStore:
    """Thin, well-documented wrapper over a persistent Chroma collection."""

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        # Local embedding model: free, runs on CPU, no network needed.
        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )
        self._collection = self._client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # --- Writing -----------------------------------------------------------
    def add_document(self, filename: str, text: str) -> int:
        """Chunk, embed and store a document. Returns the chunk count.

        Chunk IDs are derived from the file content hash so re-uploading the
        same document is idempotent (it won't create duplicates).
        """
        chunks = _chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        if not chunks:
            return 0

        file_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        ids = [f"{file_hash}-{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename} for _ in chunks]

        self._collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
        return len(chunks)

    # --- Reading -----------------------------------------------------------
    def search(self, query: str, top_k: int = config.TOP_K) -> list[RetrievedChunk]:
        """Return the most relevant chunks for a query, nearest first."""
        if self.count() == 0:
            return []

        result = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self.count()),
        )

        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]

        return [
            RetrievedChunk(text=d, source=m.get("source", "unknown"), distance=dist)
            for d, m, dist in zip(docs, metas, dists)
        ]

    def reset(self) -> None:
        """Delete ALL stored chunks and recreate an empty collection.

        Use this to rebuild the knowledge base from scratch so that documents
        removed from data/ are also removed from the index. (upsert only adds;
        it never deletes, so a plain re-ingest would leave stale chunks.)
        """
        self._client.delete_collection(name=config.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """Number of chunks currently stored."""
        return self._collection.count()

    def list_sources(self) -> list[str]:
        """Distinct filenames currently in the knowledge base."""
        if self.count() == 0:
            return []
        data = self._collection.get(include=["metadatas"])
        sources = {m.get("source", "unknown") for m in data["metadatas"]}
        return sorted(sources)
