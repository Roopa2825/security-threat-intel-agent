"""Shared access to the embedding model and the ChromaDB collection."""
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from app import config
from app.chunking import chunk_text

BATCH_SIZE = 500  # ChromaDB limits how many items can be added in one call


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_collection():
    """Open (or create) the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def embed(texts: list[str]) -> list[list[float]]:
    """Turn texts into normalized embedding vectors."""
    return get_model().encode(texts, normalize_embeddings=True).tolist()


def add_documents(docs: list[dict]) -> int:
    """Chunk, embed and store documents.

    Each doc is a dict with:
      - "id":       unique document id (e.g. "CVE-2021-44228")
      - "text":     the full text
      - "metadata": extra fields (strings/numbers only), e.g. {"source": "cisa_kev"}
    Returns the number of chunks stored.
    """
    ids, texts, metadatas = [], [], []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            ids.append(f"{doc['id']}::{i}")
            texts.append(chunk)
            metadatas.append({**doc["metadata"], "doc_id": doc["id"], "chunk": i})

    collection = get_collection()
    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            embeddings=embed(texts[start:end]),
        )
    return len(ids)
