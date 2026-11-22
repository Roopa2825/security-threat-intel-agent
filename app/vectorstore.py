"""Shared access to the embedding model and the ChromaDB collection."""
from functools import lru_cache

from app import config  # first, so its environment settings apply before the libraries load
from app.chunking import chunk_text

import chromadb
from sentence_transformers import SentenceTransformer

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
    """Turn document texts into normalized embedding vectors."""
    return get_model().encode(texts, normalize_embeddings=True).tolist()


def embed_query(query: str) -> list[float]:
    """Embed a search query (BGE models expect an instruction prefix on queries)."""
    return embed([config.QUERY_PREFIX + query])[0]


def reset_collection() -> None:
    """Delete all stored chunks (needed after changing the embedding model)."""
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass  # collection did not exist yet
    get_collection.cache_clear()


def get_all_chunks() -> tuple[list[str], list[str], list[dict]]:
    """Return every stored chunk (ids, texts, metadata), used for keyword search."""
    data = get_collection().get(include=["documents", "metadatas"])
    return data["ids"], data["documents"], data["metadatas"]


def add_documents(docs: list[dict]) -> int:
    """Chunk, embed and store documents.

    Each doc is a dict with:
      - "id":       unique document id (e.g. "CVE-2021-44228")
      - "text":     the full text; its first line is used as the document title
      - "metadata": extra fields (strings/numbers only), e.g. {"source": "cisa_kev"}
    Every chunk after the first gets the title added on top ("contextual chunk
    header"), so a chunk from the middle of a document still says what it is about.
    Returns the number of chunks stored.
    """
    ids, texts, metadatas = [], [], []
    for doc in docs:
        title = doc["text"].strip().split("\n", 1)[0].lstrip("# ").strip()
        for i, chunk in enumerate(chunk_text(doc["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            ids.append(f"{doc['id']}::{i}")
            texts.append(chunk if i == 0 else f"{title}\n\n{chunk}")
            metadatas.append({**doc["metadata"], "doc_id": doc["id"], "title": title, "chunk": i})

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
