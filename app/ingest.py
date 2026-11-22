"""Step 1: load security advisories, split them into chunks, embed them,
and store them in a ChromaDB vector database.

Run:  python -m app.ingest
"""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from app import config


def load_documents(data_dir: str) -> list[dict]:
    """Read every .md file in the data folder."""
    docs = []
    for path in sorted(Path(data_dir).glob("*.md")):
        docs.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    return docs


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks, keeping paragraphs together where possible."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            # start the next chunk with the tail of the previous one (overlap)
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n\n{para}" if tail else para
    if current:
        chunks.append(current)
    return chunks


def main() -> None:
    docs = load_documents(config.DATA_DIR)
    print(f"Loaded {len(docs)} documents")

    model = SentenceTransformer(config.EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    ids, texts, metadatas = [], [], []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            ids.append(f"{doc['source']}::{i}")
            texts.append(chunk)
            metadatas.append({"source": doc["source"], "chunk": i})

    embeddings = model.encode(texts, normalize_embeddings=True).tolist()
    collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    print(f"Stored {len(ids)} chunks in collection '{config.COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
