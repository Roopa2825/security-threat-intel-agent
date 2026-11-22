"""Step 1: load the local sample advisories (data/advisories/*.md) into ChromaDB.

Run:  python -m app.ingest
"""
from pathlib import Path

from app import config


def load_local_documents(data_dir: str) -> list[dict]:
    """Read every .md file in the data folder as one document."""
    docs = []
    for path in sorted(Path(data_dir).glob("*.md")):
        docs.append({
            "id": path.stem,
            "text": path.read_text(encoding="utf-8"),
            "metadata": {"source": "local_advisory", "file": path.name},
        })
    return docs


def main() -> None:
    from app.vectorstore import add_documents  # imported here so tests don't need ChromaDB

    docs = load_local_documents(config.DATA_DIR)
    print(f"Loaded {len(docs)} local documents")
    print(f"Stored {add_documents(docs)} chunks")


if __name__ == "__main__":
    main()
