"""Rebuild the whole knowledge base from scratch.

Use this the first time, and whenever you change the embedding model.
Run:  python -m app.build_index
"""
from app import ingest, load_public_data
from app.vectorstore import reset_collection


def main() -> None:
    print("Clearing the old index...")
    reset_collection()
    ingest.main()
    load_public_data.main()
    print("Done. Try: python -m app.retrieve \"How do I detect Log4Shell exploitation?\"")


if __name__ == "__main__":
    main()
