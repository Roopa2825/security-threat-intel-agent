"""Step 2b: semantic search over the vector store (the "R" in RAG).

Run:  python -m app.retrieve "How do I detect Log4Shell exploitation?"
"""
import sys

from app.vectorstore import embed, get_collection


def search(query: str, k: int = 5, source: str | None = None) -> list[dict]:
    """Return the k chunks whose meaning is closest to the query.

    source: optionally restrict to "cisa_kev", "mitre_attack" or "local_advisory".
    """
    where = {"source": source} if source else None
    result = get_collection().query(
        query_embeddings=embed([query]),
        n_results=k,
        where=where,
    )
    hits = []
    for text, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append({
            "text": text,
            "doc_id": meta.get("doc_id"),
            "source": meta.get("source"),
            "score": round(1 - distance, 3),  # cosine similarity: higher = more relevant
        })
    return hits


def main() -> None:
    query = " ".join(sys.argv[1:]) or "How can I tell if a link is phishing?"
    print(f"Query: {query}\n")
    for i, hit in enumerate(search(query), start=1):
        print(f"{i}. [{hit['source']}] {hit['doc_id']}  (score {hit['score']})")
        print("   " + hit["text"][:200].replace("\n", " ") + "...\n")


if __name__ == "__main__":
    main()
