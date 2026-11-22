"""Step 2: search the knowledge base (the "R" in RAG).

Three modes, from simple to most accurate:
  - "vector": meaning-based search with embeddings
  - "hybrid": vector + BM25 keyword search, merged with Reciprocal Rank Fusion
  - "rerank": hybrid, then a cross-encoder re-scores the shortlist (default)

Run:  python -m app.retrieve "How do I detect Log4Shell exploitation?"
"""
import sys
from functools import lru_cache

from app import config
from app.keyword_search import get_keyword_index

RRF_K = 60  # standard constant for Reciprocal Rank Fusion


def vector_search(query: str, n: int, source: str | None = None) -> list[dict]:
    from app.vectorstore import embed_query, get_collection  # needs ChromaDB

    result = get_collection().query(
        query_embeddings=[embed_query(query)],
        n_results=n,
        where={"source": source} if source else None,
    )
    return [
        {"id": cid, "text": text, "metadata": meta}
        for cid, text, meta in zip(result["ids"][0], result["documents"][0], result["metadatas"][0])
    ]


def keyword_search(query: str, n: int, source: str | None = None) -> list[dict]:
    index = get_keyword_index()
    hits = []
    for i in index.search(query, n * 3 if source else n):
        meta = index.metadatas[i]
        if source and meta.get("source") != source:
            continue
        hits.append({"id": index.ids[i], "text": index.texts[i], "metadata": meta})
    return hits[:n]


def reciprocal_rank_fusion(result_lists: list[list[dict]]) -> list[dict]:
    """Merge ranked lists: an item ranked high in any list gets a high combined score."""
    scores, items = {}, {}
    for results in result_lists:
        for rank, item in enumerate(results):
            scores[item["id"]] = scores.get(item["id"], 0.0) + 1.0 / (RRF_K + rank + 1)
            items[item["id"]] = item
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [{**items[i], "score": round(scores[i], 4)} for i in ordered]


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.RERANKER_MODEL)


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """Re-score each (question, chunk) pair with a cross-encoder."""
    if not candidates:
        return []
    scores = get_reranker().predict([(query, c["text"]) for c in candidates])
    rescored = [{**c, "score": round(float(s), 3)} for c, s in zip(candidates, scores)]
    return sorted(rescored, key=lambda c: c["score"], reverse=True)


def one_chunk_per_document(results: list[dict]) -> list[dict]:
    """Keep only the best chunk of each document, so results are more varied."""
    seen, unique = set(), []
    for r in results:
        doc_id = r["metadata"].get("doc_id")
        if doc_id not in seen:
            seen.add(doc_id)
            unique.append(r)
    return unique


def search(query: str, k: int = 5, mode: str = "rerank", source: str | None = None) -> list[dict]:
    """Return the k most relevant chunks for the query."""
    n = config.CANDIDATES
    if mode == "vector":
        results = vector_search(query, n, source)
        for rank, r in enumerate(results):
            r["score"] = rank + 1
    else:
        results = reciprocal_rank_fusion(
            [vector_search(query, n, source), keyword_search(query, n, source)]
        )
        if mode == "rerank":
            results = rerank(query, results[:n])
    return [
        {
            "text": r["text"],
            "doc_id": r["metadata"].get("doc_id"),
            "source": r["metadata"].get("source"),
            "score": r["score"],
        }
        for r in one_chunk_per_document(results)[:k]
    ]


def main() -> None:
    query = " ".join(sys.argv[1:]) or "How can I tell if a link is phishing?"
    print(f"Query: {query}\n")
    for i, hit in enumerate(search(query), start=1):
        print(f"{i}. [{hit['source']}] {hit['doc_id']}  (score {hit['score']})")
        print("   " + hit["text"][:200].replace("\n", " ") + "...\n")


if __name__ == "__main__":
    main()
