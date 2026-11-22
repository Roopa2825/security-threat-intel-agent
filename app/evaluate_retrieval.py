"""Measure search quality on a fixed set of questions with known answers.

Metrics (higher is better, max 1.0):
  - Hit@5: share of questions where a correct document appears in the top 5
  - MRR:   Mean Reciprocal Rank; 1.0 if the correct document is always ranked first,
           0.5 if it is usually second, and so on

Run:  python -m app.evaluate_retrieval
"""
import json

from app.retrieve import search

QUESTIONS_FILE = "eval/retrieval_questions.json"
MODES = ["vector", "hybrid", "rerank"]


def is_correct(hit: dict, expected: list[str]) -> bool:
    """A result counts as correct if its ID or title mentions an expected term."""
    title = hit.get("title") or hit["text"].split("\n", 1)[0]
    haystack = f"{hit['doc_id']} {title}".lower()
    return any(term.lower() in haystack for term in expected)


def first_correct_rank(hits: list[dict], expected: list[str]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if is_correct(hit, expected):
            return rank
    return None


def evaluate(mode: str, questions: list[dict], k: int = 5) -> dict:
    hits_at_k, reciprocal_ranks, misses = 0, [], []
    for q in questions:
        rank = first_correct_rank(search(q["question"], k=k, mode=mode), q["expected"])
        if rank:
            hits_at_k += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)
            misses.append(q["question"])
    n = len(questions)
    return {"hit_at_5": round(hits_at_k / n, 3), "mrr": round(sum(reciprocal_ranks) / n, 3), "misses": misses}


def main() -> None:
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Evaluating {len(questions)} questions\n")
    print(f"{'Mode':<8} {'Hit@5':>6} {'MRR':>6}")
    results = {}
    for mode in MODES:
        results[mode] = evaluate(mode, questions)
        print(f"{mode:<8} {results[mode]['hit_at_5']:>6} {results[mode]['mrr']:>6}")
    for mode, r in results.items():
        if r["misses"]:
            print(f"\nMissed by {mode}:")
            for m in r["misses"]:
                print(f"  - {m}")


if __name__ == "__main__":
    main()
