"""Show the latest agent evaluation from MLflow: per-question scores, the claims the
judge marked as unsupported, and a comparison with the previous run.

Run:  python -m app.show_eval
"""
import json
import os

from app import config  # noqa: F401  (loads .env)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT = "security-threat-intel-agent"
COMPARE = ["tool_accuracy", "fact_recall", "citation_rate", "faithfulness", "latency_ms_median",
           "retrieval_rerank_hit_at_5", "retrieval_rerank_mrr"]


def main() -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=TRACKING_URI)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        print("No experiment found. Run: python -m app.evaluate_agent")
        return
    runs = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=2)
    if not runs:
        print("No runs found. Run: python -m app.evaluate_agent")
        return

    latest = runs[0]
    path = mlflow.artifacts.download_artifacts(
        run_id=latest.info.run_id, artifact_path="agent_eval_results.json", tracking_uri=TRACKING_URI
    )
    with open(path, encoding="utf-8") as f:
        results = json.load(f)["results"]

    print(f"Latest run: {latest.info.run_name}\n")
    for i, r in enumerate(results, start=1):
        if "error" in r:
            print(f"{i:>2}. ERROR {r['question']}\n    {r['error']}\n")
            continue
        print(f"{i:>2}. faithfulness {r['faithfulness']} | tools: {r['tools_used']} | {r['question']}")
        if r.get("unsupported_claims"):
            for claim in r["unsupported_claims"].split("; "):
                print(f"      - unsupported: {claim}")
    print()

    if len(runs) == 2:
        previous = runs[1]
        print(f"{'Metric':<28}{'previous':>10}{'latest':>10}")
        for name in COMPARE:
            old, new = previous.data.metrics.get(name), latest.data.metrics.get(name)
            fmt = lambda v: "-" if v is None else f"{v:.3g}"  # noqa: E731
            print(f"{name:<28}{fmt(old):>10}{fmt(new):>10}")


if __name__ == "__main__":
    main()
