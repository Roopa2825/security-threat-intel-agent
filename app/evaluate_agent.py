"""Step 5: evaluate the whole agent and record the results in MLflow.

For every question in eval/agent_questions.json it measures:
  - tool accuracy:  did the agent pick the right tool?
  - fact recall:    does the answer contain the required facts?
  - citation rate:  does the answer cite a CVE or ATT&CK ID?
  - faithfulness:   an LLM judge checks that the sourced part of the answer is
                    supported by what the tools returned (catches hallucinations)
  - latency
It also re-runs the retrieval evaluation (Hit@5, MRR), then logs everything to MLflow.

Run:        python -m app.evaluate_agent
Dashboard:  mlflow ui --backend-store-uri sqlite:///mlflow.db   then open http://127.0.0.1:5000
"""
import json
import os
import statistics
import time

from app import config  # noqa: F401  (loads .env and quiet settings first)
from app.agent_metrics import fact_recall, has_citation, parse_judge_output, sourced_part, tool_correct

QUESTIONS_FILE = "eval/agent_questions.json"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT = "security-threat-intel-agent"
DELAY_SECONDS = float(os.getenv("EVAL_DELAY_SECONDS", "3"))  # stay under free-tier rate limits

JUDGE_PROMPT = """You are grading an AI security assistant for faithfulness.

CONTEXT (everything the assistant's tools returned):
{context}

ANSWER TO GRADE:
{answer}

List every specific factual claim in the answer (versions, dates, settings, CVE or
technique details, risk verdicts) that is NOT supported by the context. Ignore
general advice and formatting. Then give a faithfulness score from 0.0 (mostly
unsupported) to 1.0 (every specific claim is supported).

Reply with JSON only: {{"faithfulness": <number>, "unsupported_claims": [<strings>]}}"""


def model_name(llm) -> str:
    return getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"


def judge(llm, answer: str, tool_outputs: list[str]) -> dict:
    context = "\n\n---\n\n".join(tool_outputs)[:12000] or "(no tool output)"
    reply = llm.invoke(JUDGE_PROMPT.format(context=context, answer=sourced_part(answer)))
    return parse_judge_output(str(reply.content))


def run_question(agent, llm, item: dict) -> dict:
    from app.agent import ask

    start = time.perf_counter()
    try:
        out = ask(agent, item["question"])
    except Exception as exc:  # rate limit or provider error: record it and continue
        return {"question": item["question"], "error": str(exc)[:300]}
    latency_ms = int((time.perf_counter() - start) * 1000)
    time.sleep(DELAY_SECONDS)
    verdict = judge(llm, out["answer"], out["tool_outputs"])
    time.sleep(DELAY_SECONDS)
    return {
        "question": item["question"],
        "expected_tools": ", ".join(item["expected_tools"]),
        "tools_used": ", ".join(out["tools_used"]),
        "tool_correct": tool_correct(out["tools_used"], item["expected_tools"]),
        "fact_recall": round(fact_recall(out["answer"], item.get("must_include", [])), 3),
        "has_citation": has_citation(out["answer"]),
        "faithfulness": verdict["faithfulness"],
        "unsupported_claims": "; ".join(verdict["unsupported_claims"]),
        "latency_ms": latency_ms,
        "answer": out["answer"],
    }


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    faith = [r["faithfulness"] for r in ok if r["faithfulness"] is not None]
    mean = lambda values: round(statistics.mean(values), 3) if values else 0.0  # noqa: E731
    return {
        "questions": len(rows),
        "errors": len(rows) - len(ok),
        "tool_accuracy": mean([float(r["tool_correct"]) for r in ok]),
        "fact_recall": mean([r["fact_recall"] for r in ok]),
        "citation_rate": mean([float(r["has_citation"]) for r in ok]),
        "faithfulness": mean(faith),
        "latency_ms_median": int(statistics.median([r["latency_ms"] for r in ok])) if ok else 0,
    }


def main() -> None:
    import mlflow

    from app.agent import build_agent
    from app.evaluate_retrieval import MODES
    from app.evaluate_retrieval import QUESTIONS_FILE as RETRIEVAL_FILE
    from app.evaluate_retrieval import evaluate as evaluate_retrieval
    from app.llm import get_llm

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)
    with open(RETRIEVAL_FILE, encoding="utf-8") as f:
        retrieval_questions = json.load(f)

    llm = get_llm()
    agent = build_agent()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"{os.getenv('LLM_PROVIDER', 'groq')}:{model_name(llm)}"):
        mlflow.log_params({
            "llm_provider": os.getenv("LLM_PROVIDER", "groq"),
            "llm_model": model_name(llm),
            "embedding_model": config.EMBEDDING_MODEL,
            "reranker_model": config.RERANKER_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
        })

        print("Retrieval evaluation...")
        for mode in MODES:
            r = evaluate_retrieval(mode, retrieval_questions)
            mlflow.log_metrics({f"retrieval_{mode}_hit_at_5": r["hit_at_5"], f"retrieval_{mode}_mrr": r["mrr"]})
            print(f"  {mode:<7} Hit@5 {r['hit_at_5']}  MRR {r['mrr']}")

        print(f"\nAgent evaluation ({len(questions)} questions, this takes a few minutes)...")
        rows = []
        for i, item in enumerate(questions, start=1):
            row = run_question(agent, llm, item)
            rows.append(row)
            status = "ERROR" if "error" in row else (
                f"tool {'OK' if row['tool_correct'] else 'WRONG'}, facts {row['fact_recall']}, "
                f"faithfulness {row['faithfulness']}")
            print(f"  {i:>2}. {status} | {item['question'][:60]}")

        summary = summarize(rows)
        mlflow.log_metrics({k: v for k, v in summary.items()})
        mlflow.log_dict({"summary": summary, "results": rows}, "agent_eval_results.json")
        mlflow.log_table(
            {key: [str(r.get(key, "")) for r in rows]
             for key in ["question", "expected_tools", "tools_used", "tool_correct", "fact_recall",
                         "has_citation", "faithfulness", "unsupported_claims", "latency_ms", "error"]},
            artifact_file="agent_eval_table.json",
        )

    print("\nSummary")
    for key, value in summary.items():
        print(f"  {key:<18} {value}")
    print("\nOpen the dashboard: mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
