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
            python -m app.evaluate_agent --questions eval/agent_questions_heldout.json \
                --retrieval-questions eval/retrieval_questions_generated.json --repeats 2
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
DELAY_SECONDS = float(os.getenv("EVAL_DELAY_SECONDS", "5"))  # stay under free-tier rate limits

JUDGE_PROMPT = """You are grading an AI security assistant for faithfulness.

CONTEXT (everything the assistant's tools returned):
{context}

ANSWER TO GRADE:
{answer}

List every specific factual claim in the answer (versions, dates, settings, CVE or
technique details, risk verdicts) that is NOT supported by the context.
- Paraphrases and summaries of the context count as supported.
- A risk verdict or reason returned by the URL-checking tool counts as supported.
- Source IDs in square brackets are references, not claims.
- Ignore general advice and formatting.
Then give a faithfulness score from 0.0 (mostly unsupported) to 1.0 (every
specific claim is supported).

Reply with JSON only: {{"faithfulness": <number>, "unsupported_claims": [<strings>]}}"""


def model_name(llm) -> str:
    return getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"


def judge(llm, answer: str, tool_outputs: list[str]) -> dict:
    context = "\n\n---\n\n".join(tool_outputs)[:12000] or "(no tool output)"
    prompt = JUDGE_PROMPT.format(context=context, answer=sourced_part(answer))
    for attempt in range(1, 4):  # retry on rate limits
        try:
            return parse_judge_output(str(llm.invoke(prompt).content))
        except Exception as exc:
            if attempt == 3:
                return {"faithfulness": None, "unsupported_claims": [f"judge failed: {str(exc)[:120]}"]}
            time.sleep(10 * attempt)


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
        "expects_citation": item.get("expects_citation", True),
        "has_citation": has_citation(out["answer"]),
        "faithfulness": verdict["faithfulness"],
        "unsupported_claims": "; ".join(verdict["unsupported_claims"]),
        "latency_ms": latency_ms,
        "answer": out["answer"],
    }


def is_daily_limit(error: str) -> bool:
    """Free tiers also cap tokens per day; retrying will not help until the next day."""
    text = error.lower()
    return "per day" in text or "tokens per day" in text or "(tpd)" in text or "requests per day" in text


def average_complete_passes(summaries: list[dict], max_error_share: float = 0.2) -> dict:
    """Average the passes that mostly succeeded; a pass broken by rate limits would distort it."""
    complete = [s for s in summaries if s["questions"] and s["errors"] / s["questions"] <= max_error_share]
    if not complete:
        return {"complete_passes": 0}
    summary = {key: round(statistics.mean(s[key] for s in complete), 3) for key in complete[0]}
    summary["complete_passes"] = len(complete)
    if len(complete) > 1:
        summary["faithfulness_std"] = round(statistics.stdev(s["faithfulness"] for s in complete), 3)
    return summary


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    faith = [r["faithfulness"] for r in ok if r["faithfulness"] is not None]
    mean = lambda values: round(statistics.mean(values), 3) if values else 0.0  # noqa: E731
    return {
        "questions": len(rows),
        "errors": len(rows) - len(ok),
        "tool_accuracy": mean([float(r["tool_correct"]) for r in ok]),
        "fact_recall": mean([r["fact_recall"] for r in ok]),
        "citation_rate": mean([float(r["has_citation"]) for r in ok if r.get("expects_citation", True)]),
        "faithfulness": mean(faith),
        "latency_ms_median": int(statistics.median([r["latency_ms"] for r in ok])) if ok else 0,
    }


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the agent and log results to MLflow.")
    parser.add_argument("--questions", default=QUESTIONS_FILE, help="agent question file")
    parser.add_argument("--retrieval-questions", default=None, help="retrieval question file")
    parser.add_argument("--repeats", type=int, default=1, help="run the agent questions N times and average")
    return parser.parse_args()


def main() -> None:
    import mlflow

    from app.agent import build_agent
    from app.evaluate_retrieval import MODES
    from app.evaluate_retrieval import QUESTIONS_FILE as RETRIEVAL_FILE
    from app.evaluate_retrieval import evaluate as evaluate_retrieval
    from app.llm import get_llm

    args = parse_args()
    retrieval_file = args.retrieval_questions or RETRIEVAL_FILE
    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)
    with open(retrieval_file, encoding="utf-8") as f:
        retrieval_questions = json.load(f)

    # The judge can use a different model (set JUDGE_MODEL in .env). On free tiers, limits are
    # usually per model, so this spreads the load; a different judge also avoids self-grading.
    llm = get_llm(os.getenv("JUDGE_MODEL") or None)
    agent = build_agent()
    question_set = os.path.splitext(os.path.basename(args.questions))[0]

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    agent_model = os.getenv("LLM_MODEL") or "auto"
    with mlflow.start_run(run_name=f"{os.getenv('LLM_PROVIDER', 'groq')}:{agent_model}:{question_set}"):
        mlflow.log_params({
            "llm_provider": os.getenv("LLM_PROVIDER", "groq"),
            "agent_model": agent_model,
            "judge_model": model_name(llm),
            "embedding_model": config.EMBEDDING_MODEL,
            "reranker_model": config.RERANKER_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "agent_questions_file": args.questions,
            "retrieval_questions_file": retrieval_file,
            "repeats": args.repeats,
        })

        print(f"Retrieval evaluation ({len(retrieval_questions)} questions)...")
        for mode in MODES:
            r = evaluate_retrieval(mode, retrieval_questions)
            mlflow.log_metrics({f"retrieval_{mode}_hit_at_5": r["hit_at_5"], f"retrieval_{mode}_mrr": r["mrr"]})
            print(f"  {mode:<7} Hit@5 {r['hit_at_5']}  MRR {r['mrr']}")

        rows, summaries = [], []
        for repeat in range(1, args.repeats + 1):
            print(f"\nAgent evaluation, pass {repeat}/{args.repeats} ({len(questions)} questions)...")
            pass_rows = []
            daily_limit_hit = False
            for i, item in enumerate(questions, start=1):
                row = run_question(agent, llm, item)
                row["repeat"] = repeat
                pass_rows.append(row)
                status = "ERROR" if "error" in row else (
                    f"tool {'OK' if row['tool_correct'] else 'WRONG'}, facts {row['fact_recall']}, "
                    f"faithfulness {row['faithfulness']}")
                print(f"  {i:>2}. {status} | {item['question'][:60]}")
                if "error" in row and is_daily_limit(row["error"]):
                    print("\n  Daily token limit reached; stopping. Try again tomorrow or use another provider.")
                    daily_limit_hit = True
                    break
            pass_summary = summarize(pass_rows)
            mlflow.log_metrics({k: v for k, v in pass_summary.items()}, step=repeat)
            pass_summary["questions"] = len(questions)  # count unanswered questions as missing
            pass_summary["errors"] = len(questions) - (len(pass_rows) - pass_summary["errors"])
            summaries.append(pass_summary)
            rows += pass_rows
            if daily_limit_hit:
                break

        # Average only complete passes; the spread shows how much results vary between runs.
        summary = average_complete_passes(summaries)
        mlflow.log_metrics({f"avg_{k}": v for k, v in summary.items()})
        mlflow.log_dict({"summary": summary, "per_pass": summaries, "results": rows}, "agent_eval_results.json")
        mlflow.log_table(
            {key: [str(r.get(key, "")) for r in rows]
             for key in ["repeat", "question", "expected_tools", "tools_used", "tool_correct", "fact_recall",
                         "has_citation", "faithfulness", "unsupported_claims", "latency_ms", "error"]},
            artifact_file="agent_eval_table.json",
        )

    print(f"\nSummary (average of {summary.get('complete_passes', 0)} complete pass(es) out of {len(summaries)})")
    for key, value in summary.items():
        print(f"  {key:<18} {value}")
    print("\nOpen the dashboard: mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
