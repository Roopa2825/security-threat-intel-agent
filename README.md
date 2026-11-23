# Security Threat Intelligence Agent

An AI agent that helps security analysts investigate threats. It uses
retrieval-augmented generation (RAG) over real threat-intelligence data
(CISA Known Exploited Vulnerabilities and MITRE ATT&CK) and will use tools
to check suspicious URLs and look up CVEs.


## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env
```

## Build the knowledge base

```bash
python -m app.build_index        # clears the index, then loads local samples + CISA KEV + MITRE ATT&CK
```

Run it again whenever you change the embedding model.

## Search it

```bash
python -m app.retrieve "How do I detect Log4Shell exploitation?"
python -m app.retrieve "How do attackers use phishing for initial access?"
```

## How search works

1. **Vector search** with `BAAI/bge-small-en-v1.5` embeddings finds chunks with similar meaning.
2. **BM25 keyword search** finds exact terms such as product names and CVE IDs, and expands
   nicknames (e.g. "Log4Shell" -> "Log4j CVE-2021-44228").
3. **Reciprocal Rank Fusion** merges both result lists.
4. A **cross-encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores the shortlist.

## Ask the agent

Set `LLM_PROVIDER` and its API key in `.env` (Groq and Gemini have free tiers;
Ollama runs locally with no key), then:

```bash
python -m app.agent "Is http://paypa1.com/login/verify safe?"
python -m app.agent "What is CVE-2021-44228 and how do I fix it?"
python -m app.agent            # interactive chat
```

The agent is a LangGraph loop. The LLM chooses between three tools:

| Tool | What it does |
|---|---|
| `search_threat_intel` | Hybrid RAG search over CISA KEV, MITRE ATT&CK and local advisories |
| `lookup_cve` | Exact lookup of a CVE ID in the CISA KEV catalog |
| `check_url` | Rule-based phishing checks (typosquatting, lookalike characters, brand-in-subdomain, IP hosts, `@` tricks). The URL is never visited. |

## REST API (FastAPI)

```bash
uvicorn app.api:app --reload
```

Open http://127.0.0.1:8000/docs for interactive API docs.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service health check |
| POST | `/ask` | Ask the agent a question (`{"question": "..."}`) |
| POST | `/search` | Search the knowledge base directly (no LLM) |
| POST | `/check-url` | Phishing checks for a URL (no LLM) |
| GET | `/cve/{cve_id}` | Look up a CVE in CISA KEV and local advisories |

Example:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"Is http://paypa1.com/login/verify safe?\"}"
```

## Run with Docker

```bash
docker compose build                                   # first build takes 10-20 minutes
docker compose run --rm api python -m app.build_index  # build the knowledge base once (stored in a volume)
docker compose up                                      # start the API on http://127.0.0.1:8000/docs
```

- API keys are read from your local `.env` at runtime and are never copied into the image.
- The vector database lives in a Docker volume, so it survives restarts.
- Models are downloaded at build time and loaded at startup (warm-up), so the first request is fast.
- The container runs as a non-root user and has a health check on `/health`.

## Measure search quality

```bash
python -m app.evaluate_retrieval
```

Reports Hit@5 and MRR for vector-only, hybrid and hybrid + rerank search on a labeled
question set (`eval/retrieval_questions.json`).

## Evaluate the agent (MLflow)

```bash
python -m app.evaluate_agent
mlflow ui --backend-store-uri sqlite:///mlflow.db     # open http://127.0.0.1:5000
```

Runs the labeled questions in `eval/agent_questions.json` through the agent and logs to MLflow:

| Metric | Meaning |
|---|---|
| `tool_accuracy` | Share of questions where the agent chose the expected tool |
| `fact_recall` | Share of required facts present in the answer |
| `citation_rate` | Share of answers that cite a CVE or ATT&CK ID |
| `faithfulness` | LLM-as-judge score: are the answer's specific claims supported by tool outputs? |
| `latency_ms_median` | Median time per question |
| `retrieval_*_hit_at_5`, `retrieval_*_mrr` | Retrieval quality for each search mode |

Each run records the model and settings, so different models or prompts can be compared side by side.

```bash
python -m app.show_eval    # per-question scores, unsupported claims, and latest vs previous run
```

## Larger, held-out test sets

The hand-written question sets were also used while improving the agent, so their scores
can be optimistic. To get honest numbers, generate fresh questions from the data itself:

```bash
python -m app.generate_eval
python -m app.evaluate_retrieval eval/retrieval_questions_generated.json
python -m app.evaluate_agent --questions eval/agent_questions_heldout.json --retrieval-questions eval/retrieval_questions_generated.json --repeats 2
```

- About 150 retrieval questions (CISA KEV products and MITRE ATT&CK behaviors, with the
  technique name hidden) scored against exact document IDs.
- About 30 held-out agent questions (CVE lookups, detection questions and URL checks).
- `--repeats 2` runs the agent questions twice and reports the average and the spread,
  because LLM answers vary from run to run.

## Run the tests

```bash
pytest
```

## Data sources

- [CISA Known Exploited Vulnerabilities Catalog](https://www.cisa.gov/known-exploited-vulnerabilities-catalog)
- [MITRE ATT&CK](https://attack.mitre.org/) (c) The MITRE Corporation, used under its terms of use
