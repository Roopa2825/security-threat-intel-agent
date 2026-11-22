# Security Threat Intelligence Agent

An AI agent that helps security analysts investigate threats. It uses
retrieval-augmented generation (RAG) over real threat-intelligence data
(CISA Known Exploited Vulnerabilities and MITRE ATT&CK) and will use tools
to check suspicious URLs and look up CVEs.

> Work in progress. Planned: LangGraph agent with tools, FastAPI REST API,
> MLflow evaluation tracking, Docker.

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

## Measure search quality

```bash
python -m app.evaluate_retrieval
```

Reports Hit@5 and MRR for vector-only, hybrid and hybrid + rerank search on a labeled
question set (`eval/retrieval_questions.json`).

## Run the tests

```bash
pytest
```

## Data sources

- [CISA Known Exploited Vulnerabilities Catalog](https://www.cisa.gov/known-exploited-vulnerabilities-catalog)
- [MITRE ATT&CK](https://attack.mitre.org/) (c) The MITRE Corporation, used under its terms of use
