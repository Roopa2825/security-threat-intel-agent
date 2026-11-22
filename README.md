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
python -m app.ingest             # 6 local sample advisories (works offline)
python -m app.load_public_data   # CISA KEV + MITRE ATT&CK (needs internet)
```

## Search it

```bash
python -m app.retrieve "How do I detect Log4Shell exploitation?"
python -m app.retrieve "How do attackers use phishing for initial access?"
```

## Run the tests

```bash
pytest
```

## Data sources

- [CISA Known Exploited Vulnerabilities Catalog](https://www.cisa.gov/known-exploited-vulnerabilities-catalog)
- [MITRE ATT&CK](https://attack.mitre.org/) (c) The MITRE Corporation, used under its terms of use
