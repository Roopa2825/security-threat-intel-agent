"""Step 4: REST API for the agent, built with FastAPI.

Run:   uvicorn app.api:app --reload
Docs:  open http://127.0.0.1:8000/docs  (interactive Swagger UI)
"""
import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import config  # noqa: F401  (loads .env and quiet settings first)
from app.url_analyzer import analyze_url, is_valid_cve_id

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Warm-up: load the models, database and agent when the server starts, so the first
    request is not slow (cold start). Set WARMUP=0 to skip, e.g. in tests."""
    if os.getenv("WARMUP", "1") == "1":
        try:
            from app.keyword_search import get_keyword_index
            from app.retrieve import get_reranker
            from app.vectorstore import get_collection, get_model

            get_model(), get_reranker(), get_collection(), get_keyword_index()
            get_agent()
            print("Warm-up complete: models, index and agent loaded.")
        except Exception as exc:  # e.g. empty index or missing API key: start anyway
            print(f"Warm-up skipped: {exc}")
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Security Threat Intelligence Agent",
    description="AI agent with RAG over CISA KEV and MITRE ATT&CK, CVE lookup and phishing URL checks.",
    version="0.1.0",
)


# ---------- request and response shapes (validated automatically by Pydantic) ----------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000,
                          examples=["Is http://paypa1.com/login/verify safe?"])


class AskResponse(BaseModel):
    answer: str
    tools_used: list[str]
    latency_ms: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500,
                       examples=["How do I detect Log4Shell exploitation?"])
    k: int = Field(5, ge=1, le=20)
    mode: Literal["vector", "hybrid", "rerank"] = "rerank"
    source: Literal["cisa_kev", "mitre_attack", "local_advisory"] | None = None


class SearchHit(BaseModel):
    doc_id: str | None
    title: str
    source: str | None
    score: float
    text: str


class UrlRequest(BaseModel):
    url: str = Field(..., min_length=3, max_length=2000, examples=["http://paypa1.com/login/verify"])


class UrlResponse(BaseModel):
    url: str
    domain: str | None = None
    risk_score: int
    verdict: str
    reasons: list[str]


# ---------- the agent is built once, the first time it is needed ----------

@lru_cache(maxsize=1)
def get_agent():
    from app.agent import build_agent
    return build_agent()


# ---------- endpoints ----------

@app.get("/health")
def health() -> dict:
    """Simple check that the service is running."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_agent(request: AskRequest) -> AskResponse:
    """Ask the agent a question. It decides which tools to use."""
    from app.agent import ask

    start = time.perf_counter()
    try:
        result = ask(get_agent(), request.question)
    except Exception as exc:  # the LLM provider failed or is unreachable
        raise HTTPException(status_code=502, detail=f"Agent failed: {exc}") from exc
    return AskResponse(
        answer=result["answer"],
        tools_used=result["tools_used"],
        latency_ms=int((time.perf_counter() - start) * 1000),
    )


@app.post("/search", response_model=list[SearchHit])
def search_knowledge_base(request: SearchRequest) -> list[SearchHit]:
    """Search the knowledge base directly (no LLM)."""
    from app.retrieve import search

    hits = search(request.query, k=request.k, mode=request.mode, source=request.source)
    return [SearchHit(**{**h, "score": float(h["score"])}) for h in hits]


@app.post("/check-url", response_model=UrlResponse)
def check_url(request: UrlRequest) -> UrlResponse:
    """Check a URL for phishing indicators (no LLM; the URL is never visited)."""
    return UrlResponse(**analyze_url(request.url))


class TextRequest(BaseModel):
    text: str = Field(..., min_length=20, max_length=20000,
                      examples=["Your account is locked. Verify your password within 24 hours at the link below."])


@app.post("/classify-text")
def classify_email_text(request: TextRequest) -> dict:
    """Phishing probability for email or message text, from the fine-tuned DistilBERT model (no LLM)."""
    from app.phishing_model import classify_text

    try:
        return classify_text(request.text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/cve/{cve_id}")
def get_cve(cve_id: str) -> dict:
    """Look up a CVE in the CISA KEV catalog and local advisories (no LLM)."""
    if not is_valid_cve_id(cve_id):
        raise HTTPException(status_code=400, detail="Invalid CVE ID. Expected a format like CVE-2021-44228.")
    from app.tools import lookup_cve

    return {"cve_id": cve_id.upper(), "result": lookup_cve.invoke({"cve_id": cve_id})}
