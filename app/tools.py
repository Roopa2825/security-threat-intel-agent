"""Tools the agent can choose to call. The docstring of each tool is what the
LLM reads to decide when to use it, so it must be clear."""
import json

from langchain_core.tools import tool

from app.url_analyzer import analyze_url, is_valid_cve_id

RESULT_CHARS = 1200  # text passed to the LLM per search result (more context vs. more tokens)


@tool
def search_threat_intel(query: str) -> str:
    """Search the threat-intelligence knowledge base (CISA Known Exploited
    Vulnerabilities, MITRE ATT&CK techniques and security advisories).
    Use it for questions about vulnerabilities, attacker techniques,
    detection and remediation."""
    from app.retrieve import search

    hits = search(query, k=5)
    if not hits:
        return "No relevant documents found."
    return "\n\n".join(
        f"[{h['doc_id']}] {h['title']} (source: {h['source']})\n{h['text'][:RESULT_CHARS]}" for h in hits
    )


@tool
def check_url(url: str) -> str:
    """Check a URL or domain for common phishing indicators such as lookalike
    domains, brand names in subdomains, raw IP addresses and '@' tricks.
    Pass the URL exactly as the user wrote it; never correct spelling or digits.
    The URL is analyzed as text only and is never visited."""
    return json.dumps(analyze_url(url), indent=2)


@tool
def lookup_cve(cve_id: str) -> str:
    """Look up an exact CVE ID (for example CVE-2021-44228) in the CISA Known
    Exploited Vulnerabilities catalog and local advisories."""
    if not is_valid_cve_id(cve_id):
        return f"'{cve_id}' is not a valid CVE ID. Expected a format like CVE-2021-44228."
    from app.vectorstore import get_collection

    cve_id = cve_id.strip().upper()
    collection = get_collection()
    # 1) the official CISA KEV record (stored with the CVE ID as its document ID)
    kev = collection.get(where={"doc_id": cve_id}, include=["documents"])
    # 2) any local advisory that mentions this CVE (e.g. detailed fix versions)
    advisories = collection.get(
        where={"source": "local_advisory"},
        where_document={"$contains": cve_id},
        include=["documents"],
    )
    documents = kev["documents"] + advisories["documents"]
    if documents:
        return "\n\n---\n\n".join(documents)
    return (
        f"{cve_id} is not in the CISA Known Exploited Vulnerabilities catalog or local "
        "advisories loaded here. It may exist but not be known to be actively exploited. "
        "Try search_threat_intel."
    )


@tool
def classify_phishing_text(text: str) -> str:
    """Classify the text of an email or message (not a URL) as phishing or legitimate,
    using a DistilBERT model fine-tuned on a public phishing email dataset. Use it when
    the user pastes email or SMS content. Returns a phishing probability."""
    from app.phishing_model import classify_text

    try:
        return json.dumps(classify_text(text), indent=2)
    except FileNotFoundError as exc:
        return f"The phishing text classifier is not available: {exc}"


ALL_TOOLS = [search_threat_intel, check_url, lookup_cve, classify_phishing_text]
