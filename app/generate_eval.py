"""Generate larger, held-out test sets from the knowledge base itself.

Creates:
  eval/retrieval_questions_generated.json  ~150 search questions with exact expected IDs
  eval/agent_questions_heldout.json        ~30 agent questions never used for tuning

A fixed random seed makes the output reproducible for the same data version.
Run (after python -m app.build_index):  python -m app.generate_eval
"""
import json
import random
import re
from collections import Counter

SEED = 42
N_KEV_RETRIEVAL = 75
N_ATTACK_RETRIEVAL = 75
N_AGENT_CVE = 10
N_AGENT_DETECTION = 10


def clean_text(text: str) -> str:
    """Remove ATT&CK citation markers and markdown links, and collapse whitespace."""
    text = re.sub(r"\(Citation:[^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # [text](url) -> text
    return " ".join(text.split())


def first_sentence(text: str) -> str:
    match = re.match(r"(.+?[.!?])(\s|$)", clean_text(text))
    return match.group(1) if match else clean_text(text)


def hide_name(sentence: str, name: str) -> str:
    """Replace the technique's own name, so the question does not give away the answer."""
    return re.sub(re.escape(name), "this technique", sentence, flags=re.IGNORECASE) if name else sentence


def technique_description(document: str) -> str:
    """Technique documents are: title, tactics line, description, then Detection/Mitigations."""
    parts = document.split("\n\n")
    return parts[2] if len(parts) > 2 else ""


def kev_retrieval_questions(kev: list[dict], rng: random.Random) -> list[dict]:
    """Ask about vendor+product pairs that have exactly one CVE in the catalog,
    so there is a single correct answer."""
    counts = Counter((m["vendor"], m["product"]) for m in kev)
    unique = [m for m in kev if counts[(m["vendor"], m["product"])] == 1]
    sample = rng.sample(unique, min(N_KEV_RETRIEVAL, len(unique)))
    return [{
        "question": f"Is there an actively exploited vulnerability in {m['vendor']} {m['product']}?",
        "expected": [m["cve_id"]],
        "exact": True,
        "kind": "kev",
    } for m in sample]


def attack_retrieval_questions(attack: list[tuple[str, dict]], rng: random.Random) -> list[dict]:
    """Describe a technique's behavior (with its name hidden) and expect its exact ID."""
    candidates = []
    for document, meta in attack:
        sentence = hide_name(first_sentence(technique_description(document)), meta["name"])
        if len(sentence) >= 40:
            candidates.append({
                "question": f"Which MITRE ATT&CK technique matches this behavior: {sentence}",
                "expected": [meta["technique_id"]],
                "exact": True,
                "kind": "attack",
            })
    return rng.sample(candidates, min(N_ATTACK_RETRIEVAL, len(candidates)))


def url_questions() -> list[dict]:
    """Phishing look-alikes and genuine sites. The expected verdict comes from the
    URL analyzer, so this tests that the agent calls the tool and relays it faithfully."""
    from app.url_analyzer import analyze_url

    urls = [
        "http://amaz0n-billing.com/verify", "https://netf1ix.com/account/update",
        "http://linkedin.com.jobs-alert.example.org/login", "https://dropbox-share.click/confirm",
        "http://appIe-id-unlock.com/signin",
        "https://www.netflix.com", "https://www.linkedin.com/feed", "https://www.adobe.com",
        "https://docs.github.com", "https://www.amazon.com/orders",
    ]
    questions = []
    for url in urls:
        verdict = analyze_url(url)["verdict"]
        questions.append({
            "question": f"Is this link safe to click: {url}",
            "expected_tools": ["check_url"],
            "must_include": [[verdict, verdict.replace(" ", "-")]],
            "expects_citation": False,
        })
    return questions


def agent_questions(kev: list[dict], attack: list[tuple[str, dict]], rng: random.Random) -> list[dict]:
    cve_items = rng.sample(kev, min(N_AGENT_CVE, len(kev)))
    detectable = [meta for _, meta in attack if meta.get("has_detection")]
    detection_items = rng.sample(detectable, min(N_AGENT_DETECTION, len(detectable)))
    questions = [{
        "question": f"What is {m['cve_id']} and what should we do about it?",
        "expected_tools": ["lookup_cve"],
        "must_include": [m["cve_id"], m["product"]],
        "expects_citation": True,
    } for m in cve_items]
    questions += [{
        "question": f"How can we detect the MITRE ATT&CK technique {m['name']}?",
        "expected_tools": ["search_threat_intel"],
        "must_include": [m["technique_id"]],
        "expects_citation": True,
    } for m in detection_items]
    return questions + url_questions()


def main() -> None:
    from app.vectorstore import get_collection

    collection = get_collection()
    kev = collection.get(where={"$and": [{"source": "cisa_kev"}, {"chunk": 0}]}, include=["metadatas"])["metadatas"]
    attack_data = collection.get(where={"$and": [{"source": "mitre_attack"}, {"chunk": 0}]},
                                 include=["documents", "metadatas"])
    attack = list(zip(attack_data["documents"], attack_data["metadatas"]))
    if not kev or not attack:
        print("Knowledge base is empty. Run: python -m app.build_index")
        return

    rng = random.Random(SEED)
    retrieval = kev_retrieval_questions(kev, rng) + attack_retrieval_questions(attack, rng)
    agent = agent_questions(kev, attack, rng)

    with open("eval/retrieval_questions_generated.json", "w", encoding="utf-8") as f:
        json.dump(retrieval, f, indent=1)
    with open("eval/agent_questions_heldout.json", "w", encoding="utf-8") as f:
        json.dump(agent, f, indent=1)
    print(f"Wrote {len(retrieval)} retrieval questions -> eval/retrieval_questions_generated.json")
    print(f"Wrote {len(agent)} agent questions -> eval/agent_questions_heldout.json")
    print("\nExamples:")
    for q in retrieval[:2] + retrieval[-2:]:
        print("  -", q["question"][:140], "->", q["expected"])


if __name__ == "__main__":
    main()
