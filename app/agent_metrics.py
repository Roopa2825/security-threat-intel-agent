"""Scoring helpers for agent evaluation (pure Python, no LLM needed)."""
import json
import re

CITATION_PATTERN = re.compile(r"\[(CVE-\d{4}-\d{4,}|T\d{4}(?:\.\d{3})?)\]", re.IGNORECASE)
GENERAL_GUIDANCE_HEADING = "general guidance (not from the knowledge base)"


def normalize(text: str) -> str:
    """Lowercase and replace the special dashes/spaces LLMs often output with plain ones."""
    text = text.lower()
    for char in ["‐", "‑", "‒", "–", "—"]:
        text = text.replace(char, "-")
    return text.replace(" ", " ").replace("​", "")


def tool_correct(tools_used: list[str], expected_tools: list[str]) -> bool:
    """True if the agent used at least one of the expected tools."""
    return any(tool in tools_used for tool in expected_tools)


def fact_recall(answer: str, must_include: list) -> float:
    """Share of required facts found in the answer. Each item is a string, or a list
    of acceptable alternatives (e.g. ["high risk", "high-risk"])."""
    if not must_include:
        return 1.0
    text = normalize(answer)
    found = 0
    for item in must_include:
        options = item if isinstance(item, list) else [item]
        if any(normalize(option) in text for option in options):
            found += 1
    return found / len(must_include)


def has_citation(answer: str) -> bool:
    return bool(CITATION_PATTERN.search(normalize(answer).upper()))


def sourced_part(answer: str) -> str:
    """The part of the answer before the 'General guidance' section, which must be grounded."""
    index = normalize(answer).find(GENERAL_GUIDANCE_HEADING)
    return answer if index == -1 else answer[:index]


def parse_judge_output(text: str) -> dict:
    """Extract {"faithfulness": float, "unsupported_claims": [...]} from the judge's reply."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"faithfulness": None, "unsupported_claims": ["judge returned no JSON"]}
    try:
        data = json.loads(match.group(0))
        score = float(data.get("faithfulness"))
        return {"faithfulness": max(0.0, min(1.0, score)),
                "unsupported_claims": list(data.get("unsupported_claims", []))}
    except (ValueError, TypeError):
        return {"faithfulness": None, "unsupported_claims": ["judge returned invalid JSON"]}
