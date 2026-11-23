"""Scoring helpers for agent evaluation (pure Python, no LLM needed)."""
import json
import re

from pydantic import BaseModel, Field

# A source ID in square brackets, optionally after "source:", e.g. [T1566], [source: T1110.001]
# or [cve-2023-4966-citrix-bleed] (a local advisory ID that starts with a CVE ID).
CITATION_PATTERN = re.compile(r"\[\s*(?:source:\s*)?(CVE-\d{4}-\d{4,}|T\d{4}(?:\.\d{3})?)", re.IGNORECASE)
GENERAL_GUIDANCE_HEADING = "general guidance (not from the knowledge base)"


def normalize(text: str) -> str:
    """Lowercase, and replace the special characters LLMs often output with plain ones."""
    text = text.lower()
    for code in (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2212):  # dash variants
        text = text.replace(chr(code), "-")
    for code in (0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF):  # invisible characters
        text = text.replace(chr(code), "")
    for code in (0x00A0, 0x202F):  # non-breaking spaces
        text = text.replace(chr(code), " ")
    for fancy, plain in ((0x3010, "["), (0x3011, "]"), (0xFF3B, "["), (0xFF3D, "]")):
        text = text.replace(chr(fancy), plain)  # e.g. lenticular brackets -> [ ]
    return text


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
    return bool(CITATION_PATTERN.search(normalize(answer)))


def sourced_part(answer: str) -> str:
    """The part of the answer before the 'General guidance' section, which must be grounded."""
    index = normalize(answer).find(GENERAL_GUIDANCE_HEADING)
    return answer if index == -1 else answer[:index]


class JudgeResult(BaseModel):
    """The format the judge must reply in (structured output), instead of free text."""

    faithfulness: float = Field(description="0.0 (mostly unsupported) to 1.0 (every specific claim supported)")
    unsupported_claims: list[str] = Field(default_factory=list,
                                          description="Specific claims not supported by the context")


def judge_result_to_dict(result: JudgeResult) -> dict:
    return {"faithfulness": max(0.0, min(1.0, float(result.faithfulness))),
            "unsupported_claims": list(result.unsupported_claims)}


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
