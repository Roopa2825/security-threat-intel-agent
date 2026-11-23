"""Unit tests for chunking and data parsing (no internet or ChromaDB needed).

Run:  pytest
"""
from app.chunking import chunk_text
from app.ingest import load_local_documents
from app.load_public_data import attack_to_documents, kev_to_documents


def test_chunk_text_respects_size_and_keeps_content():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(10))
    chunks = chunk_text(text, size=400, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 400 + 50 + 2 for c in chunks)
    assert "Paragraph 9" in chunks[-1]


def test_local_documents_load():
    docs = load_local_documents("data/advisories")
    assert len(docs) == 6
    assert all(d["metadata"]["source"] == "local_advisory" for d in docs)


def test_kev_parsing():
    kev = {"vulnerabilities": [{
        "cveID": "CVE-2021-44228", "vendorProject": "Apache", "product": "Log4j2",
        "vulnerabilityName": "Apache Log4j2 Remote Code Execution",
        "dateAdded": "2021-12-10", "shortDescription": "JNDI lookup RCE.",
        "requiredAction": "Apply updates.", "knownRansomwareCampaignUse": "Known",
    }]}
    docs = kev_to_documents(kev)
    assert docs[0]["id"] == "CVE-2021-44228"
    assert docs[0]["metadata"]["source"] == "cisa_kev"
    assert "Apply updates." in docs[0]["text"]


def test_attack_parsing_skips_revoked_and_deprecated():
    bundle = {"objects": [
        {"type": "attack-pattern", "name": "Phishing", "description": "Adversaries send phishing messages.",
         "external_references": [{"source_name": "mitre-attack", "external_id": "T1566"}],
         "kill_chain_phases": [{"phase_name": "initial-access"}]},
        {"type": "attack-pattern", "name": "Old", "revoked": True,
         "external_references": [{"source_name": "mitre-attack", "external_id": "T9999"}]},
        {"type": "malware", "name": "Not a technique"},
    ]}
    docs = attack_to_documents(bundle)
    assert [d["id"] for d in docs] == ["T1566"]
    assert docs[0]["metadata"]["tactics"] == "initial-access"


def test_tokenize_keeps_cve_ids_together():
    from app.keyword_search import tokenize
    assert "cve-2021-44228" in tokenize("Patch CVE-2021-44228 now")


def test_alias_expansion_adds_official_names():
    from app.keyword_search import expand_query
    assert "cve-2021-44228" in expand_query("Detect Log4Shell")


def test_reciprocal_rank_fusion_rewards_agreement():
    from app.retrieve import reciprocal_rank_fusion
    a = [{"id": "x"}, {"id": "y"}]
    b = [{"id": "y"}, {"id": "z"}]
    fused = reciprocal_rank_fusion([a, b])
    assert fused[0]["id"] == "y"  # ranked by both lists, so it wins


def test_is_correct_matches_id_or_title():
    from app.evaluate_retrieval import is_correct
    hit = {"doc_id": "T1566", "text": "T1566 - Phishing (MITRE ATT&CK)\n\nTactics: initial-access"}
    assert is_correct(hit, ["Phishing"])
    assert not is_correct(hit, ["Brute Force"])


def test_agent_metrics():
    from app.agent_metrics import fact_recall, has_citation, parse_judge_output, sourced_part, tool_correct

    assert tool_correct(["lookup_cve"], ["search_threat_intel", "lookup_cve"])
    assert fact_recall("Upgrade to 2.17.1 to fix JNDI", ["2.17.1", "jndi"]) == 1.0
    assert fact_recall("It is high‑risk", [["high risk", "high-risk"]]) == 1.0
    assert has_citation("Log4Shell [CVE-2021-44228] is critical")
    assert has_citation("Phishing [T1566.002]")
    assert "SBOM" not in sourced_part("Fact.\n**General guidance (not from the knowledge base)**\nUse an SBOM")
    assert parse_judge_output('Here: {"faithfulness": 0.8, "unsupported_claims": ["x"]}')["faithfulness"] == 0.8


def test_citation_rate_only_counts_questions_that_expect_citations():
    from app.evaluate_agent import summarize

    rows = [
        {"tool_correct": True, "fact_recall": 1.0, "has_citation": False, "expects_citation": False,
         "faithfulness": 1.0, "latency_ms": 100},
        {"tool_correct": True, "fact_recall": 1.0, "has_citation": True, "expects_citation": True,
         "faithfulness": 0.5, "latency_ms": 300},
    ]
    summary = summarize(rows)
    assert summary["citation_rate"] == 1.0
    assert summary["faithfulness"] == 0.75


def test_citation_detection_handles_llm_formatting():
    from app.agent_metrics import has_citation

    assert has_citation("Phishing 【T1566】 is common")          # 【T1566】
    assert has_citation("[​CVE‑2021‑44228]")              # zero-width space, non-breaking hyphens
    assert has_citation("[T1566 - Phishing]")
    assert not has_citation("No sources here")


def test_citation_detection_accepts_source_prefix_and_advisory_ids():
    from app.agent_metrics import has_citation

    assert has_citation("Password guessing [source: T1110.001]")
    assert has_citation("Patch now [ cve-2023-4966-citrix-bleed ]")


def test_attack_parsing_adds_detection_and_mitigations():
    technique = {"type": "attack-pattern", "id": "attack-pattern--1", "name": "LSASS Memory",
                 "description": "Adversaries read LSASS memory.",
                 "x_mitre_detection": "Monitor for processes opening LSASS.",
                 "external_references": [{"source_name": "mitre-attack", "external_id": "T1003.001"}]}
    component = {"type": "x-mitre-data-component", "id": "dc--1", "name": "Process Access"}
    mitigation = {"type": "course-of-action", "id": "coa--1", "name": "Credential Access Protection",
                  "external_references": [{"source_name": "mitre-attack", "external_id": "M1043"}]}
    old_mitigation = {"type": "course-of-action", "id": "coa--2", "name": "Old", "revoked": True}
    relationships = [
        {"type": "relationship", "id": "r1", "relationship_type": "detects", "source_ref": "dc--1",
         "target_ref": "attack-pattern--1", "description": "Watch for access to lsass.exe."},
        {"type": "relationship", "id": "r2", "relationship_type": "mitigates", "source_ref": "coa--1",
         "target_ref": "attack-pattern--1", "description": "Enable Credential Guard."},
        {"type": "relationship", "id": "r3", "relationship_type": "mitigates", "source_ref": "coa--2",
         "target_ref": "attack-pattern--1", "description": "Should be skipped."},
    ]
    docs = attack_to_documents({"objects": [technique, component, mitigation, old_mitigation, *relationships]})
    text = docs[0]["text"]
    assert "Detection:" in text and "Monitor for processes opening LSASS." in text
    assert "Process Access: Watch for access to lsass.exe." in text
    assert "Credential Access Protection (M1043): Enable Credential Guard." in text
    assert "Should be skipped" not in text
    assert docs[0]["metadata"]["has_detection"] and docs[0]["metadata"]["has_mitigation"]


def test_question_generation_helpers():
    import random

    from app.evaluate_retrieval import is_correct
    from app.generate_eval import first_sentence, hide_name, kev_retrieval_questions

    text = "Adversaries may use Brute Force (Citation: X) to guess [passwords](https://x). More text."
    assert first_sentence(text) == "Adversaries may use Brute Force to guess passwords."
    assert hide_name("Adversaries may use Brute Force.", "Brute Force") == "Adversaries may use this technique."

    kev = [{"vendor": "A", "product": "One", "cve_id": "CVE-1"},
           {"vendor": "B", "product": "Two", "cve_id": "CVE-2"},
           {"vendor": "B", "product": "Two", "cve_id": "CVE-3"}]
    questions = kev_retrieval_questions(kev, random.Random(0))
    assert [q["expected"] for q in questions] == [["CVE-1"]]  # B/Two is ambiguous, so skipped

    hit = {"doc_id": "T1110.001", "title": "T1110.001 - Password Guessing", "text": ""}
    assert is_correct(hit, ["T1110"]) and not is_correct(hit, ["T1110"], exact=True)


def test_attack_questions_skip_named_and_duplicate_sentences():
    import random

    from app.generate_eval import attack_retrieval_questions

    def doc(sentence):
        return f"title\n\nTactics: x\n\n{sentence} More."

    attack = [
        (doc("Adversaries may use Brute Force to guess many passwords quickly."), {"name": "Brute Force", "technique_id": "T1"}),
        (doc("Adversaries may gather information that can be used during targeting."), {"name": "A", "technique_id": "T2"}),
        (doc("Adversaries may gather information that can be used during targeting."), {"name": "B", "technique_id": "T3"}),
        (doc("Adversaries may read process memory to collect stored credential material."), {"name": "LSASS Memory", "technique_id": "T4"}),
    ]
    questions = attack_retrieval_questions(attack, random.Random(0))
    assert [q["expected"] for q in questions] == [["T4"]]


def test_average_ignores_broken_passes():
    from app.evaluate_agent import average_complete_passes, is_daily_limit

    good = {"questions": 10, "errors": 0, "faithfulness": 0.9, "tool_accuracy": 1.0}
    broken = {"questions": 10, "errors": 6, "faithfulness": 0.5, "tool_accuracy": 1.0}
    summary = average_complete_passes([good, broken])
    assert summary["complete_passes"] == 1 and summary["faithfulness"] == 0.9
    assert is_daily_limit("Rate limit reached ... on tokens per day (TPD): Limit 200000")
    assert not is_daily_limit("Rate limit reached ... on tokens per minute (TPM)")


class _FakeStructured:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def invoke(self, prompt):
        if self.error:
            raise self.error
        return self.result


class _FakeLLM:
    """Stands in for a chat model, so the judge can be tested without an API key."""

    def __init__(self, structured=None, text=""):
        self.structured, self.text = structured, text

    def with_structured_output(self, schema):
        if self.structured is None:
            raise NotImplementedError
        return self.structured

    def invoke(self, prompt):
        class Reply:
            content = self.text
        return Reply()


def test_judge_uses_structured_output():
    from app.agent_metrics import JudgeResult
    from app.evaluate_agent import judge

    llm = _FakeLLM(structured=_FakeStructured(JudgeResult(faithfulness=1.4, unsupported_claims=["x"])))
    result = judge(llm, "Answer [CVE-2021-44228]", ["context"])
    assert result == {"faithfulness": 1.0, "unsupported_claims": ["x"]}  # score clamped to 0..1


def test_judge_falls_back_to_text_when_structured_output_is_unsupported():
    from app.evaluate_agent import judge

    llm = _FakeLLM(structured=None, text='{"faithfulness": 0.7, "unsupported_claims": []}')
    assert judge(llm, "Answer", ["context"])["faithfulness"] == 0.7

    failing = _FakeLLM(structured=_FakeStructured(error=ValueError("bad schema")),
                       text='{"faithfulness": 0.5, "unsupported_claims": []}')
    assert judge(failing, "Answer", ["context"])["faithfulness"] == 0.5
