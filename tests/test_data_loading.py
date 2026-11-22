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
