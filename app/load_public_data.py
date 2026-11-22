"""Step 2a: download real public threat-intelligence data and add it to ChromaDB.

Sources (free, no account needed):
  - CISA Known Exploited Vulnerabilities (KEV) catalog: vulnerabilities that
    attackers are actively exploiting, with the required fix.
  - MITRE ATT&CK Enterprise: attacker techniques (e.g. T1566 Phishing).
    ATT&CK is (c) The MITRE Corporation, used under its terms of use.

Run:  python -m app.load_public_data
"""
import json
import urllib.request

CISA_KEV_URL = (
    "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
)
MITRE_ATTACK_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/"
    "enterprise-attack/enterprise-attack.json"
)


def download_json(url: str) -> dict:
    """Download a JSON file and return it as a Python dict."""
    request = urllib.request.Request(url, headers={"User-Agent": "security-threat-intel-agent"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def kev_to_documents(kev: dict) -> list[dict]:
    """Turn each CISA KEV entry into one readable text document."""
    docs = []
    for v in kev.get("vulnerabilities", []):
        text = (
            f"{v['cveID']} - {v.get('vulnerabilityName', '')}\n\n"
            f"Vendor/Product: {v.get('vendorProject', '')} {v.get('product', '')}\n"
            f"Added to CISA KEV: {v.get('dateAdded', '')}\n"
            f"Known ransomware use: {v.get('knownRansomwareCampaignUse', 'Unknown')}\n\n"
            f"Description: {v.get('shortDescription', '')}\n\n"
            f"Required action: {v.get('requiredAction', '')}"
        )
        docs.append({
            "id": v["cveID"],
            "text": text,
            "metadata": {
                "source": "cisa_kev",
                "cve_id": v["cveID"],
                "vendor": v.get("vendorProject", ""),
                "product": v.get("product", ""),
                "date_added": v.get("dateAdded", ""),
                "ransomware": v.get("knownRansomwareCampaignUse", "Unknown"),
            },
        })
    return docs


MAX_ITEMS = 8        # detection / mitigation entries kept per technique
MAX_ITEM_CHARS = 400  # characters kept per entry


def is_active(obj: dict) -> bool:
    return not (obj.get("revoked") or obj.get("x_mitre_deprecated"))


def short(text: str, limit: int = MAX_ITEM_CHARS) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def attack_id(obj: dict) -> str | None:
    return next((ref["external_id"] for ref in obj.get("external_references", [])
                 if ref.get("source_name") == "mitre-attack"), None)


def detection_lines(technique: dict, relationships: list[dict], objects: dict) -> list[str]:
    """How to detect a technique. ATT&CK has stored this in different ways over time:
    a text field on the technique (older versions), data components that 'detect' it,
    and detection strategies with analytics (newer versions). Use whatever exists."""
    lines = []
    if technique.get("x_mitre_detection"):
        lines.append(short(technique["x_mitre_detection"], 1200))
    for rel in relationships:
        if rel.get("relationship_type") != "detects":
            continue
        source = objects.get(rel.get("source_ref"), {})
        if not source or not is_active(source):
            continue
        detail = rel.get("description") or source.get("description", "")
        for analytic_ref in source.get("x_mitre_analytic_refs", [])[:2]:  # newer ATT&CK versions
            analytic = objects.get(analytic_ref, {})
            if analytic.get("description"):
                detail += " " + analytic["description"]
        lines.append(f"{source.get('name', 'Detection')}: {short(detail)}")
    return lines[:MAX_ITEMS]


def mitigation_lines(relationships: list[dict], objects: dict) -> list[str]:
    """Mitigations ('course-of-action' objects) linked to the technique."""
    lines = []
    for rel in relationships:
        if rel.get("relationship_type") != "mitigates":
            continue
        source = objects.get(rel.get("source_ref"), {})
        if source and is_active(source):
            mitigation_id = attack_id(source)
            name = f"{source.get('name', '')} ({mitigation_id})" if mitigation_id else source.get("name", "")
            lines.append(f"{name}: {short(rel.get('description') or source.get('description', ''))}")
    return lines[:MAX_ITEMS]


def attack_to_documents(bundle: dict) -> list[dict]:
    """Turn each active MITRE ATT&CK technique into one text document, including how
    to detect it and how to mitigate it."""
    objects = {obj.get("id", f"_no_id_{i}"): obj for i, obj in enumerate(bundle.get("objects", []))}
    relationships_by_target: dict[str, list[dict]] = {}
    for obj in objects.values():
        if obj.get("type") == "relationship" and is_active(obj):
            relationships_by_target.setdefault(obj.get("target_ref"), []).append(obj)

    docs = []
    for obj in objects.values():
        if obj.get("type") != "attack-pattern" or not is_active(obj):
            continue  # skip non-techniques and techniques MITRE no longer uses
        technique_id = attack_id(obj)
        if not technique_id:
            continue
        tactics = ", ".join(p["phase_name"] for p in obj.get("kill_chain_phases", []))
        related = relationships_by_target.get(obj.get("id"), [])
        parts = [
            f"{technique_id} - {obj.get('name', '')} (MITRE ATT&CK)",
            f"Tactics: {tactics}",
            obj.get("description", ""),
        ]
        detections = detection_lines(obj, related, objects)
        if detections:
            parts.append("Detection:\n" + "\n".join(f"- {line}" for line in detections))
        mitigations = mitigation_lines(related, objects)
        if mitigations:
            parts.append("Mitigations:\n" + "\n".join(f"- {line}" for line in mitigations))
        docs.append({
            "id": technique_id,
            "text": "\n\n".join(p for p in parts if p),
            "metadata": {
                "source": "mitre_attack",
                "technique_id": technique_id,
                "name": obj.get("name", ""),
                "tactics": tactics,
                "has_detection": bool(detections),
                "has_mitigation": bool(mitigations),
            },
        })
    return docs


def main() -> None:
    from app.vectorstore import add_documents  # imported here so tests don't need ChromaDB

    print("Downloading CISA KEV catalog...")
    kev_docs = kev_to_documents(download_json(CISA_KEV_URL))
    print(f"  {len(kev_docs)} vulnerabilities -> {add_documents(kev_docs)} chunks stored")

    print("Downloading MITRE ATT&CK (Enterprise)... this file is large, please wait")
    attack_docs = attack_to_documents(download_json(MITRE_ATTACK_URL))
    with_detection = sum(d["metadata"]["has_detection"] for d in attack_docs)
    with_mitigation = sum(d["metadata"]["has_mitigation"] for d in attack_docs)
    print(f"  {len(attack_docs)} techniques ({with_detection} with detection, "
          f"{with_mitigation} with mitigations) -> {add_documents(attack_docs)} chunks stored")


if __name__ == "__main__":
    main()
