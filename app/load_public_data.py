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


def attack_to_documents(bundle: dict) -> list[dict]:
    """Turn each active MITRE ATT&CK technique into one text document."""
    docs = []
    for obj in bundle.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue  # skip techniques MITRE no longer uses
        technique_id = next(
            (ref["external_id"] for ref in obj.get("external_references", [])
             if ref.get("source_name") == "mitre-attack"),
            None,
        )
        if not technique_id:
            continue
        tactics = ", ".join(p["phase_name"] for p in obj.get("kill_chain_phases", []))
        text = (
            f"{technique_id} - {obj.get('name', '')} (MITRE ATT&CK)\n\n"
            f"Tactics: {tactics}\n\n"
            f"{obj.get('description', '')}"
        )
        docs.append({
            "id": technique_id,
            "text": text,
            "metadata": {
                "source": "mitre_attack",
                "technique_id": technique_id,
                "name": obj.get("name", ""),
                "tactics": tactics,
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
    print(f"  {len(attack_docs)} techniques -> {add_documents(attack_docs)} chunks stored")


if __name__ == "__main__":
    main()
