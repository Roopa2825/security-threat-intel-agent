"""Rule-based phishing checks for a URL (pure Python, no network calls).

The URL is only analyzed as text; it is never visited, so checking a
malicious link is safe.
"""
import ipaddress
import re
from urllib.parse import urlparse

# Brands that phishing sites often imitate, with their real registered domains.
BRANDS = {
    "paypal": "paypal.com", "microsoft": "microsoft.com", "apple": "apple.com",
    "google": "google.com", "amazon": "amazon.com", "netflix": "netflix.com",
    "facebook": "facebook.com", "instagram": "instagram.com", "linkedin": "linkedin.com",
    "chase": "chase.com", "wellsfargo": "wellsfargo.com", "bankofamerica": "bankofamerica.com",
    "office365": "office.com", "outlook": "outlook.com", "dropbox": "dropbox.com",
    "docusign": "docusign.com", "adobe": "adobe.com", "github": "github.com",
}
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rebrand.ly", "cutt.ly"}
SUSPICIOUS_WORDS = ["login", "signin", "verify", "update", "secure", "account", "password", "confirm", "unlock", "billing"]
RISKY_TLDS = {"zip", "mov", "xyz", "top", "click", "country", "gq", "tk", "ml", "cf", "work", "support"}
# Characters attackers swap in to make a domain look like a brand.
LOOKALIKE_SWAPS = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "4": "a", "7": "t", "8": "b"})


def registered_domain(host: str) -> str:
    """Approximate the registered domain as the last two labels (e.g. evil.example.net
    -> example.net). Simplification: multi-part suffixes like .co.uk are not handled."""
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance: how many single-character edits turn a into b."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def is_lookalike(word: str, brand: str) -> bool:
    """True if a word imitates a brand: lookalike characters (micros0ft), a capital I used
    as a lowercase l (appIe, which becomes 'appie' once lowercased), or one typo (paypa1)."""
    if not word or word == brand:
        return False
    normalized = word.translate(LOOKALIKE_SWAPS)
    candidates = {normalized}
    if "l" in brand:
        candidates.add(normalized.replace("i", "l"))
    if brand in candidates:
        return True
    return len(brand) >= 5 and edit_distance(word, brand) <= 1


def analyze_url(url: str) -> dict:
    """Return a 0-100 risk score, a verdict and the reasons behind it."""
    raw = url.strip()
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    host = (parsed.hostname or "").lower()
    reasons: list[tuple[int, str]] = []

    if not host:
        return {"url": url, "risk_score": 0, "verdict": "invalid URL", "reasons": ["Could not find a domain name."]}

    try:
        ipaddress.ip_address(host)
        reasons.append((35, "Uses a raw IP address instead of a domain name."))
    except ValueError:
        pass

    if "@" in parsed.netloc:
        reasons.append((40, "Contains '@' before the domain; the browser ignores everything before it."))
    if "xn--" in host:
        reasons.append((30, "Uses punycode (xn--), which can hide lookalike characters from other alphabets."))
    if host.count(".") >= 4:
        reasons.append((15, "Has many subdomains, which can hide the real domain."))
    if len(raw) > 100:
        reasons.append((10, "Very long URL, a common way to hide the real destination."))
    if host in SHORTENERS:
        reasons.append((20, "Uses a URL shortener that hides the final destination."))
    if host.rsplit(".", 1)[-1] in RISKY_TLDS:
        reasons.append((15, f"Uses a top-level domain (.{host.rsplit('.', 1)[-1]}) that is often abused."))
    if parsed.scheme == "http":
        reasons.append((5, "Not using HTTPS (note: HTTPS alone does not prove a site is safe)."))

    domain = registered_domain(host)
    domain_name = domain.split(".")[0]
    for brand, real_domain in BRANDS.items():
        if domain == real_domain or host.endswith("." + real_domain):
            break  # genuinely the brand's own domain
        if brand in host and brand not in domain:
            reasons.append((35, f"Mentions '{brand}' in a subdomain, but the real domain is {domain}."))
            break
        normalized = domain_name.translate(LOOKALIKE_SWAPS)  # e.g. "micros0ft" -> "microsoft"
        if normalized == brand or any(is_lookalike(word, brand) for word in [domain_name, *re.split(r"[-_]", domain_name)]):
            reasons.append((45, f"Domain '{domain}' looks like '{real_domain}' (possible typosquatting)."))
            break
        if brand in normalized and normalized != domain_name:
            reasons.append((45, f"Domain '{domain}' imitates '{brand}' using lookalike characters."))
            break
        if brand in domain_name and domain_name != brand:
            reasons.append((25, f"Domain '{domain}' combines the brand '{brand}' with other words."))
            break

    words = [w for w in SUSPICIOUS_WORDS if w in raw.lower()]
    if words:
        reasons.append((min(10 * len(words), 20), f"Contains words often used in phishing: {', '.join(words)}."))

    score = min(sum(points for points, _ in reasons), 100)
    verdict = "high risk" if score >= 60 else "suspicious" if score >= 30 else "low risk"
    return {
        "url": url,
        "domain": domain,
        "risk_score": score,
        "verdict": verdict,
        "reasons": [text for _, text in reasons] or ["No common phishing indicators found."],
    }


CVE_PATTERN = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)


def is_valid_cve_id(cve_id: str) -> bool:
    return bool(CVE_PATTERN.match(cve_id.strip()))
