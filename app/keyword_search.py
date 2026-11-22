"""BM25 keyword search: good at exact terms such as product names and CVE IDs,
which pure meaning-based (vector) search can miss."""
import re
from functools import lru_cache

# Well-known nicknames that do not appear in official vulnerability records.
# Expanding them lets keyword search find the official entries.
ALIASES = {
    "log4shell": "log4j log4j2 cve-2021-44228",
    "heartbleed": "openssl heartbeat cve-2014-0160",
    "eternalblue": "smbv1 ms17-010 cve-2017-0144",
    "wannacry": "smbv1 ms17-010 cve-2017-0144",
    "citrix bleed": "netscaler cve-2023-4966",
    "xz backdoor": "xz liblzma cve-2024-3094",
}


def tokenize(text: str) -> list[str]:
    """Lowercase words, keeping hyphens and dots so 'cve-2021-44228' stays one token."""
    return re.findall(r"[a-z0-9][a-z0-9.\-]*", text.lower())


def expand_query(query: str) -> str:
    """Add official names for any known nickname in the query."""
    lowered = query.lower()
    extra = [terms for alias, terms in ALIASES.items() if alias in lowered]
    return " ".join([query, *extra])


class KeywordIndex:
    def __init__(self, ids: list[str], texts: list[str], metadatas: list[dict]):
        self.ids, self.texts, self.metadatas = ids, texts, metadatas
        from rank_bm25 import BM25Okapi  # imported here so tests can run without it

        self.bm25 = BM25Okapi([tokenize(t) for t in texts]) if texts else None

    def search(self, query: str, n: int) -> list[int]:
        """Return positions of the n best-matching chunks."""
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(expand_query(query)))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in ranked[:n] if scores[i] > 0]


@lru_cache(maxsize=1)
def get_keyword_index() -> KeywordIndex:
    from app.vectorstore import get_all_chunks  # needs ChromaDB
    return KeywordIndex(*get_all_chunks())
