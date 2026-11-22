"""Central settings for the project, read from environment variables (.env)."""
import os
import sys

from dotenv import load_dotenv

# Windows terminals often use an old encoding (cp1252) that cannot print characters
# such as the brackets some LLMs use, which crashes print(). Switch to UTF-8 and
# replace anything that still cannot be shown instead of crashing.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

# Keep terminal output clean: hide harmless Hugging Face warnings and progress bars.
for _name, _value in {
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_VERBOSITY": "error",
    "TRANSFORMERS_VERBOSITY": "error",
    "TOKENIZERS_PARALLELISM": "false",
}.items():
    os.environ.setdefault(_name, _value)

# Embedding model: turns text into vectors. BGE models are trained for search.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# BGE works best when search queries (not documents) get this instruction prefix.
QUERY_PREFIX = os.getenv(
    "QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
)
# Cross-encoder reranker: re-scores (question, chunk) pairs for higher accuracy.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "security_advisories")
DATA_DIR = os.getenv("DATA_DIR", "data/advisories")

CHUNK_SIZE = 800      # max characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks

CANDIDATES = 20       # how many results each search method returns before fusion/reranking
