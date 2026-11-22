"""Central settings for the project, read from environment variables (.env)."""
import os
from dotenv import load_dotenv

load_dotenv()

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
