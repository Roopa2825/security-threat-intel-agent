"""Central settings for the project, read from environment variables (.env)."""
import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "security_advisories")
DATA_DIR = os.getenv("DATA_DIR", "data/advisories")

CHUNK_SIZE = 800      # max characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks
