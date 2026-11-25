"""Central configuration for the RAG Lecture-Notes Assistant."""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
INDEX_PATH = STORAGE_DIR / "faiss.index"
METADATA_PATH = STORAGE_DIR / "metadata.json"

LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "350"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))
TOP_K = int(os.getenv("TOP_K", "4"))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.25"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "700"))

def validate_settings(require_llm=False):
    if CHUNK_SIZE <= 0 or CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("CHUNK_SIZE must be positive and CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
    if TOP_K < 1:
        raise ValueError("TOP_K must be at least 1.")
    if require_llm:
        missing = [
            name for name, value in [
                ("LLM_API_KEY", LLM_API_KEY),
                ("LLM_BASE_URL", LLM_BASE_URL),
                ("LLM_MODEL", LLM_MODEL),
            ] if not value or value.startswith("replace_with")
        ]
        if missing:
            raise RuntimeError(
                "Configure these values in your .env file before asking questions: "
                + ", ".join(missing)
            )
