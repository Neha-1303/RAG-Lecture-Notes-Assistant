"""Create and persist normalized Sentence-Transformer vectors in FAISS."""
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import (
    EMBEDDING_MODEL, STORAGE_DIR, INDEX_PATH, METADATA_PATH
)

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def build_index(records):
    if not records:
        raise ValueError("There are no text chunks to index.")
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    texts = [record["text"] for record in records]
    vectors = get_embedding_model().encode(
        texts, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX_PATH))
    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
    return len(records)

def load_index():
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "The search index does not exist. Add PDFs to data/ and run `python ingest.py`."
        )
    index = faiss.read_index(str(INDEX_PATH))
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if index.ntotal != len(metadata):
        raise RuntimeError("FAISS index and metadata have different sizes. Rebuild the index.")
    return index, metadata
