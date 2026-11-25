"""Retrieve the top-k semantically similar note chunks."""
import numpy as np
from index import get_embedding_model, load_index
from config import TOP_K, MIN_SIMILARITY

class Retriever:
    def __init__(self):
        self.index, self.metadata = load_index()

    def search(self, question, top_k=TOP_K, min_similarity=MIN_SIMILARITY):
        if not question or not question.strip():
            return []
        vector = get_embedding_model().encode(
            [question], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(vector, min(top_k, len(self.metadata)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or float(score) < min_similarity:
                continue
            item = dict(self.metadata[int(idx)])
            item["score"] = float(score)
            results.append(item)
        return results
