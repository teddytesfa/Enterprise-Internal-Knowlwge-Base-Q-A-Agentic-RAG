import numpy as np
from typing import Sequence

class EmbeddingService:
    def __init__(self, model=None, dim: int = 384):
        self.model = model
        self.dim = dim

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """
        Return numpy array of shape (len(texts), dim).
        Tries to use sentence-transformers if available; otherwise uses TF-IDF fallback.
        """
        # lazy import to keep requirements minimal
        try:
            from sentence_transformers import SentenceTransformer
            if self.model is None:
                # default model smaller and fast; you can override externally
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
            embs = self.model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
            return embs
        except Exception:
            # fallback: TF-IDF -> dense vectors (workable for v1)
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(max_features=self.dim)
            X = vec.fit_transform(texts).toarray()
            # ensure second dim equals self.dim
            if X.shape[1] == self.dim:
                return X
            elif X.shape[1] > self.dim:
                return X[:, : self.dim]
            else:
                pad = np.zeros((X.shape[0], max(0, self.dim - X.shape[1])))
                return np.hstack([X, pad])
