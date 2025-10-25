import os
import pickle
from typing import List, Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors

class SimpleIndexer:
    """
    A file-backed in-memory indexer using sklearn NearestNeighbors (cosine).
    """
    def __init__(self, output_dir: str, metric="cosine", n_neighbors=10):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.vectors = None  # np.ndarray of shape (n, d)
        self.metadata = []   # list of metadata per vector
        self._nn = None

    def build(self, vectors: np.ndarray, metadata: List[dict]):
        self.vectors = vectors
        self.metadata = metadata
        if vectors is None or len(vectors) == 0:
            self._nn = None
            return
        self._nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)
        self._nn.fit(vectors)

    def save(self, prefix="index"):
        with open(os.path.join(self.output_dir, f"{prefix}_vectors.pkl"), "wb") as f:
            pickle.dump(self.vectors, f)
        with open(os.path.join(self.output_dir, f"{prefix}_meta.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, prefix="index"):
        vec_path = os.path.join(self.output_dir, f"{prefix}_vectors.pkl")
        meta_path = os.path.join(self.output_dir, f"{prefix}_meta.pkl")
        if not os.path.exists(vec_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("Index files not found")
        with open(vec_path, "rb") as f:
            self.vectors = pickle.load(f)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        if self.vectors is not None and len(self.vectors) > 0:
            self._nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)
            self._nn.fit(self.vectors)

    def query(self, vector: np.ndarray, top_k: int = 5) -> List[Tuple[dict, float]]:
        if self._nn is None:
            return []
        distances, indices = self._nn.kneighbors(vector.reshape(1, -1), n_neighbors=top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append((self.metadata[int(idx)], float(dist)))
        return results
