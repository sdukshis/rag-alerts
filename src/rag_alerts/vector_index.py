from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np


def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(index: faiss.IndexFlatIP, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.IndexFlatIP:
    return faiss.read_index(str(path))


def search(index: faiss.IndexFlatIP, query: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    if query.dtype != np.float32:
        query = query.astype(np.float32)
    distances, indices = index.search(query, top_k)
    return distances, indices
