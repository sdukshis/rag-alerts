from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


@dataclass(frozen=True)
class QdrantIndexHandle:
    collection_name: str


def _qdrant_url() -> str:
    # docker-compose.yml sets `QDRANT_URL: http://qdrant:6333` for the enricher container.
    return os.getenv("QDRANT_URL", "http://localhost:6333").strip()


def _client() -> QdrantClient:
    # Client/server versions can differ in this workshop stack.
    return QdrantClient(url=_qdrant_url(), check_compatibility=False)


def _collection_name(path: Path) -> str:
    # Pipeline historically uses `incidents.faiss` as a filename. For Qdrant we use its stem.
    return path.stem


def collection_exists(path: Path) -> bool:
    client = _client()
    return client.collection_exists(_collection_name(path))


def build_index(vectors: np.ndarray) -> np.ndarray:
    # Qdrant stores vectors directly; we keep the same "build then save" interface.
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    return vectors


def save_index(vectors: np.ndarray, path: Path) -> None:
    collection_name = _collection_name(path)
    dim = int(vectors.shape[1])

    client = _client()
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.DOT),
    )

    # We store the vector "as is" and rely on point id == incident index.
    points = [
        qmodels.PointStruct(id=i, vector=vec.tolist())
        for i, vec in enumerate(vectors)
    ]
    client.upsert(collection_name=collection_name, points=points)

    # Compatibility marker so existing checks like `index_path.exists()` can work.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("qdrant-index-built", encoding="utf-8")


def load_index(path: Path) -> QdrantIndexHandle:
    # We intentionally do not create/open anything here; search/load creates the client.
    return QdrantIndexHandle(collection_name=_collection_name(path))


def search(
    index: QdrantIndexHandle,
    query: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if query.dtype != np.float32:
        query = query.astype(np.float32)
    if query.ndim == 1:
        query = query.reshape(1, -1)

    client = _client()
    distances = np.zeros((query.shape[0], top_k), dtype=np.float32)
    indices = np.full((query.shape[0], top_k), -1, dtype=np.int64)

    for row_idx in range(query.shape[0]):
        # qdrant-client 1.x uses `query_points` rather than `search`.
        results = client.query_points(
            collection_name=index.collection_name,
            query=query[row_idx].tolist(),
            limit=top_k,
            with_payload=False,
        )

        # `results` is a QueryResponse-like object with `.points` attribute.
        scored_points = getattr(results, "points", results)
        for rank, scored_point in enumerate(scored_points):
            if rank >= top_k:
                break
            indices[row_idx, rank] = int(scored_point.id)
            distances[row_idx, rank] = float(scored_point.score)

    return distances, indices
