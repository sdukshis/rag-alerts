import os
from typing import List

import numpy as np


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class EmbeddingModel:
    def __init__(self, model_name: str | None = None) -> None:
        # Switch between embeddings backends:
        # - USE_EMBEDDINGS_API=true  -> call OpenAI embeddings API
        # - USE_EMBEDDINGS_API=false -> local HuggingFace sentence-transformers model
        self.use_embeddings_api = _parse_bool(os.getenv("USE_EMBEDDINGS_API"), default=True)

        if self.use_embeddings_api:
            from openai import OpenAI

            self.model_name = model_name or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
            base_url = os.getenv("OPENAI_BASE_URL")
            if not base_url:
                raise ValueError("OPENAI_BASE_URL is required for OpenAI API.")
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        else:
            from sentence_transformers import SentenceTransformer

            # "simple model from hugging face" - overridable for demos/tests
            self.model_name = model_name or os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.use_embeddings_api:
            response = self.client.embeddings.create(model=self.model_name, input=texts)
            vectors = np.asarray([item.embedding for item in response.data], dtype=np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            return vectors / norms

        # SentenceTransformer returns normalized vectors when requested.
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32, copy=False)
