import os
from typing import List

import numpy as np
from openai import OpenAI


class EmbeddingModel:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not base_url:
            raise ValueError("OPENAI_BASE_URL is required for OpenAI API.")
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

    def encode(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        vectors = np.asarray([item.embedding for item in response.data], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return vectors / norms
