from typing import List

from rag_alerts.models import Incident, RetrievedIncident

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover
    CrossEncoder = None


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = CrossEncoder(model_name) if CrossEncoder else None

    def rerank(self, query: str, candidates: List[RetrievedIncident], rerank_k: int) -> List[RetrievedIncident]:
        selected = candidates[:rerank_k]
        if not selected:
            return selected
        if not self._model:
            return selected

        pairs = [(query, item.incident.to_document()) for item in selected]
        scores = self._model.predict(pairs)
        rescored = [
            RetrievedIncident(incident=item.incident, score=float(score))
            for item, score in zip(selected, scores)
        ]
        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored + candidates[rerank_k:]
