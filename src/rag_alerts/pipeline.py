from pathlib import Path
from typing import List

from rag_alerts.embeddings import EmbeddingModel
from rag_alerts.kb import load_incidents
from rag_alerts.llm import generate_enrichment
from rag_alerts.models import Alert, EnrichmentResult, Incident, RetrievedIncident
from rag_alerts.prompts import build_enrichment_prompt
from rag_alerts.reranker import Reranker
from rag_alerts.vector_index import build_index, load_index, save_index, search


class RAGAlertPipeline:
    def __init__(self, data_dir: Path, index_dir: Path, use_reranker: bool = False) -> None:
        self.data_dir = data_dir
        self.index_dir = index_dir
        self.use_reranker = use_reranker
        self.embedding_model = EmbeddingModel()
        self.reranker = Reranker() if use_reranker else None
        self._incidents: List[Incident] = []

    @property
    def incidents_path(self) -> Path:
        return self.data_dir / "incidents.jsonl"

    @property
    def index_path(self) -> Path:
        return self.index_dir / "incidents.faiss"

    def build(self) -> int:
        self._incidents = load_incidents(self.incidents_path)
        docs = [incident.to_document() for incident in self._incidents]
        vectors = self.embedding_model.encode(docs)
        index = build_index(vectors)
        save_index(index, self.index_path)
        return len(self._incidents)

    def _load_incidents(self) -> List[Incident]:
        if not self._incidents:
            self._incidents = load_incidents(self.incidents_path)
        return self._incidents

    def enrich(
        self,
        alert: Alert,
        top_k: int = 5,
        rerank_k: int = 3,
        use_reranker: bool | None = None,
    ) -> EnrichmentResult:
        incidents = self._load_incidents()
        index = load_index(self.index_path)
        query_vector = self.embedding_model.encode([alert.to_query()])
        scores, idx = search(index, query_vector, top_k)

        retrieved: List[RetrievedIncident] = []
        for score, incident_idx in zip(scores[0], idx[0]):
            if incident_idx < 0:
                continue
            retrieved.append(
                RetrievedIncident(incident=incidents[int(incident_idx)], score=float(score))
            )

        should_rerank = self.use_reranker if use_reranker is None else use_reranker
        reranked = retrieved
        if should_rerank:
            if self.reranker is None:
                self.reranker = Reranker()
            reranked = self.reranker.rerank(alert.to_query(), retrieved, rerank_k=rerank_k)
        prompt = build_enrichment_prompt(alert, reranked)
        response = generate_enrichment(prompt, alert=alert, retrieved=reranked)
        return EnrichmentResult(retrieval=reranked, final_response=response)
