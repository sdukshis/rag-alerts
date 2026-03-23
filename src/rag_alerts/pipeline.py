from pathlib import Path
from typing import List

import logging
import time

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
        self._logger = logging.getLogger(__name__)

    @property
    def incidents_path(self) -> Path:
        return self.data_dir / "incidents.jsonl"

    @property
    def index_path(self) -> Path:
        return self.index_dir / "incidents.faiss"

    def build(self) -> int:
        started = time.perf_counter()
        self._logger.info(
            "RAG build started incidents_path=%s index_path=%s",
            str(self.incidents_path),
            str(self.index_path),
        )
        self._incidents = load_incidents(self.incidents_path)
        self._logger.info("RAG loaded incidents count=%d", len(self._incidents))
        docs = [incident.to_document() for incident in self._incidents]
        self._logger.info("RAG embedding incidents count=%d", len(docs))
        vectors = self.embedding_model.encode(docs)
        index = build_index(vectors)
        save_index(index, self.index_path)
        self._logger.info(
            "RAG build finished incidents=%d elapsed_seconds=%.3f",
            len(self._incidents),
            time.perf_counter() - started,
        )
        return len(self._incidents)

    def _load_incidents(self) -> List[Incident]:
        if not self._incidents:
            self._logger.debug("RAG loading incidents from %s", str(self.incidents_path))
            self._incidents = load_incidents(self.incidents_path)
        else:
            self._logger.debug("RAG using cached incidents count=%d", len(self._incidents))
        return self._incidents

    def enrich(
        self,
        alert: Alert,
        top_k: int = 5,
        rerank_k: int = 3,
        use_reranker: bool | None = None,
    ) -> EnrichmentResult:
        started = time.perf_counter()
        self._logger.info(
            "RAG enrich started alert_id=%s service=%s top_k=%d rerank_k=%d",
            alert.alert_id,
            alert.service,
            top_k,
            rerank_k,
        )
        try:
            incidents = self._load_incidents()
            if not self.index_path.exists():
                raise FileNotFoundError(f"Vector index not found: {self.index_path}")
            index = load_index(self.index_path)

            t_retrieve_started = time.perf_counter()
            query = alert.to_query()
            self._logger.debug(
                "RAG retrieval started alert_id=%s incidents=%d top_k=%d index_path=%s",
                alert.alert_id,
                len(incidents),
                top_k,
                str(self.index_path),
            )
            query_vector = self.embedding_model.encode([query])
            scores, idx = search(index, query_vector, top_k)

            retrieved: List[RetrievedIncident] = []
            for score, incident_idx in zip(scores[0], idx[0]):
                if incident_idx < 0:
                    continue
                retrieved.append(
                    RetrievedIncident(incident=incidents[int(incident_idx)], score=float(score))
                )

            elapsed_retrieve = time.perf_counter() - t_retrieve_started
            top_score = max((item.score for item in retrieved), default=None)
            self._logger.debug(
                "RAG retrieval finished alert_id=%s retrieved=%d elapsed_seconds=%.3f top_score=%s",
                alert.alert_id,
                len(retrieved),
                elapsed_retrieve,
                f"{top_score:.4f}" if top_score is not None else "None",
            )

            should_rerank = self.use_reranker if use_reranker is None else use_reranker
            reranked = retrieved
            if should_rerank:
                self._logger.debug(
                    "RAG reranking started alert_id=%s rerank_k=%d candidates=%d",
                    alert.alert_id,
                    rerank_k,
                    len(retrieved),
                )
                if self.reranker is None:
                    self.reranker = Reranker()
                t_rerank_started = time.perf_counter()
                reranked = self.reranker.rerank(query, retrieved, rerank_k=rerank_k)
                self._logger.debug(
                    "RAG reranking finished alert_id=%s elapsed_seconds=%.3f",
                    alert.alert_id,
                    time.perf_counter() - t_rerank_started,
                )
            else:
                self._logger.debug("RAG reranking skipped alert_id=%s", alert.alert_id)

            prompt = build_enrichment_prompt(alert, reranked)
            self._logger.debug(
                "RAG LLM call started alert_id=%s prompt_chars=%d retrieved=%d",
                alert.alert_id,
                len(prompt),
                len(reranked),
            )
            response = generate_enrichment(prompt, alert=alert, retrieved=reranked)
            self._logger.debug(
                "RAG LLM call finished alert_id=%s response_chars=%d elapsed_seconds=%.3f",
                alert.alert_id,
                len(response),
                time.perf_counter() - started,
            )

            self._logger.info(
                "RAG enrich finished alert_id=%s elapsed_seconds=%.3f retrieved=%d",
                alert.alert_id,
                time.perf_counter() - started,
                len(reranked),
            )
            return EnrichmentResult(retrieval=reranked, final_response=response)
        except Exception:
            self._logger.exception("RAG enrich failed alert_id=%s", alert.alert_id)
            raise
