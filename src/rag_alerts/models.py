from dataclasses import dataclass, field
from typing import List


@dataclass
class Incident:
    incident_id: str
    service: str
    summary: str
    symptoms: List[str]
    root_cause: str
    actions_taken: List[str]
    tags: List[str]
    severity: str

    def to_document(self) -> str:
        symptoms = "; ".join(self.symptoms)
        actions = "; ".join(self.actions_taken)
        tags = ", ".join(self.tags)
        return (
            f"Incident {self.incident_id} | service={self.service} | severity={self.severity}\n"
            f"Summary: {self.summary}\n"
            f"Symptoms: {symptoms}\n"
            f"Root cause: {self.root_cause}\n"
            f"Actions taken: {actions}\n"
            f"Tags: {tags}"
        )


@dataclass
class Alert:
    alert_id: str
    service: str
    title: str
    description: str
    metrics: List[str]
    labels: dict = field(default_factory=dict)

    def to_query(self) -> str:
        metrics = "; ".join(self.metrics)
        labels = ", ".join([f"{k}={v}" for k, v in self.labels.items()])
        return (
            f"Alert {self.alert_id} | service={self.service}\n"
            f"Title: {self.title}\n"
            f"Description: {self.description}\n"
            f"Metrics: {metrics}\n"
            f"Labels: {labels}"
        )


@dataclass
class RetrievedIncident:
    incident: Incident
    score: float


@dataclass
class EnrichmentResult:
    retrieval: List[RetrievedIncident]
    final_response: str
