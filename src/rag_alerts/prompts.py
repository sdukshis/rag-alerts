from typing import List

from rag_alerts.models import Alert, RetrievedIncident


def build_enrichment_prompt(alert: Alert, retrieved: List[RetrievedIncident]) -> str:
    context_lines = []
    for item in retrieved:
        context_lines.append(
            f"[score={item.score:.4f}] {item.incident.to_document()}"
        )

    context = "\n\n".join(context_lines) if context_lines else "No context available."
    return (
        "You are an SRE assistant. Enrich the monitoring alert with concise, actionable context.\n"
        "Return sections in this exact order:\n"
        "1) Likely scenario\n"
        "2) Similar incidents\n"
        "3) Investigation checklist\n"
        "4) Confidence (0-100)\n\n"
        f"Current alert:\n{alert.to_query()}\n\n"
        f"Retrieved historical context:\n{context}\n"
    )
