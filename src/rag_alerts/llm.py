import os

from openai import OpenAI

from rag_alerts.models import Alert, RetrievedIncident


def _fallback_response(alert: Alert, retrieved: list[RetrievedIncident]) -> str:
    similar = "\n".join(
        f"- {item.incident.incident_id}: {item.incident.summary}" for item in retrieved[:3]
    ) or "- No similar incidents found."
    return (
        "1) Likely scenario\n"
        f"Potential degradation in {alert.service} tied to symptoms in the alert.\n\n"
        "2) Similar incidents\n"
        f"{similar}\n\n"
        "3) Investigation checklist\n"
        "- Validate recent deploys and config changes\n"
        "- Check saturation and dependency latency\n"
        "- Inspect logs around first anomaly timestamp\n\n"
        "4) Suggested response plan\n"
        "- Mitigate immediate impact (scale/rollback/failover)\n"
        "- Verify recovery on key SLO metrics\n"
        "- Record findings in incident timeline\n\n"
        "5) Confidence (0-100)\n"
        "58\n"
    )


def generate_enrichment(prompt: str, alert: Alert, retrieved: list[RetrievedIncident]) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return _fallback_response(alert, retrieved)

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=prompt,
    )
    return response.output_text.strip()
