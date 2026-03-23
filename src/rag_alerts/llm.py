import os

from openai import OpenAI

from rag_alerts.models import Alert, RetrievedIncident


def generate_enrichment(prompt: str, alert: Alert, retrieved: list[RetrievedIncident]) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url:
        raise ValueError("OPENAI_BASE_URL is required for OpenAI API.")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for OpenAI API.")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content.strip()
