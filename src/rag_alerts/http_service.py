import hashlib
import html
import time
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from rag_alerts.models import Alert, EnrichmentResult, RetrievedIncident
from rag_alerts.pipeline import RAGAlertPipeline

API_REQUESTS_TOTAL = Counter(
    "rag_alert_api_requests_total",
    "Total enrichment API requests.",
    labelnames=("endpoint", "status"),
)
API_REQUEST_LATENCY = Histogram(
    "rag_alert_api_request_latency_seconds",
    "Latency of enrichment API requests.",
    labelnames=("endpoint",),
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
)


def _serialize_retrieval(items: list[RetrievedIncident]) -> list[dict[str, Any]]:
    return [
        {
            "incident_id": item.incident.incident_id,
            "service": item.incident.service,
            "severity": item.incident.severity,
            "summary": item.incident.summary,
            "score": item.score,
        }
        for item in items
    ]


def _serialize_result(result: EnrichmentResult) -> dict[str, Any]:
    return {
        "retrieval": _serialize_retrieval(result.retrieval),
        "final_response": result.final_response,
    }


def _from_direct_payload(payload: dict[str, Any]) -> Alert:
    if "alert_id" not in payload:
        raise ValueError("Expected `alert_id` field in direct alert payload.")
    return Alert(**payload)


def _from_grafana_alert(alert_payload: dict[str, Any], idx: int) -> Alert:
    labels = alert_payload.get("labels") or {}
    annotations = alert_payload.get("annotations") or {}
    values = alert_payload.get("values") or {}

    fingerprint = str(alert_payload.get("fingerprint") or "").strip()
    if not fingerprint:
        raw = f"{labels}|{annotations}|{alert_payload.get('startsAt')}|{idx}"
        fingerprint = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    alertname = str(labels.get("alertname") or "GrafanaAlert").strip()
    service = str(labels.get("service") or labels.get("job") or "unknown-service").strip()

    metric_snippets = [f"{k}={v}" for k, v in values.items()]
    if not metric_snippets:
        metric_snippets = [f"{k}={v}" for k, v in labels.items() if k in ("alertname", "severity", "source")]

    description = (
        str(annotations.get("description") or "").strip()
        or str(annotations.get("summary") or "").strip()
        or f"Grafana alert {alertname} fired for service {service}."
    )

    return Alert(
        alert_id=f"{alertname}:{fingerprint}",
        service=service,
        title=alertname,
        description=description,
        metrics=metric_snippets,
        labels=labels,
    )


def create_enrichment_app(
    data_dir: Path,
    index_dir: Path,
    top_k: int = 5,
    rerank_k: int = 3,
) -> Flask:
    pipeline: RAGAlertPipeline | None = None
    last_enriched: dict[str, Any] | None = None

    def get_pipeline() -> RAGAlertPipeline:
        nonlocal pipeline
        if pipeline is None:
            pipeline = RAGAlertPipeline(data_dir=data_dir, index_dir=index_dir)
        return pipeline

    def ensure_index_ready() -> None:
        active_pipeline = get_pipeline()
        if not active_pipeline.index_path.exists():
            active_pipeline.build()

    def remember_latest(source: str, alert: Alert, result: EnrichmentResult) -> None:
        nonlocal last_enriched
        last_enriched = {
            "source": source,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "alert": {
                "alert_id": alert.alert_id,
                "service": alert.service,
                "title": alert.title,
                "description": alert.description,
                "metrics": alert.metrics,
                "labels": alert.labels,
            },
            "enrichment": _serialize_result(result),
        }

    def render_last_enrichment_page() -> str:
        if last_enriched is None:
            return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>RAG Enricher</title>
    <style>
      body { font-family: Arial, sans-serif; max-width: 880px; margin: 32px auto; line-height: 1.45; }
      .card { border: 1px solid #ddd; border-radius: 8px; padding: 16px; }
      code { background: #f6f6f6; border-radius: 4px; padding: 2px 4px; }
      .muted { color: #666; }
    </style>
  </head>
  <body>
    <h1>RAG Enricher API</h1>
    <div class="card">
      <p><strong>No enrichment processed yet.</strong></p>
      <p class="muted">POST an alert to <code>/api/enrich</code> or <code>/api/grafana/enrich</code>, then refresh this page.</p>
      <p>Health: <a href="/healthz"><code>/healthz</code></a> | Metrics: <a href="/metrics"><code>/metrics</code></a></p>
    </div>
  </body>
</html>
"""

        alert = last_enriched["alert"]
        enrichment = last_enriched["enrichment"]
        retrieval_items = enrichment.get("retrieval", [])
        retrieval_html = "".join(
            "<li><code>{}</code> ({}) score={:.4f} - {}</li>".format(
                html.escape(item.get("incident_id", "")),
                html.escape(item.get("service", "")),
                float(item.get("score", 0.0)),
                html.escape(item.get("summary", "")),
            )
            for item in retrieval_items
        ) or "<li>No retrieved incidents.</li>"

        labels_lines = "".join(
            f"<li><code>{html.escape(str(k))}</code> = {html.escape(str(v))}</li>"
            for k, v in (alert.get("labels") or {}).items()
        ) or "<li>No labels</li>"

        metrics_lines = "".join(
            f"<li><code>{html.escape(str(metric))}</code></li>" for metric in (alert.get("metrics") or [])
        ) or "<li>No metrics</li>"

        final_response = html.escape(str(enrichment.get("final_response", "")))
        return f"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>RAG Enricher - Last Alert</title>
    <style>
      body {{ font-family: Arial, sans-serif; max-width: 980px; margin: 32px auto; line-height: 1.45; }}
      .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
      .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; }}
      pre {{ white-space: pre-wrap; background: #f8f8f8; border-radius: 8px; padding: 12px; }}
      code {{ background: #f6f6f6; border-radius: 4px; padding: 2px 4px; }}
      .meta {{ color: #555; margin-bottom: 16px; }}
      @media (max-width: 860px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    </style>
  </head>
  <body>
    <h1>RAG Enricher - Last Enriched Alert</h1>
    <p class="meta">
      Source: <code>{html.escape(str(last_enriched["source"]))}</code> |
      Updated: <code>{html.escape(str(last_enriched["updated_at"]))}</code>
    </p>

    <div class="grid">
      <div class="card">
        <h3>Alert</h3>
        <p><strong>ID:</strong> <code>{html.escape(str(alert.get("alert_id", "")))}</code></p>
        <p><strong>Service:</strong> <code>{html.escape(str(alert.get("service", "")))}</code></p>
        <p><strong>Title:</strong> {html.escape(str(alert.get("title", "")))}</p>
        <p><strong>Description:</strong> {html.escape(str(alert.get("description", "")))}</p>
        <h4>Metrics</h4>
        <ul>{metrics_lines}</ul>
        <h4>Labels</h4>
        <ul>{labels_lines}</ul>
      </div>

      <div class="card">
        <h3>Retrieved Incidents</h3>
        <ul>{retrieval_html}</ul>
      </div>
    </div>

    <div class="card" style="margin-top: 16px;">
      <h3>Final Enrichment Response</h3>
      <pre>{final_response}</pre>
    </div>
  </body>
</html>
"""

    app = Flask(__name__)

    @app.get("/")
    def home() -> str:
        return render_last_enrichment_page()

    @app.get("/healthz")
    def healthz() -> object:
        return jsonify({"status": "ok"})

    @app.get("/api/last-enrichment")
    def api_last_enrichment() -> object:
        if last_enriched is None:
            return jsonify({"message": "No enrichments yet."})
        return jsonify(last_enriched)

    @app.get("/metrics")
    def metrics() -> object:
        return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

    @app.post("/api/enrich")
    def enrich_alert() -> object:
        started = time.perf_counter()
        endpoint = "/api/enrich"
        try:
            payload = request.get_json(force=True, silent=False) or {}
            alert = _from_direct_payload(payload)
            ensure_index_ready()
            result = get_pipeline().enrich(alert=alert, top_k=top_k, rerank_k=rerank_k)
            remember_latest(source=endpoint, alert=alert, result=result)
            API_REQUESTS_TOTAL.labels(endpoint=endpoint, status="success").inc()
            return jsonify(_serialize_result(result))
        except Exception as exc:  # noqa: BLE001
            API_REQUESTS_TOTAL.labels(endpoint=endpoint, status="error").inc()
            return jsonify({"error": str(exc)}), 400
        finally:
            API_REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - started)

    @app.post("/api/grafana/enrich")
    def enrich_from_grafana() -> object:
        started = time.perf_counter()
        endpoint = "/api/grafana/enrich"
        try:
            payload = request.get_json(force=True, silent=False) or {}
            alerts_payload = payload.get("alerts")
            if not isinstance(alerts_payload, list) or not alerts_payload:
                raise ValueError("Expected Grafana webhook payload with a non-empty `alerts` list.")

            ensure_index_ready()
            enriched = []
            for idx, incoming in enumerate(alerts_payload):
                if not isinstance(incoming, dict):
                    continue
                alert = _from_grafana_alert(incoming, idx)
                result = get_pipeline().enrich(alert=alert, top_k=top_k, rerank_k=rerank_k)
                remember_latest(source=endpoint, alert=alert, result=result)
                enriched.append(
                    {
                        "input": {
                            "alert_id": alert.alert_id,
                            "title": alert.title,
                            "service": alert.service,
                        },
                        "enrichment": _serialize_result(result),
                    }
                )

            API_REQUESTS_TOTAL.labels(endpoint=endpoint, status="success").inc()
            return jsonify({"processed": len(enriched), "results": enriched})
        except Exception as exc:  # noqa: BLE001
            API_REQUESTS_TOTAL.labels(endpoint=endpoint, status="error").inc()
            return jsonify({"error": str(exc)}), 400
        finally:
            API_REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - started)

    return app
