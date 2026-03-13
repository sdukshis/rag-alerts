import argparse
import random
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

from flask import Flask, jsonify, redirect, request, url_for
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

SERVICE_NAME = "checkout-api"

ENRICHMENT_SCORE = Gauge(
    "rag_alert_enrichment_score",
    "Synthetic RAG enrichment quality score (0..1).",
    labelnames=("service", "incident_type"),
)
ENRICHMENT_FAILURES = Counter(
    "rag_alert_enrichment_failures_total",
    "Synthetic count of enrichment failures.",
    labelnames=("incident_type",),
)
ENRICHMENT_LATENCY = Histogram(
    "rag_alert_enrichment_latency_seconds",
    "Synthetic enrichment latency distribution.",
    labelnames=("incident_type",),
    buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0),
)
ENRICHED_TOTAL = Counter(
    "rag_alerts_enriched_total",
    "Synthetic number of alerts processed by enrichment pipeline.",
    labelnames=("incident_type", "status"),
)
ACTIVE_INCIDENT = Gauge(
    "rag_simulator_active_incident",
    "Whether an incident profile is active (1 active, 0 inactive).",
    labelnames=("incident_type",),
)
SIGNAL_LATENCY_MS = Gauge(
    "rag_simulator_signal_latency_ms",
    "Synthetic service latency in milliseconds.",
)
SIGNAL_ERROR_RATE = Gauge(
    "rag_simulator_signal_error_rate",
    "Synthetic service error rate (0..1).",
)
SIGNAL_CPU = Gauge(
    "rag_simulator_signal_cpu_usage",
    "Synthetic service CPU usage (0..1).",
)


@dataclass
class IncidentProfile:
    title: str
    description: str
    base_latency_ms: float
    error_rate: float
    cpu_usage: float
    enrichment_score: float
    failure_probability: float
    requests_per_tick: int


INCIDENT_PROFILES: Dict[str, IncidentProfile] = {
    "latency_spike": IncidentProfile(
        title="Latency Spike",
        description="Spiky p95 latency after traffic surge.",
        base_latency_ms=1400,
        error_rate=0.08,
        cpu_usage=0.72,
        enrichment_score=0.42,
        failure_probability=0.12,
        requests_per_tick=6,
    ),
    "dependency_outage": IncidentProfile(
        title="Dependency Outage",
        description="Downstream dependency failures causing heavy errors.",
        base_latency_ms=2400,
        error_rate=0.35,
        cpu_usage=0.58,
        enrichment_score=0.24,
        failure_probability=0.45,
        requests_per_tick=5,
    ),
    "memory_leak": IncidentProfile(
        title="Memory Leak",
        description="Steady memory pressure with degrading performance.",
        base_latency_ms=800,
        error_rate=0.05,
        cpu_usage=0.50,
        enrichment_score=0.58,
        failure_probability=0.08,
        requests_per_tick=7,
    ),
}


class IncidentSimulator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = True
        self._active_incident: Optional[str] = None
        self._active_tick = 0
        self._random = random.Random(42)
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def start_incident(self, incident_type: str) -> None:
        if incident_type not in INCIDENT_PROFILES:
            raise ValueError(f"Unknown incident type: {incident_type}")
        with self._lock:
            self._active_incident = incident_type
            self._active_tick = 0

    def stop_incident(self) -> None:
        with self._lock:
            self._active_incident = None
            self._active_tick = 0

    def get_state(self) -> Dict[str, object]:
        with self._lock:
            active = self._active_incident
            tick = self._active_tick
        return {
            "active_incident": active,
            "active_tick": tick,
            "available_incidents": {
                key: {
                    "title": value.title,
                    "description": value.description,
                }
                for key, value in INCIDENT_PROFILES.items()
            },
        }

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        while self._running:
            with self._lock:
                incident_type = self._active_incident
                active_tick = self._active_tick
                if incident_type is not None:
                    self._active_tick += 1

            self._emit_metrics(incident_type, active_tick)
            time.sleep(1.0)

    def _emit_metrics(self, incident_type: Optional[str], active_tick: int) -> None:
        for key in INCIDENT_PROFILES:
            ACTIVE_INCIDENT.labels(incident_type=key).set(1 if key == incident_type else 0)

        if incident_type is None:
            self._emit_baseline()
            return

        profile = INCIDENT_PROFILES[incident_type]
        trend = 1.0
        if incident_type == "memory_leak":
            trend = min(2.8, 1.0 + (active_tick / 90.0))

        latency_ms = max(20.0, profile.base_latency_ms * trend + self._random.gauss(0, 45))
        error_rate = min(1.0, max(0.0, profile.error_rate * trend + self._random.gauss(0, 0.01)))
        cpu_usage = min(1.0, max(0.0, profile.cpu_usage * trend + self._random.gauss(0, 0.02)))
        enrichment_score = min(
            1.0,
            max(0.0, profile.enrichment_score - (0.04 * (trend - 1.0)) + self._random.gauss(0, 0.015)),
        )
        failure_probability = min(1.0, max(0.0, profile.failure_probability + (0.02 * (trend - 1.0))))

        SIGNAL_LATENCY_MS.set(latency_ms)
        SIGNAL_ERROR_RATE.set(error_rate)
        SIGNAL_CPU.set(cpu_usage)
        ENRICHMENT_SCORE.labels(service=SERVICE_NAME, incident_type=incident_type).set(enrichment_score)

        for _ in range(profile.requests_per_tick):
            latency_seconds = max(0.02, (latency_ms / 1000.0) + self._random.gauss(0, 0.08))
            ENRICHMENT_LATENCY.labels(incident_type=incident_type).observe(latency_seconds)

            if self._random.random() < failure_probability:
                ENRICHMENT_FAILURES.labels(incident_type=incident_type).inc()
                ENRICHED_TOTAL.labels(incident_type=incident_type, status="failed").inc()
            else:
                ENRICHED_TOTAL.labels(incident_type=incident_type, status="success").inc()

    def _emit_baseline(self) -> None:
        incident_type = "none"
        latency_ms = max(20.0, 180 + self._random.gauss(0, 10))
        error_rate = min(1.0, max(0.0, 0.01 + self._random.gauss(0, 0.002)))
        cpu_usage = min(1.0, max(0.0, 0.30 + self._random.gauss(0, 0.02)))
        score = min(1.0, max(0.0, 0.82 + self._random.gauss(0, 0.01)))

        SIGNAL_LATENCY_MS.set(latency_ms)
        SIGNAL_ERROR_RATE.set(error_rate)
        SIGNAL_CPU.set(cpu_usage)
        ENRICHMENT_SCORE.labels(service=SERVICE_NAME, incident_type=incident_type).set(score)

        for _ in range(4):
            ENRICHMENT_LATENCY.labels(incident_type=incident_type).observe(max(0.02, (latency_ms / 1000.0)))
            ENRICHED_TOTAL.labels(incident_type=incident_type, status="success").inc()


def _render_ui(state: Dict[str, object]) -> str:
    active_incident = state["active_incident"]
    active_tick = state["active_tick"]
    options = []
    incidents = state["available_incidents"]
    for key, info in incidents.items():
        selected = "selected" if key == active_incident else ""
        options.append(f"<option value='{key}' {selected}>{info['title']} ({key})</option>")

    current = active_incident or "none"
    return f"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>RAG Incident Simulator</title>
    <style>
      body {{ font-family: Arial, sans-serif; max-width: 760px; margin: 32px auto; line-height: 1.4; }}
      h1 {{ margin-bottom: 8px; }}
      .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
      button {{ padding: 8px 14px; margin-right: 8px; cursor: pointer; }}
      select {{ min-width: 320px; padding: 6px; }}
      code {{ background: #f6f6f6; padding: 2px 4px; border-radius: 4px; }}
      .muted {{ color: #666; }}
    </style>
  </head>
  <body>
    <h1>RAG Alert Incident Simulator</h1>
    <p class="muted">Run synthetic incidents and monitor metrics in Grafana.</p>

    <div class="card">
      <p><strong>Active incident:</strong> <code>{current}</code></p>
      <p><strong>Incident uptime (ticks):</strong> <code>{active_tick}</code></p>
      <form method="post" action="/start">
        <label for="incident_type">Incident profile</label><br />
        <select id="incident_type" name="incident_type">
          {"".join(options)}
        </select>
        <div style="margin-top: 12px;">
          <button type="submit">Start / Switch Incident</button>
        </div>
      </form>
      <form method="post" action="/stop" style="margin-top: 12px;">
        <button type="submit">Stop Incident</button>
      </form>
    </div>

    <div class="card">
      <h3>Incident types</h3>
      <ul>
        <li><strong>latency_spike</strong> - short-term traffic pressure and high p95 latency</li>
        <li><strong>dependency_outage</strong> - upstream outage with high failures and low score</li>
        <li><strong>memory_leak</strong> - gradual degradation with increasing latency and CPU</li>
      </ul>
      <p>Prometheus scrape endpoint: <a href="/metrics"><code>/metrics</code></a></p>
      <p>JSON state endpoint: <a href="/api/state"><code>/api/state</code></a></p>
    </div>
  </body>
</html>
"""


def create_app(simulator: IncidentSimulator) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def home() -> str:
        return _render_ui(simulator.get_state())

    @app.post("/start")
    def start() -> object:
        incident_type = request.form.get("incident_type", "").strip()
        if incident_type in INCIDENT_PROFILES:
            simulator.start_incident(incident_type)
        return redirect(url_for("home"))

    @app.post("/stop")
    def stop() -> object:
        simulator.stop_incident()
        return redirect(url_for("home"))

    @app.get("/api/state")
    def api_state() -> object:
        return jsonify(simulator.get_state())

    @app.get("/metrics")
    def metrics() -> object:
        return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic RAG incidents and metrics UI.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    simulator = IncidentSimulator()
    app = create_app(simulator)

    try:
        app.run(host=args.host, port=args.port)
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
