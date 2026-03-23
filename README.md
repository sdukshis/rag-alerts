# RAG Alerts Workshop Demo

Demo project for the workshop **"Enhancing Observability Alerts with RAG"**.

This project shows how to enrich monitoring alerts with context from past incidents using:

- Retrieval over an incident knowledge base
- Embeddings for semantic search
- Optional reranking for better relevance
- Prompt-based enrichment with OpenAI LLMs

## Workshop Mapping

1. **Environment setup**: Python project with dependencies and CLI
2. **Knowledge base of past incidents**: `data/incidents.jsonl`
3. **Building a RAG pipeline**: `src/rag_alerts/pipeline.py`
4. **Document and query embeddings**: `src/rag_alerts/embeddings.py`
5. **Reranking**: `src/rag_alerts/reranker.py`
6. **Prompt engineering**: `src/rag_alerts/prompts.py`
7. **Alert enrichment**: `src/rag_alerts/llm.py` + CLI command

## Quickstart

### 1) Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install package and dependencies

```bash
pip install -e .
```

### 3) Configure environment

```bash
cp .env.example .env
```

Optional for LLM enrichment:

- Set `OPENAI_API_KEY` in `.env`

Embedding backend (for semantic search):

- Set `USE_EMBEDDINGS_API=false` to use a local HuggingFace `sentence-transformers` model instead of OpenAI embeddings.
- Optionally set `HF_EMBEDDING_MODEL` to change the HuggingFace model (default: `sentence-transformers/all-MiniLM-L6-v2`).

### 4) Build vector index

```bash
python -m rag_alerts.cli build-index
```

### 5) Run demo on sample alert

```bash
python -m rag_alerts.cli run-demo
```

### 6) Enrich a custom alert

```bash
python -m rag_alerts.cli enrich-alert --alert-file data/sample_alert.json --top-k 5 --rerank-k 3
```

### 7) Run enrichment as HTTP API

```bash
python -m rag_alerts.cli serve-http --host 0.0.0.0 --port 8081
```

- Health check: `http://localhost:8081/healthz`
- Enricher web page (last processed alert): `http://localhost:8081/`
- Direct enrichment endpoint: `POST /api/enrich`
- Grafana webhook endpoint: `POST /api/grafana/enrich`

## Project Structure

```text
data/
  incidents.jsonl
  sample_alert.json
src/rag_alerts/
  cli.py
  embeddings.py
  kb.py
  llm.py
  models.py
  pipeline.py
  prompts.py
  reranker.py
  vector_index.py
```

## Notes

- If OpenAI credentials are missing, the demo still runs and returns a deterministic fallback enrichment.
- First run can be slow because Hugging Face models are downloaded.

## Observability Demo (Prometheus + Grafana)

This repository includes a Docker Compose stack with:

- a synthetic incident simulator (`simulator`) that exposes a control UI and Prometheus metrics
- an enrichment HTTP API (`enricher`) that receives Grafana webhooks and runs RAG enrichment
- Prometheus for scrape + alert rules
- Grafana with a prebuilt dashboard and provisioned alerting webhook to the enrichment API

### Start stack

```bash
docker compose up -d
```

- Incident simulator UI: `http://localhost:8000`
- Enrichment API: `http://localhost:8081`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (login: `admin` / `admin`)

### Run incidents from UI

Open `http://localhost:8000` and start one of the incident profiles:

1. `latency_spike` - traffic surge with p95 latency spikes
2. `dependency_outage` - dependency failures with high error ratio
3. `memory_leak` - gradual degradation with rising latency and CPU

Use **Stop Incident** to return to healthy baseline.

### What to expose from your RAG app

Prometheus is configured to scrape `http://simulator:8000/metrics` inside Docker.

Recommended metric names:

- `rag_alert_enrichment_score` (gauge, `0..1`)
- `rag_alert_enrichment_failures_total` (counter)
- `rag_alert_enrichment_latency_seconds` (histogram)
- `rag_alerts_enriched_total` (counter)

### Included alert rules

`observability/prometheus/alerts/rag-alerts.yml` contains starter alerts:

- low enrichment quality (`RAGAlertEnrichmentQualityLow`)
- high enrichment failure rate (`RAGAlertEnrichmentFailuresHigh`)

`observability/grafana/provisioning/alerting/rules.yml` contains a Grafana-managed rule:

- low enrichment score with webhook delivery to `http://enricher:8081/api/grafana/enrich`

### Included Grafana dashboard

`RAG Incident Simulator` dashboard is auto-provisioned and includes:

- current enrichment quality, failure rate, active incident flag
- enrichment p95 latency and throughput
- synthetic service latency, CPU, and error rate
