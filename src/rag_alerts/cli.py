import json
from pathlib import Path

import typer
from dotenv import load_dotenv

from rag_alerts.http_service import create_enrichment_app
from rag_alerts.models import Alert, EnrichmentResult
from rag_alerts.pipeline import RAGAlertPipeline

app = typer.Typer(help="RAG alert enrichment workshop CLI")
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / ".artifacts"
DEFAULT_ALERT_FILE = DATA_DIR / "sample_alert.json"


def _load_alert(path: Path) -> Alert:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Alert(**payload)


def _build_index() -> tuple[int, Path]:
    load_dotenv()
    pipeline = RAGAlertPipeline(data_dir=DATA_DIR, index_dir=INDEX_DIR)
    count = pipeline.build()
    return count, pipeline.index_path


@app.command("build-index")
def build_index() -> None:
    """Build vector index from incident knowledge base."""
    count, index_path = _build_index()
    typer.echo(f"Built index for {count} incidents at {index_path}")


def _enrich_alert(alert_file: Path, top_k: int, rerank_k: int) -> EnrichmentResult:
    load_dotenv()
    pipeline = RAGAlertPipeline(data_dir=DATA_DIR, index_dir=INDEX_DIR)
    if not pipeline.index_path.exists():
        _build_index()

    alert = _load_alert(alert_file)
    return pipeline.enrich(alert=alert, top_k=top_k, rerank_k=rerank_k)


@app.command("enrich-alert")
def enrich_alert(
    alert_file: Path = typer.Option(DEFAULT_ALERT_FILE, exists=True),
    top_k: int = typer.Option(5, min=1, max=20),
    rerank_k: int = typer.Option(3, min=1, max=20),
) -> None:
    """Enrich alert using retrieval, reranking, and LLM response."""
    result = _enrich_alert(alert_file=alert_file, top_k=top_k, rerank_k=rerank_k)

    typer.echo("\n=== Top Retrieved Incidents ===")
    for item in result.retrieval:
        typer.echo(
            f"- {item.incident.incident_id} | score={item.score:.4f} | {item.incident.summary}"
        )

    typer.echo("\n=== Enriched Alert Brief ===")
    typer.echo(result.final_response)


@app.command("run-demo")
def run_demo() -> None:
    """Build index and run enrichment on default sample alert."""
    count, index_path = _build_index()
    typer.echo(f"Built index for {count} incidents at {index_path}")
    result = _enrich_alert(alert_file=DEFAULT_ALERT_FILE, top_k=5, rerank_k=3)
    typer.echo("\n=== Top Retrieved Incidents ===")
    for item in result.retrieval:
        typer.echo(
            f"- {item.incident.incident_id} | score={item.score:.4f} | {item.incident.summary}"
        )
    typer.echo("\n=== Enriched Alert Brief ===")
    typer.echo(result.final_response)


@app.command("serve-http")
def serve_http(
    host: str = typer.Option("0.0.0.0", help="Bind host for HTTP API."),
    port: int = typer.Option(8081, min=1, max=65535, help="Bind port for HTTP API."),
    top_k: int = typer.Option(5, min=1, max=20, help="Top-k retrieval count."),
    rerank_k: int = typer.Option(3, min=1, max=20, help="Top-k items for rerank stage."),
) -> None:
    """Run alert enrichment as an HTTP service."""
    load_dotenv()
    app_server = create_enrichment_app(
        data_dir=DATA_DIR,
        index_dir=INDEX_DIR,
        top_k=top_k,
        rerank_k=rerank_k,
    )
    typer.echo(f"Starting enrichment API on http://{host}:{port}")
    app_server.run(host=host, port=port)


if __name__ == "__main__":
    app()
