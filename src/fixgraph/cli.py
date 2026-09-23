"""FixGraph command-line interface (spec §12.4). Stages are added milestone by milestone."""

import logging

import typer
from pydantic import BaseModel

from fixgraph import __version__
from fixgraph.bench.cli import app as bench_app
from fixgraph.bench.cli import index_app
from fixgraph.core.config import load_settings
from fixgraph.gnn.cli import app as gnn_app
from fixgraph.ingest.cli import app as ingest_app
from fixgraph.kg.cli import app as kg_app
from fixgraph.llm import ChatMessage, LLMRequest, complete_structured
from fixgraph.llm.factory import build_llm_client

app = typer.Typer(no_args_is_help=True, help="FixGraph: troubleshooting KG + GraphRAG benchmark.")
llm_app = typer.Typer(no_args_is_help=True, help="LLM backend utilities.")
app.add_typer(llm_app, name="llm")
app.add_typer(ingest_app, name="ingest")
app.add_typer(kg_app, name="kg")
app.add_typer(gnn_app, name="gnn")
app.add_typer(index_app, name="index")
app.add_typer(bench_app, name="bench")


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


class _SmokeExtraction(BaseModel):
    product: str
    component: str
    symptom: str


@llm_app.command("smoke")
def llm_smoke(
    model: str | None = typer.Option(None, help="Defaults to llm.extraction_model."),
    no_cache: bool = typer.Option(False, help="Bypass the SQLite cache."),
) -> None:
    """Send one schema-constrained request and print the validated object."""
    settings = load_settings()
    client = build_llm_client(settings, use_cache=not no_cache)
    request = LLMRequest(
        model=model or settings.llm.extraction_model,
        messages=[
            ChatMessage(
                role="user",
                content=(
                    "Extract the product, component and symptom from: "
                    "'My Apple Watch won't pair over Bluetooth since I updated my iPhone.'"
                ),
            )
        ],
        num_ctx=settings.llm.num_ctx,
        temperature=settings.llm.temperature,
        think=settings.llm.think,
    )
    try:
        result = complete_structured(client, request, _SmokeExtraction)
    finally:
        client.close()
    typer.echo(result.model_dump_json(indent=2))


@app.command()
def serve(host: str = typer.Option("127.0.0.1"), port: int = typer.Option(8000)) -> None:
    """Run the FastAPI service (fake mode when LLM__BACKEND=fake or no corpus exists)."""
    import uvicorn

    from fixgraph.api.app import create_app

    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    app()
