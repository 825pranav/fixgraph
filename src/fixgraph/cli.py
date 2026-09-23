"""FixGraph command-line interface (spec §12.4). Stages are added milestone by milestone."""

import logging

import typer
from pydantic import BaseModel

from fixgraph import __version__
from fixgraph.core.config import load_settings
from fixgraph.llm import ChatMessage, LLMRequest, complete_structured
from fixgraph.llm.factory import build_llm_client

app = typer.Typer(no_args_is_help=True, help="FixGraph: troubleshooting KG + GraphRAG benchmark.")
llm_app = typer.Typer(no_args_is_help=True, help="LLM backend utilities.")
app.add_typer(llm_app, name="llm")


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


if __name__ == "__main__":
    app()
