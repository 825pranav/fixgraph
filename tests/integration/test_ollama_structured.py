"""Opt-in: needs Ollama running with the extraction model pulled. `uv run poe test-integration`."""

import pytest
from pydantic import BaseModel

from fixgraph.core.config import load_settings
from fixgraph.llm import ChatMessage, LLMRequest, complete_structured
from fixgraph.llm.ollama import OllamaClient

pytestmark = pytest.mark.integration


class Extraction(BaseModel):
    product: str
    symptom: str


def test_ollama_returns_schema_valid_json() -> None:
    cfg = load_settings().llm
    client = OllamaClient(cfg.base_url, keep_alive=cfg.keep_alive, timeout_s=cfg.timeout_s)
    request = LLMRequest(
        model=cfg.extraction_model,
        messages=[
            ChatMessage(
                role="user",
                content=(
                    "Extract the product and symptom: 'My AirPods Pro won't connect to my Mac.'"
                ),
            )
        ],
        num_ctx=cfg.num_ctx,
    )
    try:
        out = complete_structured(client, request, Extraction)
    finally:
        client.close()
    assert "airpods" in out.product.lower()
    assert out.symptom
