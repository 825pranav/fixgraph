"""Backend request/response shapes, via httpx.MockTransport (no network).

Covers llm/ollama.py and llm/openai_compat.py, including HTTP error wrapping into LLMError.
"""

import json
from typing import Any

import httpx
import pytest

from fixgraph.llm import ChatMessage, LLMError, LLMRequest
from fixgraph.llm.ollama import OllamaClient
from fixgraph.llm.openai_compat import OpenAICompatClient

SCHEMA: dict[str, Any] = {"type": "object", "properties": {"a": {"type": "string"}}}
REQ = LLMRequest(
    model="qwen3:4b",
    messages=[ChatMessage(role="user", content="q")],
    num_ctx=8192,
    json_schema=SCHEMA,
    max_tokens=64,
)


def _capture(response: dict[str, Any], sink: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        sink.append(request)
        return httpx.Response(200, json=response)

    return httpx.MockTransport(handler)


def test_ollama_native_body_and_parse() -> None:
    seen: list[httpx.Request] = []
    reply = {
        "model": "qwen3:4b",
        "message": {"role": "assistant", "content": '{"a": "x"}'},
        "prompt_eval_count": 10,
        "eval_count": 5,
    }
    client = OllamaClient(
        "http://localhost:11434/v1", keep_alive="0", transport=_capture(reply, seen)
    )
    resp = client.complete(REQ)
    client.close()

    assert str(seen[0].url) == "http://localhost:11434/api/chat"  # /v1 stripped
    body = json.loads(seen[0].content)
    assert body["format"] == SCHEMA
    assert body["options"]["num_ctx"] == 8192
    assert body["options"]["num_predict"] == 64
    assert body["think"] is False and body["stream"] is False
    assert body["keep_alive"] == "0"
    assert resp.text == '{"a": "x"}'
    assert (resp.prompt_tokens, resp.completion_tokens) == (10, 5)


def test_openai_compat_body_and_parse() -> None:
    seen: list[httpx.Request] = []
    reply = {
        "model": "qwen3:4b",
        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }
    client = OpenAICompatClient("http://host/v1/", api_key="k", transport=_capture(reply, seen))
    resp = client.complete(REQ)
    client.close()

    assert str(seen[0].url) == "http://host/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer k"
    body = json.loads(seen[0].content)
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert body["reasoning_effort"] == "none"
    assert resp.text == "hi" and resp.completion_tokens == 1


def test_http_error_wrapped() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(500, text="boom"))
    client = OllamaClient(transport=transport)
    with pytest.raises(LLMError):
        client.complete(REQ)
    client.close()
