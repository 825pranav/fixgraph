"""Client for any OpenAI-compatible /v1/chat/completions endpoint (vLLM, hosted, Ollama /v1)."""

import time
from typing import Any

import httpx

from fixgraph.llm.base import LLMError, LLMRequest, LLMResponse


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "none",
        timeout_s: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
        )

    def build_body(self, request: LLMRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.seed is not None:
            body["seed"] = request.seed
        if not request.think:
            body["reasoning_effort"] = "none"
        if request.json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "strict": True, "schema": request.json_schema},
            }
        return body

    def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        try:
            resp = self._http.post("/chat/completions", json=self.build_body(request))
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"OpenAI-compatible request failed: {exc}") from exc
        data = resp.json()
        usage = data.get("usage") or {}
        return LLMResponse(
            text=data["choices"][0]["message"]["content"] or "",
            model=data.get("model", request.model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_s=time.perf_counter() - start,
        )

    def close(self) -> None:
        self._http.close()
