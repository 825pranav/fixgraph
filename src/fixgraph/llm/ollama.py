"""Ollama native /api/chat client.

The OpenAI-compatible /v1 endpoint ignores `num_ctx` (verified on Ollama 0.34: the model
loads with a 4096 context), so the default backend uses the native API, which honours
`num_ctx`, `think` and `keep_alive`. See docs/DECISIONS.md.

Used by: llm.factory (default backend); the kg and bench CLIs also use it directly to unload
models between stages. Uses: llm.base, httpx.
"""

import time
from typing import Any

import httpx

from fixgraph.llm.base import LLMError, LLMRequest, LLMResponse


def _native_root(base_url: str) -> str:
    root = base_url.rstrip("/")
    return root.removesuffix("/v1")


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        keep_alive: str = "2m",
        timeout_s: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.keep_alive = keep_alive
        self._http = httpx.Client(
            base_url=_native_root(base_url), timeout=timeout_s, transport=transport
        )

    def build_body(self, request: LLMRequest) -> dict[str, Any]:
        options: dict[str, Any] = {"num_ctx": request.num_ctx, "temperature": request.temperature}
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if request.seed is not None:
            options["seed"] = request.seed
        body: dict[str, Any] = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "stream": False,
            "think": request.think,
            "keep_alive": self.keep_alive,
            "options": options,
        }
        if request.json_schema is not None:
            body["format"] = request.json_schema
        return body

    def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        try:
            resp = self._http.post("/api/chat", json=self.build_body(request))
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        data = resp.json()
        return LLMResponse(
            text=data["message"]["content"],
            model=data.get("model", request.model),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            latency_s=time.perf_counter() - start,
        )

    def unload(self, model: str) -> None:
        """Free VRAM immediately (spec §5.4: one heavy model on the GPU at a time)."""
        try:
            self._http.post("/api/generate", json={"model": model, "keep_alive": 0})
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama unload failed: {exc}") from exc

    def close(self) -> None:
        self._http.close()
