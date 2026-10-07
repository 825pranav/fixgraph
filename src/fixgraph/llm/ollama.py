"""Ollama native /api/chat client.

The OpenAI-compatible /v1 endpoint ignores `num_ctx` (verified on Ollama 0.34: the model
loads with a 4096 context), so the default backend uses the native API, which honours
`num_ctx`, `think` and `keep_alive`. See docs/DECISIONS.md.

Used by: llm.factory (default backend); the kg and bench CLIs also use it directly to unload
models between stages. Uses: llm.base, httpx.
"""

# Imports: timing for latency, httpx for HTTP, and the shared request/response types.
import time
from typing import Any

import httpx

from fixgraph.llm.base import LLMError, LLMRequest, LLMResponse


# Strip a trailing "/v1" so the client always hits Ollama's native API root.
def _native_root(base_url: str) -> str:
    root = base_url.rstrip("/")
    return root.removesuffix("/v1")


# Default backend: talks to Ollama's native /api/chat so num_ctx, think and keep_alive work.
class OllamaClient:
    # Keep the keep_alive setting and open one reusable HTTP client pointed at the native root.
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

    # Translate our LLMRequest into Ollama's body; model-level knobs go inside "options".
    def build_body(self, request: LLMRequest) -> dict[str, Any]:
        options: dict[str, Any] = {"num_ctx": request.num_ctx, "temperature": request.temperature}
        # Optional limits only go in when set, so Ollama keeps its defaults otherwise.
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if request.seed is not None:
            options["seed"] = request.seed
        # Main body: non-streaming chat, thinking on/off, and how long to keep the model loaded.
        body: dict[str, Any] = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "stream": False,
            "think": request.think,
            "keep_alive": self.keep_alive,
            "options": options,
        }
        # Structured output: Ollama's `format` field takes the JSON schema and constrains decoding.
        if request.json_schema is not None:
            body["format"] = request.json_schema
        return body

    # Send one chat request to Ollama and return the reply text, token counts and latency.
    def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()
        # If Ollama is down or errors, raise LLMError; answer_question catches it and abstains.
        try:
            resp = self._http.post("/api/chat", json=self.build_body(request))
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        # Pull the message text and Ollama's prompt/eval token counts out of the JSON reply.
        data = resp.json()
        return LLMResponse(
            text=data["message"]["content"],
            model=data.get("model", request.model),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            latency_s=time.perf_counter() - start,
        )

    # Ask Ollama to drop a model from GPU memory now, so the next stage's model fits on the GPU.
    def unload(self, model: str) -> None:
        """Free VRAM immediately (spec §5.4: one heavy model on the GPU at a time)."""
        try:
            self._http.post("/api/generate", json={"model": model, "keep_alive": 0})
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama unload failed: {exc}") from exc

    # Close the underlying HTTP connection pool.
    def close(self) -> None:
        self._http.close()
