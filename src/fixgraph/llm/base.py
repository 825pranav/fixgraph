"""LLMClient protocol and the request/response models every backend speaks.

Implemented by llm.ollama, llm.openai_compat, llm.fake and wrapped by llm.cache. Every LLM
caller (kg extraction/resolve, retrieval.linking, answer, bench judge/generate) codes against
this protocol only. No fixgraph imports.
"""

# Imports: typing helpers for the protocol and pydantic for the request/response shapes.
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

# The three chat roles a message can have, same as the Ollama and OpenAI chat APIs.
Role = Literal["system", "user", "assistant"]


# One chat turn (who is speaking plus the text); a request carries a list of these.
class ChatMessage(BaseModel):
    role: Role
    content: str


# Everything sent to a model: model name, messages, sampling settings and an optional JSON schema
# that forces structured output. Backends translate this into their own HTTP payload.
class LLMRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int | None = None
    num_ctx: int = 8192
    think: bool = False
    json_schema: dict[str, Any] | None = Field(
        default=None, description="JSON Schema the output must satisfy (structured output)."
    )
    seed: int | None = None

    # Dump the whole request to JSON; the cache hashes it, so any changed setting is a new key.
    def cache_payload(self) -> dict[str, Any]:
        """Everything that influences the output; used as the cache key."""
        return self.model_dump(mode="json")


# What every backend hands back: the reply text plus token counts, latency and a cached flag.
class LLMResponse(BaseModel):
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float = 0.0
    cached: bool = False


# Single error type for any backend failure, so callers like answer_question can catch one thing.
class LLMError(RuntimeError):
    """Transport or protocol failure talking to an LLM backend."""


# The interface every backend (Ollama, OpenAI-compatible, fake, cache wrapper) must satisfy.
@runtime_checkable
class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...

    def close(self) -> None: ...
