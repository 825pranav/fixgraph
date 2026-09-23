"""LLMClient protocol and the request/response models every backend speaks."""

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str


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

    def cache_payload(self) -> dict[str, Any]:
        """Everything that influences the output; used as the cache key."""
        return self.model_dump(mode="json")


class LLMResponse(BaseModel):
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float = 0.0
    cached: bool = False


class LLMError(RuntimeError):
    """Transport or protocol failure talking to an LLM backend."""


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...

    def close(self) -> None: ...
