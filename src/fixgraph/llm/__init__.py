"""LLM access: protocol, backends (Ollama native, OpenAI-compatible, fake), cache."""

from fixgraph.llm.base import ChatMessage, LLMClient, LLMError, LLMRequest, LLMResponse
from fixgraph.llm.cache import CachedLLMClient, SQLiteCache
from fixgraph.llm.fake import FakeLLMClient
from fixgraph.llm.structured import StructuredOutputError, complete_structured

__all__ = [
    "CachedLLMClient",
    "ChatMessage",
    "FakeLLMClient",
    "LLMClient",
    "LLMError",
    "LLMRequest",
    "LLMResponse",
    "SQLiteCache",
    "StructuredOutputError",
    "complete_structured",
]
