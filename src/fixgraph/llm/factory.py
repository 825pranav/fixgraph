"""Build the configured LLMClient (optionally cached)."""

from fixgraph.core.config import Settings
from fixgraph.llm.base import LLMClient
from fixgraph.llm.cache import CachedLLMClient, SQLiteCache
from fixgraph.llm.fake import FakeLLMClient
from fixgraph.llm.ollama import OllamaClient
from fixgraph.llm.openai_compat import OpenAICompatClient


def build_llm_client(settings: Settings, use_cache: bool = True) -> LLMClient:
    cfg = settings.llm
    client: LLMClient
    if cfg.backend == "fake":
        client = FakeLLMClient()
    elif cfg.backend == "ollama":
        client = OllamaClient(cfg.base_url, keep_alive=cfg.keep_alive, timeout_s=cfg.timeout_s)
    else:
        client = OpenAICompatClient(cfg.base_url, api_key=cfg.api_key, timeout_s=cfg.timeout_s)
    if use_cache:
        client = CachedLLMClient(client, SQLiteCache(settings.resolve(cfg.cache_path)))
    return client
