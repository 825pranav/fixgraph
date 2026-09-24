"""Build the configured LLMClient (optionally cached).

Picks the backend from `settings.llm` (Ollama native, OpenAI-compatible or fake) and wraps it in
llm.cache. Used by: cli.py (`llm smoke`), the kg and bench CLIs, and api/app.py.
Uses: core.config.Settings, llm.ollama, llm.openai_compat, llm.fake, llm.cache.
"""

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
