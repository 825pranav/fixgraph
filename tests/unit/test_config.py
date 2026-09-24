"""Settings precedence (YAML defaults, env overrides) and LLM client construction.

Covers core/config.py and llm/factory.py.
"""

from pathlib import Path

import pytest

from fixgraph.core.config import load_settings
from fixgraph.core.paths import REPO_ROOT
from fixgraph.llm import CachedLLMClient, FakeLLMClient
from fixgraph.llm.factory import build_llm_client


def test_yaml_defaults_loaded() -> None:
    s = load_settings()
    assert s.llm.num_ctx == 8192
    assert s.llm.think is False
    assert s.resolve(s.data_dir) == REPO_ROOT / "data"


def test_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM__BACKEND", "fake")
    monkeypatch.setenv("LLM__NUM_CTX", "4096")
    s = load_settings()
    assert s.llm.backend == "fake"
    assert s.llm.num_ctx == 4096


def test_factory_builds_fake_uncached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM__BACKEND", "fake")
    client = build_llm_client(load_settings(), use_cache=False)
    assert isinstance(client, FakeLLMClient)


def test_factory_wraps_in_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLM__BACKEND", "fake")
    monkeypatch.setenv("LLM__CACHE_PATH", str(tmp_path / "c.sqlite"))
    client = build_llm_client(load_settings())
    assert isinstance(client, CachedLLMClient)
    client.close()
