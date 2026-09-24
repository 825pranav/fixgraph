"""SQLite response cache: stable keys, cache hits, persistence across reopen, unicode.

Covers llm/cache.py.
"""

from pathlib import Path

from fixgraph.llm import CachedLLMClient, ChatMessage, FakeLLMClient, LLMRequest, SQLiteCache
from fixgraph.llm.cache import cache_key


def _req(content: str = "hi", **kw: object) -> LLMRequest:
    return LLMRequest(model="m", messages=[ChatMessage(role="user", content=content)], **kw)  # type: ignore[arg-type]


def test_key_is_stable_and_param_sensitive() -> None:
    assert cache_key(_req()) == cache_key(_req())
    assert cache_key(_req()) != cache_key(_req("other"))
    assert cache_key(_req()) != cache_key(_req(temperature=0.5))
    assert cache_key(_req()) != cache_key(_req(json_schema={"type": "object"}))


def test_second_call_served_from_cache(tmp_path: Path) -> None:
    fake = FakeLLMClient(responder=lambda r: "answer")
    client = CachedLLMClient(fake, SQLiteCache(tmp_path / "c.sqlite"))
    first = client.complete(_req())
    second = client.complete(_req())
    client.close()
    assert first.text == second.text == "answer"
    assert not first.cached and second.cached
    assert len(fake.calls) == 1


def test_cache_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    with SQLiteCache(path) as cache:
        CachedLLMClient(FakeLLMClient(responder=lambda r: "x"), cache).complete(_req())
    fake = FakeLLMClient(responder=lambda r: "y")
    with SQLiteCache(path) as cache:
        resp = CachedLLMClient(fake, cache).complete(_req())
        assert len(cache) == 1
    assert resp.text == "x" and resp.cached
    assert fake.calls == []


def test_unicode_roundtrip(tmp_path: Path) -> None:
    with SQLiteCache(tmp_path / "c.sqlite") as cache:
        client = CachedLLMClient(FakeLLMClient(responder=lambda r: "Réglages → Bluetooth ✓"), cache)
        client.complete(_req("Apple Watch Series 10 — ne s'appaire pas"))
        hit = client.complete(_req("Apple Watch Series 10 — ne s'appaire pas"))
    assert hit.text == "Réglages → Bluetooth ✓"
