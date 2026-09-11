"""Scraper: sitemap parsing, article ids, robots.txt, caching and User-Agent (mock transport).

Covers ingest/scrape.py.
"""

from pathlib import Path

import httpx

from fixgraph.ingest.scrape import Scraper, article_id_from_url, parse_sitemap

ROBOTS = "User-agent: *\nDisallow: /en-us/666\n"


def test_parse_sitemap() -> None:
    xml = "<urlset><url><loc> https://a/1 </loc></url><url><loc>https://a/2</loc></url></urlset>"
    assert parse_sitemap(xml) == ["https://a/1", "https://a/2"]


def test_article_id_from_url() -> None:
    assert article_id_from_url("https://support.apple.com/en-us/108806") == "108806"
    assert article_id_from_url("https://support.apple.com/iphone/repair") is None
    assert article_id_from_url("https://support.apple.com/en-us/HT204938") is None


def _transport(seen: list[str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=ROBOTS)
        if request.url.path.endswith("/500"):
            return httpx.Response(500)
        return httpx.Response(200, text=f"<html>{request.url.path}</html>")

    return httpx.MockTransport(handler)


def test_fetch_all_respects_robots_caches_and_counts(tmp_path: Path) -> None:
    seen: list[str] = []
    scraper = Scraper(tmp_path, min_interval_s=0, transport=_transport(seen))
    urls = [f"https://support.apple.com/en-us/{i}" for i in (1, 666, 500)]
    stats = scraper.fetch_all(urls)
    assert stats == {"fetched": 1, "cached": 0, "skipped_robots": 1, "failed": 1}
    assert (tmp_path / "1.html").read_text(encoding="utf-8") == "<html>/en-us/1</html>"
    assert "/en-us/666" not in seen

    seen.clear()
    assert scraper.fetch_all(urls[:1]) == {
        "fetched": 0,
        "cached": 1,
        "skipped_robots": 0,
        "failed": 0,
    }
    assert seen == []
    scraper.close()


def test_user_agent_is_descriptive(tmp_path: Path) -> None:
    agents: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        agents.append(request.headers["user-agent"])
        return httpx.Response(200, text="User-agent: *\nAllow: /\n")

    scraper = Scraper(tmp_path, min_interval_s=0, transport=httpx.MockTransport(handler))
    scraper.robots()
    scraper.close()
    assert "FixGraph" in agents[0] and "@" in agents[0]
