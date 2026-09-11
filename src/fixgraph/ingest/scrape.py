"""Polite scraper for support.apple.com (spec §6.1).

robots.txt is obeyed, requests are rate-limited to <= 1/s with a descriptive User-Agent, and raw
HTML is cached under data/raw/html so reruns only fetch what's missing.

Used by: `fixgraph ingest scrape` (ingest/cli.py). Uses httpx only; no fixgraph imports.
"""

import logging
import re
import time
from collections.abc import Iterable
from pathlib import Path
from urllib.robotparser import RobotFileParser

import httpx
from tqdm import tqdm

logger = logging.getLogger(__name__)

BASE = "https://support.apple.com"
SITEMAP_INDEX = f"{BASE}/en-us/sitemaps/sitemap-index-en-us.xml"
USER_AGENT = "FixGraph-research-bot/0.1 (student research project; pranavnegi@gmail.com)"
ARTICLE_URL_RE = re.compile(r"^https://support\.apple\.com/en-us/(\d+)$")
_LOC_RE = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>")


def parse_sitemap(xml: str) -> list[str]:
    """Return every <loc> URL in a sitemap or sitemap index, in document order."""
    return _LOC_RE.findall(xml)


def article_id_from_url(url: str) -> str | None:
    m = ARTICLE_URL_RE.match(url.strip())
    return m.group(1) if m else None


class RateLimiter:
    def __init__(self, min_interval_s: float = 1.0) -> None:
        self.min_interval_s = min_interval_s
        self._last = 0.0

    def wait(self) -> None:
        delay = self._last + self.min_interval_s - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()


class Scraper:
    def __init__(
        self,
        raw_dir: Path,
        min_interval_s: float = 1.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._limiter = RateLimiter(min_interval_s)
        self._http = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=30.0,
            follow_redirects=True,
            transport=transport,
        )
        self._robots: RobotFileParser | None = None

    def _get(self, url: str) -> httpx.Response:
        self._limiter.wait()
        return self._http.get(url)

    def robots(self) -> RobotFileParser:
        if self._robots is None:
            rp = RobotFileParser()
            rp.parse(self._get(f"{BASE}/robots.txt").text.splitlines())
            self._robots = rp
        return self._robots

    def allowed(self, url: str) -> bool:
        return self.robots().can_fetch(USER_AGENT, url)

    def article_urls(self) -> list[str]:
        """All en-us article URLs from the sitemap index, deduplicated, sorted by article id."""
        urls: set[str] = set()
        for sitemap in parse_sitemap(self._get(SITEMAP_INDEX).text):
            for url in parse_sitemap(self._get(sitemap).text):
                if article_id_from_url(url):
                    urls.add(url)
        return sorted(urls, key=lambda u: int(article_id_from_url(u) or 0))

    def html_path(self, article_id: str) -> Path:
        return self.raw_dir / f"{article_id}.html"

    def fetch_all(self, urls: Iterable[str], limit: int | None = None) -> dict[str, int]:
        """Fetch and cache each article. Returns counts: fetched / cached / skipped / failed."""
        stats = {"fetched": 0, "cached": 0, "skipped_robots": 0, "failed": 0}
        todo = list(urls)[:limit] if limit else list(urls)
        for url in tqdm(todo, desc="scrape", unit="page"):
            article_id = article_id_from_url(url)
            if article_id is None:
                continue
            path = self.html_path(article_id)
            if path.exists():
                stats["cached"] += 1
                continue
            if not self.allowed(url):
                stats["skipped_robots"] += 1
                continue
            try:
                resp = self._get(url)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("fetch failed %s: %s", url, exc)
                stats["failed"] += 1
                continue
            tmp = path.with_suffix(".tmp")
            tmp.write_text(resp.text, encoding="utf-8")
            tmp.replace(path)  # atomic: a crash never leaves a half-written page
            stats["fetched"] += 1
        return stats

    def close(self) -> None:
        self._http.close()
