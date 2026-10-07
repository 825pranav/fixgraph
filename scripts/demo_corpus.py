"""Fetch a small demo corpus: only the articles the dev questions and the gold set point at.

The full corpus takes ~35 min to scrape and lives on the machine that built it. For a demo on
another laptop this fetches the ~70 articles behind data/bench/dev_handwritten.jsonl and
data/gold/gold_chunk_ids.txt (same polite Scraper: robots.txt, 1 req/s, cached), so the 30 dev
questions are answerable. Then run the usual steps:

    uv run python scripts/demo_corpus.py
    uv run fixgraph ingest parse; uv run fixgraph ingest chunk
    uv run fixgraph index build
    uv run fixgraph serve

Uses: ingest.scrape, core.config. Not used by the pipeline itself.
"""

# Imports: json for the question file, plus the scraper and settings the ingest CLI also uses.
import json

from fixgraph.core.config import load_settings
from fixgraph.ingest.scrape import BASE, Scraper


# Article ids named by the gold chunk list and by every dev question's gold chunks.
def demo_article_ids() -> list[str]:
    paths = load_settings().paths
    ids = {cid.split(":")[0] for cid in (paths.gold / "gold_chunk_ids.txt").read_text().split()}
    for line in (paths.bench / "dev_handwritten.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids |= {cid.split(":")[0] for cid in json.loads(line)["gold_chunk_ids"]}
    return sorted(ids, key=int)


# Fetch those pages into data/raw/html; pages already cached are skipped.
def main() -> None:
    ids = demo_article_ids()
    print(f"{len(ids)} demo articles (~{len(ids) // 60 + 1} min at 1 request/s)")
    scraper = Scraper(load_settings().paths.raw_html)
    try:
        print(json.dumps(scraper.fetch_all(f"{BASE}/en-us/{i}" for i in ids), indent=2))
    finally:
        scraper.close()


if __name__ == "__main__":
    main()
