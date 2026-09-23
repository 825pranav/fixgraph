"""`fixgraph ingest scrape | parse | chunk`."""

import json
import logging
from collections import Counter

import typer
from tqdm import tqdm

from fixgraph.core.config import load_settings
from fixgraph.core.ontology import load_ontology
from fixgraph.ingest.chunk import chunk_article
from fixgraph.ingest.parse import parse_file
from fixgraph.ingest.scrape import Scraper
from fixgraph.ingest.select import select_corpus
from fixgraph.ingest.store import read_articles, write_articles, write_chunks

logger = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=True, help="Corpus ingestion: scrape -> parse -> chunk.")


@app.command()
def scrape(
    limit: int | None = typer.Option(None, help="Fetch at most N articles."),
    dry_run: bool = typer.Option(False, help="List URLs and projected time; fetch nothing."),
    interval: float = typer.Option(1.0, help="Seconds between requests (spec: >= 1)."),
) -> None:
    """Fetch en-us support articles from the sitemap into data/raw/html (cached, resumable)."""
    paths = load_settings().paths
    scraper = Scraper(paths.raw_html, min_interval_s=max(interval, 1.0))
    try:
        urls = scraper.article_urls()
        paths.article_urls.write_text(json.dumps(urls, indent=0), encoding="utf-8")
        todo = urls[:limit] if limit else urls
        missing = [u for u in todo if not scraper.html_path(u.rsplit("/", 1)[-1]).exists()]
        typer.echo(
            f"{len(urls)} article URLs; {len(missing)} to fetch "
            f"(~{len(missing) * max(interval, 1.0) / 60:.0f} min at {interval:.1f}s/request)"
        )
        if dry_run:
            return
        typer.echo(json.dumps(scraper.fetch_all(todo), indent=2))
    finally:
        scraper.close()


@app.command()
def parse(
    target: int = typer.Option(1000, help="Number of articles to keep in the corpus."),
    limit: int | None = typer.Option(None, help="Parse at most N cached pages."),
) -> None:
    """Parse cached HTML, select the research corpus, write data/corpus/articles.parquet."""
    paths = load_settings().paths
    ontology = load_ontology()
    files = sorted(paths.raw_html.glob("*.html"), key=lambda p: int(p.stem))
    files = files[:limit] if limit else files
    parsed = [a for f in tqdm(files, desc="parse") if (a := parse_file(f, ontology)) is not None]
    chosen = select_corpus(parsed, ontology, target)
    write_articles(chosen, paths.articles)
    families = Counter(t for a in chosen for t in a.product_tags)
    typer.echo(
        f"parsed {len(parsed)}/{len(files)} pages; kept {len(chosen)} -> {paths.articles}\n"
        f"family mentions: {dict(families.most_common())}"
    )


@app.command()
def chunk(
    min_tokens: int = typer.Option(200),
    max_tokens: int = typer.Option(500),
) -> None:
    """Chunk the corpus into data/corpus/chunks.parquet with stable IDs."""
    paths = load_settings().paths
    articles = read_articles(paths.articles)
    chunks = [
        c for a in articles for c in chunk_article(a, min_tokens=min_tokens, max_tokens=max_tokens)
    ]
    write_chunks(chunks, paths.chunks)
    sizes = sorted(c.n_tokens for c in chunks)
    typer.echo(
        f"{len(chunks)} chunks from {len(articles)} articles -> {paths.chunks}\n"
        f"tokens: min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}, "
        f"total {sum(sizes):,}"
    )
