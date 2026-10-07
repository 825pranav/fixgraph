"""`fixgraph ingest scrape | parse | chunk | subset | links`.

scrape -> data/raw/html (ingest.scrape); parse + select -> articles.parquet (ingest.parse,
ingest.select); chunk -> chunks.parquet (ingest.chunk); subset trims the corpus to a chunk
budget; links -> links.parquet (ingest.links). Used by: cli.py (mounted as `ingest`).
Uses: core.config, core.ontology, ingest.store.
"""

# Imports: typer builds the commands; each ingest step lives in its own module.
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
from fixgraph.ingest.select import select_corpus, subset_by_chunk_budget
from fixgraph.ingest.store import read_articles, read_chunks, write_articles, write_chunks

logger = logging.getLogger(__name__)
# The `fixgraph ingest` sub-app; the root cli.py mounts it.
app = typer.Typer(no_args_is_help=True, help="Corpus ingestion: scrape -> parse -> chunk.")


# Step 1: read the sitemap and download article HTML into data/raw/html.
@app.command()
def scrape(
    limit: int | None = typer.Option(None, help="Fetch at most N articles."),
    dry_run: bool = typer.Option(False, help="List URLs and projected time; fetch nothing."),
    interval: float = typer.Option(1.0, help="Seconds between requests (spec: >= 1)."),
) -> None:
    """Fetch en-us support articles from the sitemap into data/raw/html (cached, resumable)."""
    # Get the article URL list from the sitemap and save it, so later steps know the full set.
    paths = load_settings().paths
    scraper = Scraper(paths.raw_html, min_interval_s=max(interval, 1.0))
    try:
        urls = scraper.article_urls()
        paths.article_urls.write_text(json.dumps(urls, indent=0), encoding="utf-8")
        # Work out which pages are still missing and print how long fetching them will take.
        todo = urls[:limit] if limit else urls
        missing = [u for u in todo if not scraper.html_path(u.rsplit("/", 1)[-1]).exists()]
        typer.echo(
            f"{len(urls)} article URLs; {len(missing)} to fetch "
            f"(~{len(missing) * max(interval, 1.0) / 60:.0f} min at {interval:.1f}s/request)"
        )
        # Dry run stops here; otherwise fetch, print the counts, and always close the HTTP client.
        if dry_run:
            return
        typer.echo(json.dumps(scraper.fetch_all(todo), indent=2))
    finally:
        scraper.close()


# Step 2: parse cached HTML into Articles, keep the best troubleshooting ones, save to parquet.
@app.command()
def parse(
    target: int = typer.Option(1000, help="Number of articles to keep in the corpus."),
    limit: int | None = typer.Option(None, help="Parse at most N cached pages."),
) -> None:
    """Parse cached HTML, select the research corpus, write data/corpus/articles.parquet."""
    paths = load_settings().paths
    ontology = load_ontology()
    # Parse every cached page in article-id order, dropping pages that fail to parse.
    files = sorted(paths.raw_html.glob("*.html"), key=lambda p: int(p.stem))
    files = files[:limit] if limit else files
    parsed = [a for f in tqdm(files, desc="parse") if (a := parse_file(f, ontology)) is not None]
    # Select the corpus and write articles.parquet, then print a product family summary.
    chosen = select_corpus(parsed, ontology, target)
    write_articles(chosen, paths.articles)
    families = Counter(t for a in chosen for t in a.product_tags)
    typer.echo(
        f"parsed {len(parsed)}/{len(files)} pages; kept {len(chosen)} -> {paths.articles}\n"
        f"family mentions: {dict(families.most_common())}"
    )


# Step 3: cut every saved article into chunks and write chunks.parquet.
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
    # Print chunk size stats as a quick sanity check.
    sizes = sorted(c.n_tokens for c in chunks)
    typer.echo(
        f"{len(chunks)} chunks from {len(articles)} articles -> {paths.chunks}\n"
        f"tokens: min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}, "
        f"total {sum(sizes):,}"
    )


# Step 4: trim the corpus to a chunk budget for faster runs, keeping gold-set articles.
@app.command()
def subset(
    max_chunks: int = typer.Option(1000, help="Chunk budget for the working corpus."),
) -> None:
    """Shrink the working corpus to whole articles under a chunk budget (time-boxed runs).

    The full corpus is kept as articles_full.parquet / chunks_full.parquet; gold-set articles are
    always kept. Chunk IDs are unchanged, so existing extractions stay valid.
    """
    import shutil

    # Back up the full corpus once, and always subset from that full copy.
    paths = load_settings().paths
    full_articles = paths.corpus / "articles_full.parquet"
    full_chunks = paths.corpus / "chunks_full.parquet"
    if not full_articles.exists():
        shutil.copyfile(paths.articles, full_articles)
        shutil.copyfile(paths.chunks, full_chunks)
    articles = read_articles(full_articles)
    chunks = read_chunks(full_chunks)
    # Count chunks per article and load the article ids that the gold set needs.
    counts = Counter(c.article_id for c in chunks)
    gold_ids = paths.gold / "gold_chunk_ids.txt"
    must = (
        {line.split(":")[0] for line in gold_ids.read_text(encoding="utf-8").split()}
        if gold_ids.exists()
        else set()
    )
    # Pick the articles to keep and overwrite the working articles and chunks files.
    kept = subset_by_chunk_budget(articles, counts, load_ontology(), max_chunks, must)
    keep_ids = {a.article_id for a in kept}
    kept_chunks = [c for c in chunks if c.article_id in keep_ids]
    write_articles(kept, paths.articles)
    write_chunks(kept_chunks, paths.chunks)
    typer.echo(
        f"kept {len(kept)}/{len(articles)} articles, {len(kept_chunks)}/{len(chunks)} chunks "
        f"({len(must)} gold articles always kept)"
    )


# Optional step: record Apple's own article-to-article links for the graph and bridge questions.
@app.command()
def links() -> None:
    """Hyperlinks between corpus articles, with the sentence and chunk they sit in (D34)."""
    from fixgraph.ingest.links import attach_chunks, extract_links, write_links

    # Extract links from each article's cached HTML, keeping only targets inside the corpus.
    paths = load_settings().paths
    articles = read_articles(paths.articles)
    ids = {a.article_id for a in articles}
    found = []
    for a in articles:
        html = (paths.raw_html / f"{a.article_id}.html").read_text(encoding="utf-8")
        found += extract_links(html, a.article_id, ids)
    # Match each link sentence to its source chunk, then save links.parquet.
    found = attach_chunks(found, read_chunks(paths.chunks))
    out = paths.corpus / "links.parquet"
    write_links(found, out)
    in_chunks = sum(bool(x.src_chunk_id) for x in found)
    typer.echo(f"{len(found)} links ({in_chunks} located in a chunk) -> {out}")
