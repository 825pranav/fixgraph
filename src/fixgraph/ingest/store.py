"""Parquet persistence for articles and chunks.

Written by ingest/cli.py; read by the kg and bench CLIs and api/app.py.
Uses: core.models (Article, ArticleSection, Chunk).
"""

# Imports: polars writes and reads parquet; core models define the row shapes.
import json
from pathlib import Path

import polars as pl

from fixgraph.core.models import Article, ArticleSection, Chunk


# Save parsed articles to articles.parquet, one row per article.
def write_articles(articles: list[Article], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Flatten each Article into a row; nested sections are stored as one JSON string column.
    rows = [
        {
            "article_id": a.article_id,
            "url": a.url,
            "title": a.title,
            "summary": a.summary,
            "product_tags": a.product_tags,
            "os_versions_mentioned": a.os_versions_mentioned,
            "last_updated": a.last_updated,
            "sections_json": json.dumps([s.model_dump() for s in a.sections], ensure_ascii=False),
        }
        for a in articles
    ]
    # Fixed column types so an empty list or None still gets the right parquet type.
    schema = {
        "article_id": pl.String,
        "url": pl.String,
        "title": pl.String,
        "summary": pl.String,
        "product_tags": pl.List(pl.String),
        "os_versions_mentioned": pl.List(pl.String),
        "last_updated": pl.String,
        "sections_json": pl.String,
    }
    pl.DataFrame(rows, schema=schema).write_parquet(path)


# Load articles.parquet back into Article objects, rebuilding sections from the JSON column.
def read_articles(path: Path) -> list[Article]:
    df = pl.read_parquet(path)
    out: list[Article] = []
    for row in df.iter_rows(named=True):
        sections = [ArticleSection.model_validate(s) for s in json.loads(row["sections_json"])]
        out.append(
            Article(
                article_id=row["article_id"],
                url=row["url"],
                title=row["title"],
                summary=row["summary"],
                product_tags=row["product_tags"],
                os_versions_mentioned=row["os_versions_mentioned"],
                last_updated=row["last_updated"],
                sections=sections,
            )
        )
    return out


# Save chunks to chunks.parquet with a fixed schema; chunks are flat, so no JSON is needed.
def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([c.model_dump() for c in chunks], schema=_CHUNK_SCHEMA).write_parquet(path)


# Load chunks.parquet back into Chunk objects; used by KG extraction, the index build and the API.
def read_chunks(path: Path) -> list[Chunk]:
    return [Chunk.model_validate(r) for r in pl.read_parquet(path).iter_rows(named=True)]


# Column types for chunks.parquet, matching the Chunk model fields.
_CHUNK_SCHEMA = {
    "chunk_id": pl.String,
    "article_id": pl.String,
    "section_idx": pl.Int64,
    "chunk_idx": pl.Int64,
    "heading": pl.String,
    "text": pl.String,
    "char_start": pl.Int64,
    "char_end": pl.Int64,
    "n_tokens": pl.Int64,
}
