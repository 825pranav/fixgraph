"""Parquet persistence for articles and chunks."""

import json
from pathlib import Path

import polars as pl

from fixgraph.core.models import Article, ArticleSection, Chunk


def write_articles(articles: list[Article], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([c.model_dump() for c in chunks], schema=_CHUNK_SCHEMA).write_parquet(path)


def read_chunks(path: Path) -> list[Chunk]:
    return [Chunk.model_validate(r) for r in pl.read_parquet(path).iter_rows(named=True)]


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
