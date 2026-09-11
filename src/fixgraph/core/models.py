"""Corpus data models shared across stages (disk boundary: parquet).

Article (sections of text units) and Chunk. Produced by ingest.parse / ingest.chunk, persisted by
ingest.store, and consumed by kg (extraction, run_extract, build, annotate) and retrieval.index.
No fixgraph imports.
"""

from typing import Literal

from pydantic import BaseModel, Field

UnitKind = Literal["paragraph", "step", "note"]


class TextUnit(BaseModel):
    """A paragraph, numbered step or note, in reading order."""

    kind: UnitKind
    text: str


class ArticleSection(BaseModel):
    heading: str
    level: int = Field(ge=1, le=6)
    units: list[TextUnit] = Field(default_factory=lambda: list[TextUnit]())


class Article(BaseModel):
    article_id: str
    url: str
    title: str
    summary: str = ""
    product_tags: list[str] = Field(default_factory=lambda: list[str]())
    os_versions_mentioned: list[str] = Field(default_factory=lambda: list[str]())
    last_updated: str | None = None
    sections: list[ArticleSection] = Field(default_factory=lambda: list[ArticleSection]())


class Chunk(BaseModel):
    chunk_id: str  # {article_id}:{section_idx}:{chunk_idx}
    article_id: str
    section_idx: int
    chunk_idx: int
    heading: str
    text: str
    char_start: int  # offsets into the article's canonical text (ingest.chunk.article_text)
    char_end: int
    n_tokens: int
