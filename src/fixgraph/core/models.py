"""Corpus data models shared across stages (disk boundary: parquet).

Article (sections of text units) and Chunk. Produced by ingest.parse / ingest.chunk, persisted by
ingest.store, and consumed by kg (extraction, run_extract, build, annotate) and retrieval.index.
No fixgraph imports.
"""

# Imports: pydantic models give validation and easy conversion to and from parquet rows.
from typing import Literal

from pydantic import BaseModel, Field

# The three kinds of text unit the parser emits inside a section.
UnitKind = Literal["paragraph", "step", "note"]


# Smallest piece of parsed text; chunking later groups these units into chunks.
class TextUnit(BaseModel):
    """A paragraph, numbered step or note, in reading order."""

    kind: UnitKind
    text: str


# One headed section of an article, holding its units in reading order.
class ArticleSection(BaseModel):
    heading: str
    level: int = Field(ge=1, le=6)
    units: list[TextUnit] = Field(default_factory=lambda: list[TextUnit]())


# A parsed help article: parse.py builds it, select.py filters it, store.py saves it to parquet.
class Article(BaseModel):
    article_id: str
    url: str
    title: str
    summary: str = ""
    product_tags: list[str] = Field(default_factory=lambda: list[str]())
    os_versions_mentioned: list[str] = Field(default_factory=lambda: list[str]())
    last_updated: str | None = None
    sections: list[ArticleSection] = Field(default_factory=lambda: list[ArticleSection]())


# A retrievable passage: chunk.py builds it from an Article; the index and KG extraction read it.
# The id and character offsets are what citations and provenance point back to.
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
