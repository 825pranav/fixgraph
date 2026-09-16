"""Hyperlinks between support articles (DECISIONS.md D34).

Apple articles point to each other inside their instructions ("If your computer doesn't
recognize your device, learn how to use recovery mode"). These links are authored by Apple, not
extracted by a model, so they are trustworthy bridges between articles. For every link from a
content block (paragraph, list item, note: the same blocks ingest.parse keeps) to another corpus
article we record the anchor text, the sentence around it and the chunk that sentence is in.

Used by: `fixgraph ingest links` (ingest/cli.py); retrieval.graph (document link layer) and
bench.bridge (bridge questions) read data/corpus/links.parquet.
Uses: ingest.parse (clean, block classes), core.models.Chunk.
"""

import re
from collections.abc import Iterable
from pathlib import Path

import polars as pl
from pydantic import BaseModel
from selectolax.parser import HTMLParser

from fixgraph.core.models import Chunk
from fixgraph.ingest.parse import clean

_ARTICLE_HREF = re.compile(r"(?:support\.apple\.com)?/(?:[a-z]{2}-[a-z]{2}/)?(\d{5,7})(?:$|[/?#])")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Content blocks, as in ingest.parse._walk.
_BLOCKS = "#content p.gb-paragraph, #content ul.gb-list li, #content ol.gb-list li, " + (
    "#content div.gb-note, #content div.gb-callout"
)

LINK_SCHEMA = {
    "src_article": pl.String,
    "dst_article": pl.String,
    "anchor": pl.String,
    "sentence": pl.String,
    "src_chunk_id": pl.String,
}


class Link(BaseModel):
    src_article: str
    dst_article: str
    anchor: str
    sentence: str  # the sentence of the content block that contains the anchor
    src_chunk_id: str = ""  # chunk of src_article containing the sentence ("" if not chunked)


def _sentence_with(block_text: str, anchor: str) -> str:
    for sentence in _SENTENCE_END.split(block_text):
        if anchor and anchor in sentence:
            return sentence.strip()
    return block_text.strip()


def extract_links(html: str, article_id: str, corpus_ids: set[str]) -> list[Link]:
    tree = HTMLParser(html)
    out: list[Link] = []
    seen: set[tuple[str, str]] = set()
    for block in tree.css(_BLOCKS):
        text = clean(block.text(separator=" "))
        for a in block.css("a"):
            m = _ARTICLE_HREF.search(a.attributes.get("href") or "")
            if not m or m.group(1) == article_id or m.group(1) not in corpus_ids:
                continue
            anchor = clean(a.text(separator=" "))
            sentence = _sentence_with(text, anchor)
            if not anchor or (m.group(1), sentence) in seen:
                continue
            seen.add((m.group(1), sentence))
            out.append(
                Link(
                    src_article=article_id, dst_article=m.group(1), anchor=anchor, sentence=sentence
                )
            )
    return out


def attach_chunks(links: list[Link], chunks: Iterable[Chunk]) -> list[Link]:
    """Fill `src_chunk_id` with the first chunk of the source article containing the sentence."""
    by_article: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_article.setdefault(c.article_id, []).append(c)
    out = []
    for link in links:
        cid = next(
            (
                c.chunk_id
                for c in sorted(by_article.get(link.src_article, []), key=lambda c: c.chunk_id)
                if link.sentence in c.text
            ),
            "",
        )
        out.append(link.model_copy(update={"src_chunk_id": cid}))
    return out


def write_links(links: list[Link], path: Path) -> None:
    rows = [x.model_dump() for x in links]
    pl.DataFrame(rows, schema=LINK_SCHEMA).write_parquet(path)


def read_links(path: Path) -> list[Link]:
    if not path.exists():
        return []
    return [Link(**r) for r in pl.read_parquet(path).iter_rows(named=True)]
