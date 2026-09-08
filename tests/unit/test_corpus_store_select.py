from pathlib import Path

from fixgraph.core.models import Article, ArticleSection, TextUnit
from fixgraph.core.ontology import load_ontology
from fixgraph.ingest.chunk import chunk_article
from fixgraph.ingest.select import relevance, select_corpus
from fixgraph.ingest.store import read_articles, read_chunks, write_articles, write_chunks


def _art(aid: str, title: str, tags: list[str], steps: int = 5) -> Article:
    return Article(
        article_id=aid,
        url=f"https://support.apple.com/en-us/{aid}",
        title=title,
        product_tags=tags,
        os_versions_mentioned=["iOS 26"],
        sections=[
            ArticleSection(
                heading=title,
                level=1,
                units=[TextUnit(kind="step", text=f"Step {i} – café") for i in range(steps)],
            )
        ],
    )


def test_articles_and_chunks_roundtrip(tmp_path: Path) -> None:
    arts = [_art("1", "If your iPhone won't charge", ["iPhone"])]
    write_articles(arts, tmp_path / "a.parquet")
    assert read_articles(tmp_path / "a.parquet") == arts
    chunks = chunk_article(arts[0])
    write_chunks(chunks, tmp_path / "c.parquet")
    assert read_chunks(tmp_path / "c.parquet") == chunks


def test_selection_prefers_troubleshooting_and_targets() -> None:
    onto = load_ontology()
    trouble = _art("3", "If your AirPods won't connect", ["AirPods"])
    howto = _art("2", "Use Maps on your iPhone", ["iPhone"])
    other = _art("1", "Set up your HomePod", ["HomePod"])
    legal = _art("4", "Apple security releases for iPhone", ["iPhone"])
    assert relevance(trouble, onto) > relevance(howto, onto) > 0
    assert relevance(other, onto) == 0 and relevance(legal, onto) == 0
    chosen = select_corpus([other, howto, trouble, legal], onto, target=1)
    assert [a.article_id for a in chosen] == ["3"]
    assert [a.article_id for a in select_corpus([trouble, howto], onto, 10)] == ["2", "3"]
