from hypothesis import given, settings
from hypothesis import strategies as st

from fixgraph.core.models import Article, ArticleSection, TextUnit
from fixgraph.ingest.chunk import article_text, chunk_article, count_tokens

_word = st.text(alphabet="abcdefghij", min_size=1, max_size=8)
_unit_text = st.lists(_word, min_size=1, max_size=120).map(" ".join)
_unit = st.builds(TextUnit, kind=st.sampled_from(["paragraph", "step", "note"]), text=_unit_text)
_section = st.builds(
    ArticleSection,
    heading=_unit_text.map(lambda t: t[:40]),
    level=st.integers(1, 3),
    units=st.lists(_unit, min_size=1, max_size=15),
)
articles_st = st.builds(
    Article,
    article_id=st.just("123"),
    url=st.just("u"),
    title=st.just("t"),
    sections=st.lists(_section, min_size=1, max_size=8),
)


def _words(n: int) -> str:
    return " ".join(["w"] * n)


def _article(*sections: list[int]) -> Article:
    return Article(
        article_id="42",
        url="u",
        title="t",
        sections=[
            ArticleSection(
                heading=f"H{i}",
                level=2,
                units=[TextUnit(kind="step", text=_words(n)) for n in units],
            )
            for i, units in enumerate(sections)
        ],
    )


@settings(max_examples=150, deadline=None)
@given(articles_st)
def test_offsets_are_exact_substrings(article: Article) -> None:
    text = article_text(article)
    for c in chunk_article(article):
        assert text[c.char_start : c.char_end] == c.text


@settings(max_examples=150, deadline=None)
@given(articles_st)
def test_every_unit_covered_and_ids_unique(article: Article) -> None:
    chunks = chunk_article(article)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    joined = "\n".join(c.text for c in chunks)
    for s in article.sections:
        for u in s.units:
            assert u.text in joined


@settings(max_examples=150, deadline=None)
@given(articles_st)
def test_deterministic(article: Article) -> None:
    assert chunk_article(article) == chunk_article(article.model_copy(deep=True))


@settings(max_examples=150, deadline=None)
@given(articles_st)
def test_size_bound_except_oversized_single_units(article: Article) -> None:
    max_tokens = 500
    for c in chunk_article(article, max_tokens=max_tokens):
        if c.n_tokens > max_tokens:
            # Only allowed when the chunk is a heading plus a single oversized unit.
            assert c.text.count("\n") <= 1


def test_small_sections_merge_forward() -> None:
    chunks = chunk_article(_article([30], [30], [300]), min_tokens=200, max_tokens=500)
    assert [c.chunk_id for c in chunks] == ["42:0:0"]
    assert chunks[0].heading == "H0"


def test_large_section_splits_with_overlap() -> None:
    chunks = chunk_article(_article([200, 200, 200, 200]), min_tokens=200, max_tokens=500)
    assert [c.chunk_id for c in chunks] == ["42:0:0", "42:0:1", "42:0:2"]
    # Consecutive windows share one unit.
    assert chunks[0].text.split("\n")[-1] == chunks[1].text.split("\n")[0]
    assert chunks[0].text.startswith("H0\n")


def test_tiny_last_section_merges_backward() -> None:
    chunks = chunk_article(_article([250], [250], [20]), min_tokens=200, max_tokens=500)
    assert [c.chunk_id for c in chunks] == ["42:0:0", "42:1:0"]
    assert chunks[-1].text.endswith(_words(20))


def test_section_idx_in_ids() -> None:
    chunks = chunk_article(_article([250], [250]), min_tokens=200, max_tokens=500)
    assert [c.chunk_id for c in chunks] == ["42:0:0", "42:1:0"]


def test_count_tokens() -> None:
    assert count_tokens("Tap Settings > Bluetooth, then turn it on.") == 10
