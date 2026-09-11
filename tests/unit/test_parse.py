"""HTML parsing of a fixture article: metadata, sections/units, boilerplate removal.

Covers ingest/parse.py (fixtures in tests/fixtures/html).
"""

from pathlib import Path

import pytest

from fixgraph.core.models import Article
from fixgraph.core.ontology import Ontology, load_ontology
from fixgraph.ingest.parse import clean, parse_file

FIXTURE = Path(__file__).parents[1] / "fixtures" / "html" / "900001.html"


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    return load_ontology()


@pytest.fixture(scope="module")
def article(ontology: Ontology) -> Article:
    a = parse_file(FIXTURE, ontology)
    assert a is not None
    return a


def test_metadata(article: Article) -> None:
    assert article.article_id == "900001"
    assert article.url == "https://support.apple.com/en-us/900001"
    assert article.title == "If your Apple Watch won't pair with your iPhone"
    assert article.summary.startswith("If you can't pair")
    assert article.last_updated == "2026-09-22"
    assert article.product_tags == ["Apple Watch", "Mac", "iPhone"]
    assert article.os_versions_mentioned == ["iOS 26.1", "macOS 26", "watchOS 26"]


def test_sections_and_units(article: Article) -> None:
    headings = [(s.level, s.heading) for s in article.sections]
    assert headings == [
        (1, "If your Apple Watch won't pair with your iPhone"),
        (2, "Check Bluetooth"),
        (3, "Unpair and pair again › On your iPhone"),
    ]
    check = article.sections[1]
    assert [u.kind for u in check.units] == ["paragraph", "step", "step", "note"]
    assert check.units[0].text == "Make sure that Bluetooth is turned on on your iPhone."
    assert check.units[2].text == "Tap Bluetooth, then turn it on."


def test_toc_nav_and_boilerplate_dropped(article: Article) -> None:
    all_text = " ".join(u.text for s in article.sections for u in s.units)
    assert "Unpair and pair again" not in all_text  # TOC entry
    assert "Store" not in all_text  # global nav outside #content
    assert "not manufactured by Apple" not in all_text
    assert "macOS Tahoe" in all_text


def test_empty_heading_folded_into_child(article: Article) -> None:
    # "Unpair and pair again" (h2) has no units of its own before the h3.
    assert all(s.units for s in article.sections)
    assert article.sections[2].units[-1].text == "This also works on a Mac with macOS Tahoe."


def test_clean() -> None:
    assert clean("  a  b  .\n c ) ") == "a b. c)"
