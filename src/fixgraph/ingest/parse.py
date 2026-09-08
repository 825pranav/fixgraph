"""Parse cached support-article HTML into structured `Article`s (spec §6.1)."""

import logging
import re
from datetime import datetime
from pathlib import Path

from selectolax.parser import HTMLParser, Node

from fixgraph.core.models import Article, ArticleSection, TextUnit, UnitKind
from fixgraph.core.ontology import Ontology

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?)\]])")
_HEADING_LEVEL = {"h2": 2, "h3": 3, "h4": 4}
# Site-wide boilerplate that carries no troubleshooting content.
_BOILERPLATE_PREFIXES = (
    "Information about products not manufactured by Apple",
    "Apple makes no representations regarding third-party website",
)


def clean(text: str) -> str:
    text = _WS_RE.sub(" ", text).strip()  # \s also matches NBSP
    return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)


def _classes(node: Node) -> set[str]:
    return set((node.attributes.get("class") or "").split())


def _is_toc(list_node: Node) -> bool:
    """In-page table of contents: every link points to an #anchor."""
    links = list_node.css("a")
    return bool(links) and all((a.attributes.get("href") or "").startswith("#") for a in links)


def _parse_date(raw: str) -> str | None:
    try:
        return datetime.strptime(clean(raw), "%B %d, %Y").date().isoformat()
    except ValueError:
        return None


class _Builder:
    def __init__(self, title: str) -> None:
        self.sections: list[ArticleSection] = [ArticleSection(heading=title, level=1)]

    def heading(self, text: str, level: int) -> None:
        self.sections.append(ArticleSection(heading=text, level=level))

    def add(self, kind: UnitKind, text: str) -> None:
        text = clean(text)
        if text and not text.startswith(_BOILERPLATE_PREFIXES):
            self.sections[-1].units.append(TextUnit(kind=kind, text=text))

    def result(self) -> list[ArticleSection]:
        """Drop empty sections; an empty parent heading is folded into its first child's."""
        out: list[ArticleSection] = []
        pending: list[ArticleSection] = []
        for s in self.sections:
            if not s.units:
                pending = [p for p in pending if p.level < s.level] + [s]
                continue
            parents = [p.heading for p in pending if p.level < s.level and p.level > 1]
            pending = []
            if parents:
                s = s.model_copy(update={"heading": " › ".join([*parents, s.heading])})
            out.append(s)
        return out


def _walk(node: Node, b: _Builder) -> None:
    for child in node.iter(include_text=False):
        tag = child.tag
        cls = _classes(child)
        if tag in _HEADING_LEVEL and "gb-header" in cls:
            b.heading(clean(child.text()), _HEADING_LEVEL[tag])
        elif tag == "h1":
            continue
        elif tag == "p" and "gb-subheader" in cls:
            continue  # captured separately as the summary
        elif tag == "p" and "gb-paragraph" in cls:
            b.add("paragraph", child.text(separator=" "))
        elif tag in ("ol", "ul") and "gb-list" in cls:
            if _is_toc(child):
                continue
            kind: UnitKind = "step" if tag == "ol" else "paragraph"
            for li in child.iter(include_text=False):
                if li.tag == "li":
                    b.add(kind, li.text(separator=" "))
        elif tag == "div" and cls & {"gb-note", "gb-callout"}:
            b.add("note", child.text(separator=" "))
        else:
            _walk(child, b)


def parse_article(html: str, article_id: str, url: str, ontology: Ontology) -> Article | None:
    tree = HTMLParser(html)
    main = tree.css_first("#content") or tree.body
    h1 = tree.css_first("h1")
    if main is None or h1 is None:
        return None
    title = clean(h1.text())
    sub = tree.css_first("p.gb-subheader")
    summary = clean(sub.text(separator=" ")) if sub else ""

    builder = _Builder(title)
    _walk(main, builder)
    sections = builder.result()
    if summary and sections and sections[0].level == 1:
        sections[0].units.insert(0, TextUnit(kind="paragraph", text=summary))
    elif summary:
        sections.insert(
            0,
            ArticleSection(
                heading=title, level=1, units=[TextUnit(kind="paragraph", text=summary)]
            ),
        )
    if not sections:
        return None

    last_updated = None
    for t in tree.css("time"):
        if "datePublished" in (t.html or ""):
            last_updated = _parse_date(t.text())
            break

    full_text = " ".join([title] + [u.text for s in sections for u in s.units])
    return Article(
        article_id=article_id,
        url=url,
        title=title,
        summary=summary,
        product_tags=ontology.families_in(full_text),
        os_versions_mentioned=ontology.os_versions_in(full_text),
        last_updated=last_updated,
        sections=sections,
    )


def parse_file(path: Path, ontology: Ontology) -> Article | None:
    article_id = path.stem
    html = path.read_text(encoding="utf-8")
    return parse_article(
        html, article_id, f"https://support.apple.com/en-us/{article_id}", ontology
    )
