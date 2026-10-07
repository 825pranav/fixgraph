"""Parse cached support-article HTML into structured `Article`s (spec §6.1).

Used by: `fixgraph ingest parse` (ingest/cli.py) on the HTML cached by ingest.scrape.
Uses: core.models (Article, sections, text units), core.ontology (product tagging).
"""

# Imports: selectolax is a fast HTML parser; core models are the output shape.
import logging
import re
from datetime import datetime
from pathlib import Path

from selectolax.parser import HTMLParser, Node

from fixgraph.core.models import Article, ArticleSection, TextUnit, UnitKind
from fixgraph.core.ontology import Ontology

logger = logging.getLogger(__name__)

# Regexes and tables used to tidy text and map h2/h3/h4 tags to section levels.
_WS_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?)\]])")
_HEADING_LEVEL = {"h2": 2, "h3": 3, "h4": 4}
# Site-wide boilerplate that carries no troubleshooting content.
_BOILERPLATE_PREFIXES = (
    "Information about products not manufactured by Apple",
    "Apple makes no representations regarding third-party website",
)


# Collapse whitespace and remove spaces before punctuation, so stored text is clean and stable.
def clean(text: str) -> str:
    text = _WS_RE.sub(" ", text).strip()  # \s also matches NBSP
    return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)


# Return the CSS classes of an HTML node as a set.
def _classes(node: Node) -> set[str]:
    return set((node.attributes.get("class") or "").split())


# Detect the in-page table of contents list so it is not stored as content.
def _is_toc(list_node: Node) -> bool:
    """In-page table of contents: every link points to an #anchor."""
    links = list_node.css("a")
    return bool(links) and all((a.attributes.get("href") or "").startswith("#") for a in links)


# Turn a date like "March 3, 2024" into ISO format, or None if it does not parse.
def _parse_date(raw: str) -> str | None:
    try:
        return datetime.strptime(clean(raw), "%B %d, %Y").date().isoformat()
    except ValueError:
        return None


# Collects sections and text units while the HTML tree is walked in reading order.
class _Builder:
    # Start with a level-1 section named after the article title to hold intro text.
    def __init__(self, title: str) -> None:
        self.sections: list[ArticleSection] = [ArticleSection(heading=title, level=1)]

    # A new heading starts a new section; later units go into it.
    def heading(self, text: str, level: int) -> None:
        self.sections.append(ArticleSection(heading=text, level=level))

    # Add a cleaned text unit to the current section, skipping empty text and site boilerplate.
    def add(self, kind: UnitKind, text: str) -> None:
        text = clean(text)
        if text and not text.startswith(_BOILERPLATE_PREFIXES):
            self.sections[-1].units.append(TextUnit(kind=kind, text=text))

    # Return the final section list once the walk is finished.
    def result(self) -> list[ArticleSection]:
        """Drop empty sections; an empty parent heading is folded into its first child's."""
        out: list[ArticleSection] = []
        pending: list[ArticleSection] = []
        # Empty headings wait in `pending`; the next non-empty section gets their names as a prefix.
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


# Recursively walk the HTML and send headings, paragraphs, list items and notes to the builder.
def _walk(node: Node, b: _Builder) -> None:
    for child in node.iter(include_text=False):
        tag = child.tag
        cls = _classes(child)
        # Map Apple's CSS classes to our structure: headers, paragraphs, lists, notes.
        # Anything else is a wrapper, so recurse into it.
        if tag in _HEADING_LEVEL and "gb-header" in cls:
            b.heading(clean(child.text()), _HEADING_LEVEL[tag])
        elif tag == "h1":
            continue
        elif tag == "p" and "gb-subheader" in cls:
            continue  # captured separately as the summary
        elif tag == "p" and "gb-paragraph" in cls:
            b.add("paragraph", child.text(separator=" "))
        # Ordered lists become numbered steps, bullet lists become paragraphs; TOC lists skipped.
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


# Turn one article's HTML into an Article, or None if it has no title or no content.
def parse_article(html: str, article_id: str, url: str, ontology: Ontology) -> Article | None:
    # Find the main content area, the title and the summary line.
    tree = HTMLParser(html)
    main = tree.css_first("#content") or tree.body
    h1 = tree.css_first("h1")
    if main is None or h1 is None:
        return None
    title = clean(h1.text())
    sub = tree.css_first("p.gb-subheader")
    summary = clean(sub.text(separator=" ")) if sub else ""

    # Walk the content into sections, then put the summary at the start of the first section.
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

    # Pick up the published date from the <time> tag if there is one.
    last_updated = None
    for t in tree.css("time"):
        if "datePublished" in (t.html or ""):
            last_updated = _parse_date(t.text())
            break

    # Tag products and OS versions from the full text using the ontology, then build the Article.
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


# Read one cached HTML file; the file name is the article id. Called by `ingest parse`.
def parse_file(path: Path, ontology: Ontology) -> Article | None:
    article_id = path.stem
    html = path.read_text(encoding="utf-8")
    return parse_article(
        html, article_id, f"https://support.apple.com/en-us/{article_id}", ontology
    )
