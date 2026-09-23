"""Structure-aware, deterministic chunking (spec §6.3).

Canonical article text = sections joined by a blank line; each section is its heading line
followed by one line per unit. Every chunk's text is exactly
`article_text(article)[char_start:char_end]`, so offsets are verifiable.

Rules:
1. One chunk per section.
2. A section below `min_tokens` is merged forward with the next section(s) while the group
   stays within `max_tokens`; a tiny final section (< TINY_TOKENS) merges backward instead.
3. A group above `max_tokens` is split at unit (paragraph/step) boundaries, repeating
   `overlap_units` units between consecutive windows.
"""

import re
from dataclasses import dataclass

from fixgraph.core.models import Article, Chunk

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
TINY_TOKENS = 60


def count_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (words + punctuation); close to BPE for English."""
    return len(_TOKEN_RE.findall(text))


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    tokens: int


@dataclass(frozen=True)
class _SectionLayout:
    idx: int
    heading: str
    heading_span: _Span
    unit_spans: list[_Span]

    @property
    def start(self) -> int:
        return self.heading_span.start

    @property
    def end(self) -> int:
        return self.unit_spans[-1].end if self.unit_spans else self.heading_span.end

    @property
    def tokens(self) -> int:
        return self.heading_span.tokens + sum(u.tokens for u in self.unit_spans)


def _layout(article: Article) -> tuple[str, list[_SectionLayout]]:
    parts: list[str] = []
    layouts: list[_SectionLayout] = []
    pos = 0

    def emit(text: str) -> _Span:
        nonlocal pos
        span = _Span(pos, pos + len(text), count_tokens(text))
        parts.append(text)
        pos += len(text)
        return span

    for i, section in enumerate(article.sections):
        if i > 0:
            emit("\n\n")
        heading_span = emit(section.heading)
        units: list[_Span] = []
        for unit in section.units:
            emit("\n")
            units.append(emit(unit.text))
        layouts.append(_SectionLayout(i, section.heading, heading_span, units))
    return "".join(parts), layouts


def article_text(article: Article) -> str:
    return _layout(article)[0]


def _group_sections(
    layouts: list[_SectionLayout], min_tokens: int, max_tokens: int
) -> list[list[_SectionLayout]]:
    groups: list[list[_SectionLayout]] = []
    for sec in layouts:
        if groups:
            current = groups[-1]
            current_tokens = sum(s.tokens for s in current)
            if current_tokens < min_tokens and current_tokens + sec.tokens <= max_tokens:
                current.append(sec)
                continue
        groups.append([sec])
    # A tiny final group (nothing left to merge forward into) joins the previous group.
    if len(groups) > 1:
        last_tokens = sum(s.tokens for s in groups[-1])
        prev_tokens = sum(s.tokens for s in groups[-2])
        if last_tokens < TINY_TOKENS and prev_tokens + last_tokens <= max_tokens:
            groups[-2].extend(groups.pop())
    return groups


def _windows(
    units: list[_Span], max_tokens: int, overlap_units: int, first_extra: int = 0
) -> list[tuple[int, int]]:
    """Split unit indices into [lo, hi) windows of <= max_tokens (a single oversized unit
    becomes its own window). `first_extra` tokens (the heading) count against window one."""
    windows: list[tuple[int, int]] = []
    lo = 0
    while lo < len(units):
        hi = lo
        tokens = first_extra if not windows else 0
        while hi < len(units) and (hi == lo or tokens + units[hi].tokens <= max_tokens):
            tokens += units[hi].tokens
            hi += 1
        windows.append((lo, hi))
        if hi >= len(units):
            break
        lo = max(hi - overlap_units, lo + 1)
    return windows


def chunk_article(
    article: Article, min_tokens: int = 200, max_tokens: int = 500, overlap_units: int = 1
) -> list[Chunk]:
    text, layouts = _layout(article)
    chunks: list[Chunk] = []
    for group in _group_sections(layouts, min_tokens, max_tokens):
        first = group[0]
        group_tokens = sum(s.tokens for s in group)
        if group_tokens <= max_tokens:
            spans = [(group[0].start, group[-1].end)]
        else:
            # Oversized groups are always single sections (merging never exceeds max_tokens).
            sec = first
            units = sec.unit_spans
            spans = []
            windows = _windows(units, max_tokens, overlap_units, sec.heading_span.tokens)
            for n, (lo, hi) in enumerate(windows):
                start = sec.start if n == 0 else units[lo].start
                spans.append((start, units[hi - 1].end))
        for chunk_idx, (start, end) in enumerate(spans):
            body = text[start:end]
            chunks.append(
                Chunk(
                    chunk_id=f"{article.article_id}:{first.idx}:{chunk_idx}",
                    article_id=article.article_id,
                    section_idx=first.idx,
                    chunk_idx=chunk_idx,
                    heading=first.heading,
                    text=body,
                    char_start=start,
                    char_end=end,
                    n_tokens=count_tokens(body),
                )
            )
    return chunks
