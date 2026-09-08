"""Choose the research corpus (~1,000 articles) from everything scraped.

Keeps articles about the target product families that look like troubleshooting or how-to
content, ranked by a transparent keyword score. Deterministic (ties broken by article id).
"""

import re

from fixgraph.core.models import Article
from fixgraph.core.ontology import Ontology

_TROUBLESHOOT_RE = re.compile(
    r"\b(if|can't|cannot|won't|isn't|aren't|doesn't|don't|not working|error|alert|fix|"
    r"troubleshoot|reset|restart|restore|update|pair|unpair|connect|charge|charging|battery|"
    r"sync|back up|backup|stuck|frozen|slow|drain|lost|forgot|locked|disabled|unresponsive)\b",
    re.IGNORECASE,
)
_EXCLUDE_TITLE_RE = re.compile(
    r"security (releases|content)|about the security|legal|warranty|service program|"
    r"exchange program|trade in|apple store|developer|enterprise|business|education|"
    r"accessibility|logic pro|final cut|garageband|compressor|motion \d|mainstage|alchemy|"
    r"logic remote|pro apps?\b|xsan|exchange server|\bmdm\b|apple configurator",
    re.IGNORECASE,
)


def relevance(article: Article, ontology: Ontology) -> float:
    targets = set(ontology.target_families)
    title_families = set(ontology.families_in(article.title)) & targets
    body_families = set(article.product_tags) & targets
    if not body_families or _EXCLUDE_TITLE_RE.search(article.title):
        return 0.0
    n_units = sum(len(s.units) for s in article.sections)
    score = 0.0
    score += 2.0 if title_families else 0.5
    score += 2.0 if _TROUBLESHOOT_RE.search(article.title) else 0.0
    score += min(len(_TROUBLESHOOT_RE.findall(article.summary)), 3) * 0.3
    score += 1.0 if 4 <= n_units <= 200 else 0.0
    score += 0.5 if any(u.kind == "step" for s in article.sections for u in s.units) else 0.0
    return score


def select_corpus(articles: list[Article], ontology: Ontology, target: int) -> list[Article]:
    scored = [(relevance(a, ontology), a) for a in articles]
    kept = [(s, a) for s, a in scored if s > 0]
    kept.sort(key=lambda sa: (-sa[0], int(sa[1].article_id)))
    chosen = [a for _, a in kept[:target]]
    return sorted(chosen, key=lambda a: int(a.article_id))


def subset_by_chunk_budget(
    articles: list[Article],
    chunk_counts: dict[str, int],
    ontology: Ontology,
    max_chunks: int,
    must_keep: set[str] | None = None,
) -> list[Article]:
    """Whole articles under a chunk budget: `must_keep` first (e.g. gold-set articles), then by
    relevance (ties by id). Articles never split, so every kept article is fully covered."""
    must = must_keep or set()
    order = sorted(
        articles,
        key=lambda a: (a.article_id not in must, -relevance(a, ontology), int(a.article_id)),
    )
    kept: list[Article] = []
    used = 0
    for a in order:
        n = chunk_counts.get(a.article_id, 0)
        if a.article_id in must or used + n <= max_chunks:
            kept.append(a)
            used += n
    return sorted(kept, key=lambda a: int(a.article_id))
