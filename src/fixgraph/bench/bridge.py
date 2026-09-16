"""Bridge benchmark: true multi-hop questions from Apple's own links (DECISIONS.md D35).

A link from article A to article C sits in a conditional sentence of A ("If your computer
doesn't recognize your device, learn how to use recovery mode"). A *bridge* question describes
A's situation, including the condition, and its answer is only in C; the linked concept (the
anchor, e.g. "recovery mode") is never named. Each bridge question gets a matched *direct*
question that asks about C's topic outright, with the same gold answer and gold C chunk, so the
cost of the hop is measured within pairs.

The rule is fixed before any system runs (D35) and never looks at retrieval results:
1. Candidates: links whose sentence contains if / when / unless, located in a corpus chunk.
2. One candidate per (source, target) article pair; at most MAX_PER_SOURCE per source article,
   chosen by a seeded shuffle.
3. The judge-class model writes the pair from A's chunk and C's first chunks; the answer must
   quote C verbatim (checked in code; the gold C chunk is the one containing the quote).
4. Mechanical leak checks on the bridge question: no content word of the anchor (except words
   that are already in A's title or are generic, e.g. "learn", "help"), and answer overlap
   (bench.screen.leak_overlap) below LEAK_THRESHOLD.
A pair survives only if both questions pass; review then keeps or drops pairs as a unit.

Used by: `fixgraph bench bridge-generate` (bench/cli.py).
Uses: ingest.links (Link), bench.screen (content words, leak overlap), llm.structured.
"""

import random
import re
from collections import defaultdict
from typing import Any

from pydantic import BaseModel

from fixgraph.bench.schema import Question
from fixgraph.bench.screen import _TOPIC_WORDS as TOPIC_WORDS  # pyright: ignore[reportPrivateUsage]
from fixgraph.bench.screen import (
    _content_words as content_words,  # pyright: ignore[reportPrivateUsage]
)
from fixgraph.bench.screen import _stem as stem  # pyright: ignore[reportPrivateUsage]
from fixgraph.bench.screen import leak_overlap
from fixgraph.ingest.links import Link
from fixgraph.kg.validation import partial_ratio
from fixgraph.llm.base import ChatMessage, LLMClient, LLMRequest
from fixgraph.llm.structured import StructuredOutputError, complete_structured

MAX_PER_SOURCE = 2
_TOPIC = {stem(w) for w in TOPIC_WORDS}  # device / OS / app names are the topic, not a bridge
LEAK_THRESHOLD = 0.6
QUOTE_THRESHOLD = 0.85
MAX_TARGET_CHUNKS = 3
MAX_CHARS = 1500
_CONDITIONAL = re.compile(r"\b(if|when|unless)\b", re.IGNORECASE)
# Link wording that names no concept ("learn what to do", "get help").
_GENERIC = content_words(
    "learn how what more about get help see check make sure use using find set turn try steps "
    "information article instructions support contact but all any some each every other also "
    "only just still after before while when then than into over under two one say says "
    "can cannot need needs want there their they them these those which who"
)


class BridgeCandidate(BaseModel):
    link: Link
    source_title: str
    target_title: str
    target_chunk_ids: list[str]


def select_candidates(
    links: list[Link],
    titles: dict[str, str],
    chunks_by_article: dict[str, list[str]],
    seed: int = 13,
) -> list[BridgeCandidate]:
    """Rules 1-2. Deterministic for a given seed."""
    by_pair: dict[tuple[str, str], Link] = {}
    for link in sorted(links, key=lambda x: (x.src_article, x.dst_article, x.src_chunk_id)):
        if not link.src_chunk_id or not _CONDITIONAL.search(link.sentence):
            continue
        if not chunks_by_article.get(link.dst_article):
            continue
        by_pair.setdefault((link.src_article, link.dst_article), link)
    by_source: dict[str, list[Link]] = defaultdict(list)
    for (src, _), link in sorted(by_pair.items()):
        by_source[src].append(link)
    rng = random.Random(seed)
    out: list[BridgeCandidate] = []
    for src in sorted(by_source):
        pool = by_source[src]
        rng.shuffle(pool)
        for link in pool[:MAX_PER_SOURCE]:
            out.append(
                BridgeCandidate(
                    link=link,
                    source_title=titles.get(link.src_article, ""),
                    target_title=titles.get(link.dst_article, ""),
                    target_chunk_ids=sorted(chunks_by_article[link.dst_article])[
                        :MAX_TARGET_CHUNKS
                    ],
                )
            )
    return out


class BridgePair(BaseModel):
    bridge_question: str
    direct_question: str
    answer: str
    key_facts: list[str]
    answer_quote: str


_PROMPT = """You write two customer questions for Apple Support that share one answer.

Source article A: "{source_title}"
Passage from A (the customer's situation):
{source_text}

In A, this sentence sends the reader to article C: "{sentence}"
Article C: "{target_title}"
Text of C:
{target_text}

Write:
- bridge_question: a customer in A's situation, facing the condition in the sentence above, asks
  what to do. Describe what they were doing and what they see. Do NOT name or hint at "{anchor}"
  or at the answer; the customer does not know article C exists.
- direct_question: a customer asks directly about C's topic (they may name it).
- answer: 1-2 sentences, using ONLY article C, that answer both questions.
- key_facts: 1-3 short phrases from C that a correct answer must contain.
- answer_quote: one sentence copied exactly from C that supports the answer.
Use natural customer wording. Do not mention articles, links or sources."""


def generation_request(cand: BridgeCandidate, chunk_text: dict[str, str], model: str) -> LLMRequest:
    target = "\n\n".join(chunk_text.get(c, "")[:MAX_CHARS] for c in cand.target_chunk_ids)
    prompt = _PROMPT.format(
        source_title=cand.source_title,
        source_text=chunk_text.get(cand.link.src_chunk_id, "")[:MAX_CHARS],
        sentence=cand.link.sentence,
        target_title=cand.target_title,
        target_text=target,
        anchor=cand.link.anchor,
    )
    return LLMRequest(
        model=model,
        messages=[ChatMessage(role="user", content=prompt)],
        num_ctx=8192,
        max_tokens=400,
    )


class BridgeResult(BaseModel):
    candidate: BridgeCandidate
    ok: bool
    reason: str
    questions: list[Question] = []


def anchor_leak(question: str, anchor: str, source_title: str) -> set[str]:
    """Anchor content words in the question, ignoring generic link wording and words the
    customer already has from A's title."""
    allowed = _GENERIC | _TOPIC | content_words(source_title)
    return (content_words(anchor) - allowed) & content_words(question)


def check_pair(
    pair: BridgePair,
    cand: BridgeCandidate,
    chunk_text: dict[str, str],
    pid: str,
    source: str = "bridge",
) -> BridgeResult:
    """Rules 3-4, all mechanical."""
    quote_scores = {
        c: partial_ratio(pair.answer_quote, chunk_text.get(c, "")) for c in cand.target_chunk_ids
    }
    gold_c, score = max(quote_scores.items(), key=lambda x: x[1])
    if score < QUOTE_THRESHOLD:
        return BridgeResult(candidate=cand, ok=False, reason="answer quote not found in C")
    leaked = anchor_leak(pair.bridge_question, cand.link.anchor, cand.source_title)
    if leaked:
        return BridgeResult(
            candidate=cand, ok=False, reason=f"bridge question names the link: {sorted(leaked)}"
        )
    if leak_overlap(pair.bridge_question, [pair.answer]) >= LEAK_THRESHOLD:
        return BridgeResult(candidate=cand, ok=False, reason="bridge question leaks the answer")
    facts = pair.key_facts[:3]
    common = {"gold_answer": pair.answer.strip(), "key_facts": facts, "source": source,
              "split": "test"}  # fmt: skip
    bridge = Question(
        qid=f"{pid}a",
        question=pair.bridge_question.strip(),
        qtype="bridge",
        gold_chunk_ids=sorted({cand.link.src_chunk_id, gold_c}),
        **common,
    )
    direct = Question(
        qid=f"{pid}d",
        question=pair.direct_question.strip(),
        qtype="bridge_direct",
        gold_chunk_ids=[gold_c],
        **common,
    )
    return BridgeResult(candidate=cand, ok=True, reason="ok", questions=[bridge, direct])


def generate_pair(
    client: LLMClient,
    cand: BridgeCandidate,
    chunk_text: dict[str, str],
    model: str,
    pid: str,
) -> BridgeResult:
    try:
        pair = complete_structured(client, generation_request(cand, chunk_text, model), BridgePair)
    except (StructuredOutputError, RuntimeError) as exc:
        return BridgeResult(candidate=cand, ok=False, reason=f"generation failed: {exc}"[:200])
    return check_pair(pair, cand, chunk_text, pid)


# ---------------------------------------------------------------------------
# Analysis (D35): answer-chunk hit@8, hop cost, crossover vs S1
# ---------------------------------------------------------------------------


def _ci(values: list[float]) -> dict[str, float]:
    from fixgraph.bench.stats import bootstrap_ci

    c = bootstrap_ci(values)
    return {"mean": round(c.mean, 3), "lo": round(c.lo, 3), "hi": round(c.hi, 3), "n": c.n}


def _tests(
    per: dict[str, dict[str, float]], baseline: str, keys: list[str]
) -> list[dict[str, object]]:
    """Paired permutation test of every system vs `baseline` over `keys`, Holm-corrected."""
    from fixgraph.bench.stats import holm, paired_effect_size, paired_permutation_test

    rows: list[dict[str, object]] = []
    pvals: list[float] = []
    for s in sorted(per):
        if s == baseline:
            continue
        shared = [k for k in keys if k in per[s] and k in per[baseline]]
        if len(shared) < 3:
            continue
        a = [per[s][k] for k in shared]
        b = [per[baseline][k] for k in shared]
        p = paired_permutation_test(a, b)
        pvals.append(p)
        diff = sum(a) / len(a) - sum(b) / len(b)
        rows.append(
            {
                "system": s,
                "n": len(shared),
                "diff": round(diff, 3),
                "p": round(p, 4),
                "d_z": round(paired_effect_size(a, b), 2),
            }
        )
    for row, adj in zip(rows, holm(pvals), strict=True):
        row["p_holm"] = round(adj, 4)
    return rows


def bridge_report(
    questions: list[Question],
    ranked: dict[tuple[str, str], list[str]],
    correctness: dict[tuple[str, str], float],
    k: int = 8,
) -> dict[str, Any]:
    """`ranked[(qid, system)]` = retrieved chunk ids; `correctness[(qid, system)]` = judge
    score. Only complete pairs (both `a` and `d` present) are analysed."""
    by_qid = {q.qid: q for q in questions}
    pairs = sorted(
        p for p in {q.qid[:-1] for q in questions} if f"{p}a" in by_qid and f"{p}d" in by_qid
    )
    systems = sorted({s for _, s in ranked} | {s for _, s in correctness})
    gold_c = {p: by_qid[f"{p}d"].gold_chunk_ids[0] for p in pairs}
    hit: dict[str, dict[str, float]] = {s: {} for s in systems}
    corr: dict[str, dict[str, float]] = {s: {} for s in systems}
    for p in pairs:
        for side in ("a", "d"):
            qid = f"{p}{side}"
            for s in systems:
                if (qid, s) in ranked and s != "S0":
                    hit[s][qid] = float(gold_c[p] in ranked[(qid, s)][:k])
                if (qid, s) in correctness:
                    corr[s][qid] = correctness[(qid, s)]
    hit = {s: v for s, v in hit.items() if v}
    corr = {s: v for s, v in corr.items() if v}  # empty for retrieval-only runs

    def side_keys(side: str) -> list[str]:
        return [f"{p}{side}" for p in pairs]

    def by_side(per: dict[str, dict[str, float]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for side, name in (("a", "bridge"), ("d", "direct")):
            keys = side_keys(side)
            out[name] = {
                "mean": {s: _ci([v[q] for q in keys if q in v]) for s, v in sorted(per.items())},
                "tests_vs_S1": _tests(per, "S1", keys),
            }
        # hop cost per pair = direct - bridge; crossover = hop cost of a system vs S1's
        cost = {
            s: {p: v[f"{p}d"] - v[f"{p}a"] for p in pairs if f"{p}d" in v and f"{p}a" in v}
            for s, v in per.items()
        }
        out["hop_cost"] = {
            "mean": {s: _ci(list(c.values())) for s, c in sorted(cost.items())},
            "crossover_tests_vs_S1": _tests(cost, "S1", pairs),
        }
        return out

    return {
        "n_pairs": len(pairs),
        "k": k,
        f"answer_chunk_hit@{k}": by_side(hit),
        "correctness": by_side(corr),
        "note": "hop cost = direct - bridge per pair (positive = the hop hurts); tests are "
        "paired permutation tests vs S1, Holm-corrected across systems (DECISIONS.md D35).",
    }
