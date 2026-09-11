"""Benchmark generation from KG paths + human verification (spec §11.2).

What it does:
1. `sample_paths` picks schema-valid KG paths per question type (stratified, seeded). Each path
   records which articles its evidence comes from, so multi-article questions are explicit.
2. `generate_question` has the judge-class model write a customer question from the path's
   facts and evidence chunks. Gold answer = the path's terminal node(s); gold chunks = the edges'
   provenance; gold seeds = the path's non-answer nodes.
3. `verify_loop` makes human verification a few keystrokes per question; it shows the automatic
   pre-screen verdict (bench/screen.py) next to the evidence, but only a human sets `verified`.

Question types and why each needs the graph:
- single_hop            symptom -RESOLVED_BY-> fix. One article. Control group.
- multi_constraint      two symptoms from DIFFERENT articles that share a fix node
                        ("I have problem A and problem B, is there one thing to try?"). The answer
                        is only confirmed by reading both articles.
- version_conditional   symptom -> fix -REQUIRES/APPLIES_TO-> OS version; multi-article paths
                        (fix and version requirement in different articles) are taken first.
- cross_device          product -DEPENDS_ON-> product, child EXHIBITS symptom -> fix.
- error_code            code -SIGNALS-> symptom -> fix.

Used by: `fixgraph bench generate` / `fixgraph bench verify` (bench/cli.py).
Uses: kg/store.py (KG tables), llm/structured.py (schema-constrained generation),
bench/schema.py (Question format).
"""

import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl
from pydantic import BaseModel

from fixgraph.bench.schema import QType, Question, article_of
from fixgraph.kg.store import KG
from fixgraph.llm.base import ChatMessage, LLMClient, LLMRequest
from fixgraph.llm.structured import StructuredOutputError, complete_structured

Edge = dict[str, Any]  # one row of the KG edges table

# A fix shared by more symptoms than this is generic ("Restart your device") and makes a
# multi_constraint question guessable without reading anything.
MAX_SHARED_FIX_DEGREE = 3


class PathSample(BaseModel):
    qtype: QType
    facts: list[str]  # "Apple Watch EXHIBITS won't pair", ...
    answer_nodes: list[str]
    seed_nodes: list[str]
    chunk_ids: list[str]
    answer_texts: list[str] = []  # canonical text of answer_nodes, filled by sample_paths

    @property
    def article_ids(self) -> list[str]:
        return sorted({article_of(c) for c in self.chunk_ids})


@dataclass(frozen=True)
class _GraphView:
    """Outgoing extracted edges by source node, plus node text/label lookups."""

    out: dict[str, list[Edge]]
    text: dict[str, str]
    label: dict[str, str]

    def fact(self, e: Edge) -> str:
        return f"{self.text[e['src']]} {e['rel']} {self.text[e['dst']]}"

    def fixes_of(self, symptom: str) -> list[Edge]:
        return [e for e in self.out.get(symptom, []) if e["rel"] == "RESOLVED_BY"]


def _view(kg: KG) -> _GraphView:
    edges = kg.extracted_edges().filter(pl.col("extraction_model") != "ontology")
    out: dict[str, list[Edge]] = defaultdict(list)
    for r in edges.iter_rows(named=True):
        out[r["src"]].append(r)
    text = dict(zip(kg.nodes["node_id"], kg.nodes["canonical_text"], strict=True))
    label = dict(zip(kg.nodes["node_id"], kg.nodes["label"], strict=True))
    return _GraphView(out=out, text=text, label=label)


def _chunks(*edges: Edge) -> list[str]:
    return sorted({c for e in edges for c in e["source_chunk_ids"]})


def _articles(*edges: Edge) -> set[str]:
    return {article_of(c) for e in edges for c in e["source_chunk_ids"]}


def _single_hop(g: _GraphView, symptoms: list[str]) -> list[PathSample]:
    out = []
    for s in symptoms:
        es = g.fixes_of(s)[:3]
        out.append(
            PathSample(
                qtype="single_hop",
                facts=[g.fact(e) for e in es],
                answer_nodes=[e["dst"] for e in es],
                seed_nodes=[s],
                chunk_ids=_chunks(*es),
            )
        )
    return out


def _multi_constraint(g: _GraphView, symptoms: list[str]) -> list[PathSample]:
    """Pairs of symptoms whose only shared evidence is a common, non-generic fix node, with each
    symptom's RESOLVED_BY edge supported by a different article."""
    by_fix: dict[str, list[Edge]] = defaultdict(list)
    for s in symptoms:
        for e in g.fixes_of(s):
            by_fix[e["dst"]].append(e)
    out = []
    for fix in sorted(by_fix):
        es = by_fix[fix]
        if len({e["src"] for e in es}) > MAX_SHARED_FIX_DEGREE:
            continue
        for i, a in enumerate(es):
            for b in es[i + 1 :]:
                if a["src"] == b["src"] or _articles(a) & _articles(b):
                    continue  # same symptom, or both facts readable in one article
                out.append(
                    PathSample(
                        qtype="multi_constraint",
                        facts=[g.fact(a), g.fact(b)],
                        answer_nodes=[fix],
                        seed_nodes=[a["src"], b["src"]],
                        chunk_ids=_chunks(a, b),
                    )
                )
    return out


def _version_conditional(g: _GraphView, symptoms: list[str]) -> list[PathSample]:
    out = []
    for s in symptoms:
        for fe in g.fixes_of(s):
            for ve in g.out.get(fe["dst"], []):
                if (
                    ve["rel"] in ("REQUIRES", "APPLIES_TO")
                    and g.label.get(ve["dst"]) == "OSVersion"
                ):
                    out.append(
                        PathSample(
                            qtype="version_conditional",
                            facts=[g.fact(fe), g.fact(ve)],
                            answer_nodes=[fe["dst"], ve["dst"]],
                            seed_nodes=[fe["src"]],
                            chunk_ids=_chunks(fe, ve),
                        )
                    )
    return out


def _cross_device(g: _GraphView) -> list[PathSample]:
    out = []
    for es in g.out.values():
        deps = [e for e in es if e["rel"] == "DEPENDS_ON"]
        syms = [e for e in es if e["rel"] == "EXHIBITS" and g.fixes_of(e["dst"])]
        for d in deps:
            for se in syms:
                fe = g.fixes_of(se["dst"])[0]
                out.append(
                    PathSample(
                        qtype="cross_device",
                        facts=[g.fact(d), g.fact(se), g.fact(fe)],
                        answer_nodes=[fe["dst"]],
                        seed_nodes=[d["src"], d["dst"], se["dst"]],
                        chunk_ids=_chunks(d, se, fe),
                    )
                )
    return out


def _error_code(g: _GraphView) -> list[PathSample]:
    out = []
    for es in g.out.values():
        for se in es:
            if se["rel"] == "SIGNALS" and g.fixes_of(se["dst"]):
                fe = g.fixes_of(se["dst"])[0]
                out.append(
                    PathSample(
                        qtype="error_code",
                        facts=[g.fact(se), g.fact(fe)],
                        answer_nodes=[fe["dst"]],
                        seed_nodes=[se["src"]],
                        chunk_ids=_chunks(se, fe),
                    )
                )
    return out


def _take(
    rng: random.Random, cands: list[PathSample], n: int, prefer_multi_article: bool
) -> list[PathSample]:
    """Sample n paths; with `prefer_multi_article`, multi-article paths are used up first."""
    if not prefer_multi_article:
        return rng.sample(cands, min(n, len(cands)))
    multi = [c for c in cands if len(c.article_ids) > 1]
    single = [c for c in cands if len(c.article_ids) <= 1]
    picked = rng.sample(multi, min(n, len(multi)))
    return picked + rng.sample(single, min(n - len(picked), len(single)))


def sample_paths(kg: KG, per_type: int | dict[str, int], seed: int = 13) -> list[PathSample]:
    """Stratified, seeded path sample. `per_type` is one count for every type or a
    {qtype: count} map (types left out get 0)."""
    rng = random.Random(seed)
    g = _view(kg)
    symptoms = sorted(n for n, lab in g.label.items() if lab == "Symptom")
    with_fix = [s for s in symptoms if g.fixes_of(s)]
    candidates: dict[str, list[PathSample]] = {
        "single_hop": _single_hop(g, with_fix),
        "multi_constraint": _multi_constraint(g, with_fix),
        "version_conditional": _version_conditional(g, with_fix),
        "cross_device": _cross_device(g),
        "error_code": _error_code(g),
    }
    counts = per_type if isinstance(per_type, dict) else dict.fromkeys(candidates, per_type)
    samples: list[PathSample] = []
    for qtype, cands in candidates.items():
        prefer_multi = qtype in ("version_conditional", "cross_device")
        samples += _take(rng, cands, counts.get(qtype, 0), prefer_multi)
    return [
        s.model_copy(update={"answer_texts": [g.text[n] for n in s.answer_nodes]}) for s in samples
    ]


class _Generated(BaseModel):
    question: str
    answer: str
    key_facts: list[str]


_GEN_PROMPT = """Write one realistic question a customer might ask Apple Support, plus a short
reference answer and 1-3 key facts.
Question type: {qtype}. {hint}

The correct answer (from the sources): {answer}

Rules:
- The question must NOT name or hint at the answer; it describes the customer's situation.
- The reference answer restates the correct answer in 1-2 sentences using ONLY information in
  the sources. No speculation, no reasons that are not in the sources.
- Key facts are short phrases taken from the sources that a correct answer must contain.
- Do not mention "knowledge graph", facts or sources. Use natural customer wording.

Facts:
{facts}

Sources:
{sources}"""
_HINTS = {
    "single_hop": "Describe the problem and ask how to fix it.",
    "multi_constraint": (
        "The customer has BOTH problems from the facts (name the devices involved) and asks "
        "what one thing they could try that helps with both."
    ),
    "cross_device": (
        "The customer uses the first device together with the second one and describes the "
        "problem; ask how to fix it."
    ),
    "version_conditional": (
        "Describe the problem and ask what to do and whether a particular software version is "
        "needed. Do not name the version."
    ),
    "error_code": "The question should mention the error code and ask what to do.",
}
_MAX_SOURCE_CHARS = 1200
_MAX_SOURCES = 4


def generate_question(
    client: LLMClient, sample: PathSample, chunk_text: dict[str, str], model: str, qid: str
) -> Question | None:
    """One question per path, or None if the model's output fails validation twice."""
    sources = "\n\n".join(
        chunk_text.get(c, "")[:_MAX_SOURCE_CHARS] for c in sample.chunk_ids[:_MAX_SOURCES]
    )
    prompt = _GEN_PROMPT.format(
        qtype=sample.qtype,
        hint=_HINTS.get(sample.qtype, ""),
        facts="\n".join(sample.facts),
        sources=sources,
        answer="; ".join(sample.answer_texts or sample.answer_nodes),
    )
    req = LLMRequest(
        model=model,
        messages=[ChatMessage(role="user", content=prompt)],
        num_ctx=4096,
        max_tokens=300,
        json_schema=_Generated.model_json_schema(),
    )
    try:
        g = complete_structured(client, req, _Generated)
    except (StructuredOutputError, RuntimeError):
        return None
    return Question(
        qid=qid,
        question=g.question.strip(),
        qtype=sample.qtype,
        gold_answer=g.answer.strip(),
        key_facts=g.key_facts[:3],
        gold_chunk_ids=sample.chunk_ids,
        gold_seed_nodes=sample.seed_nodes,
        gold_answer_nodes=sample.answer_nodes,
        source="generated",
        split="test",
        verified=False,
    )


def _screen_line(q: Question) -> str:
    if q.screen is None:
        return "AUTO-SCREEN: not run"
    verdict = "PASS" if q.screen.passed else "FLAGGED"
    return f"AUTO-SCREEN: {verdict} - {q.screen.reason}"


def verify_loop(
    questions: list[Question],
    chunk_text: dict[str, str],
    save: Callable[[list[Question]], None],
    reviewer: str,
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
) -> int:
    """y = correct and answerable from evidence, n = reject (removed), s = skip, q = quit."""
    done = 0
    i = 0
    todo = sum(not q.verified for q in questions)
    while i < len(questions):
        q = questions[i]
        if q.verified:
            i += 1
            continue
        evidence = "\n---\n".join(
            f"[{c}]\n{chunk_text.get(c, '?')[:600]}" for c in q.gold_chunk_ids
        )
        show(
            f"\n({done + 1}/{todo}) [{q.qid} | {q.qtype}] {q.question}\nGOLD: {q.gold_answer}"
            f"\nKEY FACTS: {'; '.join(q.key_facts)}\n{_screen_line(q)}\nEVIDENCE:\n{evidence}"
        )
        key = ask("[y]es  [n]o/reject  [s]kip  [q]uit > ").strip().lower()[:1]
        if key == "q":
            break
        if key == "y":
            questions[i] = q.model_copy(update={"verified": True, "verified_by": reviewer})
            done += 1
            save(questions)
        elif key == "n":
            questions.pop(i)
            save(questions)
            continue
        i += 1
    return done
