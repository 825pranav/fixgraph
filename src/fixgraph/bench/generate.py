"""Benchmark generation from KG paths + human verification (spec §11.2).

1. Sample schema-valid paths by question type (stratified, seeded).
2. The judge-class model writes a question from the path's facts and evidence chunks; the gold
   answer is the path's terminal node(s), gold chunks are the edges' provenance, gold seeds are
   the path's non-answer nodes.
3. Unanswerable questions: perturb a real question with a product/feature combination absent
   from the KG.
4. `verify_loop` makes human verification a few keystrokes per question.
"""

import random
from collections import defaultdict
from collections.abc import Callable

import polars as pl
from pydantic import BaseModel

from fixgraph.bench.schema import QType, Question
from fixgraph.kg.store import KG
from fixgraph.llm.base import ChatMessage, LLMClient, LLMRequest
from fixgraph.llm.structured import StructuredOutputError, complete_structured


class PathSample(BaseModel):
    qtype: QType
    facts: list[str]  # "Apple Watch EXHIBITS won't pair", ...
    answer_nodes: list[str]
    seed_nodes: list[str]
    chunk_ids: list[str]


def _edges_by(kg: KG) -> tuple[dict[str, list[dict]], dict[str, str], dict[str, str]]:  # type: ignore[type-arg]
    edges = kg.extracted_edges().filter(pl.col("extraction_model") != "ontology")
    out: dict[str, list[dict]] = defaultdict(list)  # type: ignore[type-arg]
    for r in edges.iter_rows(named=True):
        out[r["src"]].append(r)
    text = dict(zip(kg.nodes["node_id"], kg.nodes["canonical_text"], strict=True))
    label = dict(zip(kg.nodes["node_id"], kg.nodes["label"], strict=True))
    return out, text, label


def sample_paths(kg: KG, per_type: int, seed: int = 13) -> list[PathSample]:
    rng = random.Random(seed)
    out_e, text, label = _edges_by(kg)

    def fact(e: dict) -> str:  # type: ignore[type-arg]
        return f"{text[e['src']]} {e['rel']} {text[e['dst']]}"

    samples: list[PathSample] = []
    symptoms = sorted(n for n, lab in label.items() if lab == "Symptom")
    fixes_of = {s: [e for e in out_e.get(s, []) if e["rel"] == "RESOLVED_BY"] for s in symptoms}
    with_fix = [s for s in symptoms if fixes_of[s]]

    # single_hop: symptom -> fix
    for s in rng.sample(with_fix, min(per_type, len(with_fix))):
        es = fixes_of[s][:3]
        samples.append(
            PathSample(
                qtype="single_hop",
                facts=[fact(e) for e in es],
                answer_nodes=[e["dst"] for e in es],
                seed_nodes=[s],
                chunk_ids=sorted({c for e in es for c in e["source_chunk_ids"]}),
            )
        )
    # cross_device: product DEPENDS_ON product, child EXHIBITS symptom with a fix
    cands = []
    for es in out_e.values():
        deps = [e for e in es if e["rel"] == "DEPENDS_ON"]
        syms = [e for e in es if e["rel"] == "EXHIBITS" and fixes_of.get(e["dst"])]
        for d in deps:
            for se in syms:
                cands.append((d, se, fixes_of[se["dst"]][0]))
    for d, se, fe in rng.sample(cands, min(per_type, len(cands))):
        samples.append(
            PathSample(
                qtype="cross_device",
                facts=[fact(d), fact(se), fact(fe)],
                answer_nodes=[fe["dst"]],
                seed_nodes=[d["src"], d["dst"], se["dst"]],
                chunk_ids=sorted({c for e in (d, se, fe) for c in e["source_chunk_ids"]}),
            )
        )
    # version_conditional: symptom -> fix -> REQUIRES/APPLIES_TO OSVersion
    cands2 = []
    for s in with_fix:
        for fe in fixes_of[s]:
            for ve in out_e.get(fe["dst"], []):
                if ve["rel"] in ("REQUIRES", "APPLIES_TO") and label.get(ve["dst"]) == "OSVersion":
                    cands2.append((fe, ve))
    for fe, ve in rng.sample(cands2, min(per_type, len(cands2))):
        samples.append(
            PathSample(
                qtype="version_conditional",
                facts=[fact(fe), fact(ve)],
                answer_nodes=[fe["dst"], ve["dst"]],
                seed_nodes=[fe["src"]],
                chunk_ids=sorted({c for e in (fe, ve) for c in e["source_chunk_ids"]}),
            )
        )
    # error_code: code SIGNALS symptom -> fix
    cands3 = [
        (e, fixes_of[e["dst"]][0])
        for es in out_e.values()
        for e in es
        if e["rel"] == "SIGNALS" and fixes_of.get(e["dst"])
    ]
    for se, fe in rng.sample(cands3, min(per_type, len(cands3))):
        samples.append(
            PathSample(
                qtype="error_code",
                facts=[fact(se), fact(fe)],
                answer_nodes=[fe["dst"]],
                seed_nodes=[se["src"]],
                chunk_ids=sorted({c for e in (se, fe) for c in e["source_chunk_ids"]}),
            )
        )
    return samples


class _Generated(BaseModel):
    question: str
    answer: str
    key_facts: list[str]


_GEN_PROMPT = """Write one realistic question a customer might ask Apple Support, answerable
from the facts and sources below, plus a short reference answer and 1-3 key facts.
Question type: {qtype}. {hint}
Do not mention "knowledge graph", facts or sources. Use natural customer wording.

Facts:
{facts}

Sources:
{sources}"""
_HINTS = {
    "single_hop": "Ask how to fix the problem.",
    "cross_device": "Mention both devices; the answer depends on how they relate.",
    "version_conditional": "Ask whether/what OS version is needed for the fix.",
    "error_code": "The question should mention the error code and ask what to do.",
}


def generate_question(
    client: LLMClient, sample: PathSample, chunk_text: dict[str, str], model: str, qid: str
) -> Question | None:
    sources = "\n\n".join(chunk_text.get(c, "")[:1200] for c in sample.chunk_ids[:3])
    prompt = _GEN_PROMPT.format(
        qtype=sample.qtype,
        hint=_HINTS.get(sample.qtype, ""),
        facts="\n".join(sample.facts),
        sources=sources,
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
        source="generated",
        split="test",
        verified=False,
    )


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
    while i < len(questions):
        q = questions[i]
        if q.verified:
            i += 1
            continue
        evidence = "\n---\n".join(chunk_text.get(c, "?")[:600] for c in q.gold_chunk_ids)
        show(f"\n[{q.qid} | {q.qtype}] {q.question}\nGOLD: {q.gold_answer}\nEVIDENCE:\n{evidence}")
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
