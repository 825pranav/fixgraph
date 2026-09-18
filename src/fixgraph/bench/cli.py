"""`fixgraph index build` and `fixgraph bench run | report | ... | label-split |
judge-calibrate`."""

import json
import logging
from pathlib import Path
from typing import cast, get_args

import typer
from tqdm import tqdm

from fixgraph.bench import run as R
from fixgraph.bench.combo import CacheRow
from fixgraph.bench.judge import JudgeExample, JudgeVariant
from fixgraph.bench.schema import Question, QuestionFilter, read_questions, select_questions
from fixgraph.core.config import Settings, load_settings
from fixgraph.ingest.store import read_articles, read_chunks
from fixgraph.retrieval.index import chunk_document

logger = logging.getLogger(__name__)
index_app = typer.Typer(no_args_is_help=True, help="Search indexes (Qdrant local mode).")
app = typer.Typer(no_args_is_help=True, help="TroubleshootQA benchmark.")


def _chunk_docs(settings: Settings) -> tuple[dict[str, str], dict[str, str]]:
    """chunk_id -> indexed document (title + text), chunk_id -> text shown to the answer model."""
    paths = settings.paths
    titles = {a.article_id: a.title for a in read_articles(paths.articles)}
    chunks = read_chunks(paths.chunks)
    docs = {c.chunk_id: chunk_document(c, titles.get(c.article_id, "")) for c in chunks}
    return docs, docs


@index_app.command("build")
def index_build() -> None:
    """Embed chunks (dense + BM25) and KG node names into Qdrant local mode."""
    import polars as pl

    from fixgraph.embeddings import SentenceTransformerEmbedder
    from fixgraph.kg.store import read_kg
    from fixgraph.retrieval.index import build_chunk_index, build_node_index, open_client

    settings = load_settings()
    paths = settings.paths
    titles = {a.article_id: a.title for a in read_articles(paths.articles)}
    chunks = read_chunks(paths.chunks)
    embedder = SentenceTransformerEmbedder()
    client = open_client(paths.index)
    try:
        build_chunk_index(client, chunks, titles, embedder, paths.index)
        if (paths.kg / "nodes.parquet").exists():
            kg = read_kg(paths.kg)
            nodes = kg.nodes.filter(pl.col("label") != "Article")
            names = {
                (r["node_id"], r["label"], r["canonical_text"]) for r in nodes.iter_rows(named=True)
            }
            labels = dict(zip(nodes["node_id"], nodes["label"], strict=True))
            for nid, surface in kg.mentions.select("node_id", "surface").iter_rows():
                if nid in labels:
                    names.add((nid, labels[nid], surface))  # surface forms act as aliases
            build_node_index(client, sorted(names), embedder)
    finally:
        client.close()
        embedder.release()
    typer.echo(f"indexed {len(chunks)} chunks -> {paths.index}")


_WHICH_HELP = "all | screened (auto-screen passed or verified) | verified (see verified_by)."


def _load_questions(questions_file: str, which: str, limit: int | None) -> list[Question]:
    if which not in get_args(QuestionFilter):
        raise typer.BadParameter(f"--questions must be one of {get_args(QuestionFilter)}")
    selected = select_questions(read_questions(Path(questions_file)), cast(QuestionFilter, which))
    return selected[: limit or None]


def _unload(settings: Settings, model: str) -> None:
    if settings.llm.backend == "ollama":
        from fixgraph.llm.ollama import OllamaClient

        c = OllamaClient(settings.llm.base_url)
        c.unload(model)
        c.close()


@app.command("run")
def bench_run(
    questions_file: str = typer.Option("data/bench/dev_handwritten.jsonl"),
    run_name: str = typer.Option("dev"),
    systems: str = typer.Option("S0,S1,S2,S3"),
    stages: str = typer.Option("mentions,retrieve,answer,judge,report"),
    limit: int | None = typer.Option(None, help="Only the first N questions (smoke runs)."),
    type_weights: bool = typer.Option(True, help="--no-type-weights = PPR ablation."),
    kg_dir: str | None = typer.Option(None, help="Alternative KG (e.g. data/kg_nocanon)."),
    which: str = typer.Option("all", "--questions", help=_WHICH_HELP),
    judge_variant: str = typer.Option("v1", help="Judge prompt v1-v4 (D33)."),
) -> None:
    """Run benchmark stages; one heavy model on the GPU at a time."""
    from fixgraph.llm.factory import build_llm_client

    settings = load_settings()
    paths = settings.paths
    questions = _load_questions(questions_file, which, limit)
    out = paths.results / run_name
    system_list = [s.strip() for s in systems.split(",") if s.strip()]
    stage_list = [s.strip() for s in stages.split(",")]
    _, chunk_text = _chunk_docs(settings)
    answer_model, judge_model = settings.llm.answer_model, settings.llm.judge_model

    if "mentions" in stage_list:
        client = build_llm_client(settings)
        try:
            R.stage_mentions(questions, client, settings.llm.linking_model, out / "mentions.jsonl")
        finally:
            client.close()
            _unload(settings, settings.llm.linking_model)

    if "retrieve" in stage_list:
        retrievals = _retrieve(settings, questions, system_list, out, type_weights, kg_dir)
        logger.info("retrieved %d rows", len(retrievals))

    if "answer" in stage_list:
        retrievals = [R.RetrievalRow.model_validate(r) for r in R._read(out / "retrieval.jsonl")]
        client = build_llm_client(settings)
        try:
            R.stage_answer(
                questions,
                retrievals,
                system_list,
                chunk_text,
                client,
                answer_model,
                out / "answers.jsonl",
            )
        finally:
            client.close()
            _unload(settings, answer_model)

    if "judge" in stage_list:
        answers = [R.AnswerRow.model_validate(r) for r in R._read(out / "answers.jsonl")]
        client = build_llm_client(settings)
        try:
            variant = _judge_variant(judge_variant)
            R.stage_judge(
                questions,
                answers,
                chunk_text,
                client,
                judge_model,
                out / "judged.jsonl",
                variant,
                _judge_examples(settings) if variant == "v4" else [],
            )
        finally:
            client.close()
            _unload(settings, judge_model)

    if "report" in stage_list:
        report_(run_name=run_name, questions_file=questions_file, limit=limit, which=which)


def _retrieve(
    settings: Settings,
    questions: list[Question],
    system_list: list[str],
    out: Path,
    type_weights: bool,
    kg_dir: str | None,
) -> list[R.RetrievalRow]:
    from fixgraph.core.ontology import load_ontology
    from fixgraph.embeddings import SentenceTransformerEmbedder
    from fixgraph.kg.store import read_kg
    from fixgraph.retrieval.base import NoRetrieval, Retriever
    from fixgraph.retrieval.graph import build_graph_index
    from fixgraph.retrieval.graphrag import PathRetriever, PPRRetriever
    from fixgraph.retrieval.hybrid import HybridRetriever
    from fixgraph.retrieval.index import load_bm25, open_client
    from fixgraph.retrieval.linking import EntityLinker, merge_mentions, rule_mentions
    from fixgraph.retrieval.rerank import CrossEncoderReranker

    paths = settings.paths
    docs, _ = _chunk_docs(settings)
    embedder = SentenceTransformerEmbedder()
    reranker = CrossEncoderReranker()
    client = open_client(paths.index)
    try:
        hybrid = HybridRetriever(client, embedder, load_bm25(paths.index), reranker, docs)
        systems: dict[str, Retriever] = {}
        if "S1" in system_list:
            systems["S1"] = hybrid
        if {"S2", "S3", "S2L", "S4"} & set(system_list):
            kg = read_kg(Path(kg_dir) if kg_dir else paths.kg)
            graph = build_graph_index(kg, use_type_weights=type_weights)
            counts = {graph.node_ids[i]: len(c) for i, c in graph.node_chunks.items()}
            linker = EntityLinker(client, embedder, counts)
            onto = load_ontology()
            llm_m = R.load_mentions(out / "mentions.jsonl")
            mentions = {
                q.question: merge_mentions(
                    llm_m.get(q.question, rule_mentions("", onto)), rule_mentions(q.question, onto)
                )
                for q in questions
            }
            if "S2" in system_list:
                systems["S2"] = PPRRetriever(graph, linker, hybrid, mentions)
            if "S3" in system_list:
                systems["S3"] = PathRetriever(graph, linker, hybrid, mentions)
            if {"S2L", "S4"} & set(system_list):
                from fixgraph.bench.schema import article_of
                from fixgraph.ingest.links import read_links
                from fixgraph.retrieval.graph import add_document_layer
                from fixgraph.retrieval.graphrag import FusionRetriever

                links = [
                    (x.src_article, x.dst_article, x.src_chunk_id)
                    for x in read_links(paths.corpus / "links.parquet")
                ]
                if not links:
                    raise typer.BadParameter("S2L/S4 need data/corpus/links.parquet")
                s2l = PPRRetriever(add_document_layer(graph, links, article_of), linker, hybrid,
                                   mentions)  # fmt: skip
                s2l.name = "S2L"
                if "S2L" in system_list:
                    systems["S2L"] = s2l
                if "S4" in system_list:
                    systems["S4"] = FusionRetriever(hybrid, s2l)
        if "S0" in system_list:
            systems["S0"] = NoRetrieval()
        return R.stage_retrieve(questions, systems, out / "retrieval.jsonl")
    finally:
        client.close()
        reranker.release()
        embedder.release()


@app.command("report")
def report_(
    run_name: str = typer.Option("dev"),
    questions_file: str = typer.Option("data/bench/dev_handwritten.jsonl"),
    limit: int | None = typer.Option(None),
    which: str = typer.Option("all", "--questions", help=_WHICH_HELP),
) -> None:
    """Metrics with bootstrap CIs, significance tests, per-type table; logs to MLflow."""
    settings = load_settings()
    out = settings.paths.results / run_name
    questions = _load_questions(questions_file, which, limit)
    retrievals = [R.RetrievalRow.model_validate(r) for r in R._read(out / "retrieval.jsonl")]
    answers = [R.AnswerRow.model_validate(r) for r in R._read(out / "answers.jsonl")]
    judged = [R.JudgeRow.model_validate(r) for r in R._read(out / "judged.jsonl")]
    report = R.build_report(questions, retrievals, answers, judged)
    meta = out / "judge_meta.json"
    report["judge"] = (
        json.loads(meta.read_text(encoding="utf-8"))
        if meta.exists()
        else {"model": settings.llm.judge_model, "variant": "v1"}  # runs judged before D33
    )
    stem = "report" if which == "all" else f"report_{which}"  # one file per question filter
    (out / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = R.render_markdown(report)
    (out / f"{stem}.md").write_text(md, encoding="utf-8")
    try:
        import mlflow

        mlflow.set_tracking_uri((settings.paths.root.parent / "mlruns").as_uri())
        mlflow.set_experiment("troubleshootqa")
        with mlflow.start_run(run_name=f"{run_name}-{which}"):
            mlflow.log_params({"questions": questions_file, "filter": which, "n": len(questions)})
            for system, metrics in report["table"].items():
                for name, v in metrics.items():
                    if v["mean"] == v["mean"]:  # skip NaN
                        mlflow.log_metric(f"{system}.{name.replace('@', '_at_')}", float(v["mean"]))
            mlflow.log_artifact(str(out / f"{stem}.md"))
    except Exception as exc:  # MLflow is bookkeeping; never fail the report on it
        logger.warning("MLflow logging skipped: %s", exc)
    typer.echo(md)


# Default mix for the test set: multi-article types dominate; single_hop is the control group.
DEFAULT_MIX = (
    "single_hop=40,multi_constraint=80,version_conditional=30,cross_device=30,error_code=11"
)


def _parse_mix(mix: str) -> dict[str, int]:
    """Parse `single_hop=40,multi_constraint=80` into {"single_hop": 40, "multi_constraint": 80}."""
    out: dict[str, int] = {}
    for part in mix.split(","):
        name, _, n = part.partition("=")
        out[name.strip()] = int(n)
    return out


@app.command("generate")
def bench_generate(
    mix: str = typer.Option(DEFAULT_MIX, help="Paths per question type, e.g. single_hop=40,..."),
    out_file: str = typer.Option("data/bench/generated.jsonl"),
) -> None:
    """Generate questions from KG paths with the judge-class model (unverified until reviewed)."""
    from fixgraph.bench.generate import generate_question, sample_paths
    from fixgraph.bench.schema import write_questions
    from fixgraph.kg.store import read_kg
    from fixgraph.llm.factory import build_llm_client

    settings = load_settings()
    _, chunk_text = _chunk_docs(settings)
    samples = sample_paths(read_kg(settings.paths.kg), _parse_mix(mix), seed=settings.seed)
    client = build_llm_client(settings)
    qs = []
    try:
        for i, s in enumerate(tqdm(samples, desc="generate")):
            q = generate_question(client, s, chunk_text, settings.llm.judge_model, f"t{i:03d}")
            if q is not None:
                qs.append(q)
    finally:
        client.close()
        _unload(settings, settings.llm.judge_model)
    write_questions(qs, Path(out_file))
    typer.echo(f"{len(qs)} questions from {len(samples)} paths -> {out_file}")


@app.command("screen")
def bench_screen(
    questions_file: str = typer.Option("data/bench/generated.jsonl"),
    rescreen: bool = typer.Option(False, help="Re-run questions that already have a verdict."),
) -> None:
    """Automatic pre-screen (answerable? supported? leaks answer? truly multi-article?).

    Advisory only: results are stored on each question and shown during `bench verify`."""
    from collections import Counter

    from fixgraph.bench.schema import write_questions
    from fixgraph.bench.screen import review_order, screen_question
    from fixgraph.kg.store import read_kg
    from fixgraph.llm.factory import build_llm_client

    settings = load_settings()
    _, chunk_text = _chunk_docs(settings)
    nodes = read_kg(settings.paths.kg).nodes
    node_text = dict(zip(nodes["node_id"], nodes["canonical_text"], strict=True))
    path = Path(questions_file)
    qs = read_questions(path)
    client = build_llm_client(settings)
    try:
        for i, q in enumerate(tqdm(qs, desc="screen")):
            if q.screen is None or rescreen:
                qs[i] = q.model_copy(
                    update={
                        "screen": screen_question(
                            client, q, chunk_text, settings.llm.judge_model, node_text
                        )
                    }
                )
                write_questions(qs, path)  # resumable: progress survives a crash
    finally:
        client.close()
        _unload(settings, settings.llm.judge_model)
    qs = review_order(qs)
    write_questions(qs, path)
    passed = Counter(q.qtype for q in qs if q.screen and q.screen.passed)
    multi = sum(bool(q.screen and q.screen.passed and q.screen.needs_multiple_articles) for q in qs)
    typer.echo(f"passed {sum(passed.values())}/{len(qs)} ({dict(passed)}); {multi} need >1 article")


@app.command("verify")
def bench_verify(
    questions_file: str = typer.Option("data/bench/generated.jsonl"),
    reviewer: str = typer.Option("developer"),
) -> None:
    """Human verification per question (keys: y = keep, n = reject, s = skip, q = quit).

    Screen-passed questions come first, so stopping early keeps the most likely keepers."""
    from fixgraph.bench.generate import verify_loop
    from fixgraph.bench.schema import write_questions
    from fixgraph.bench.screen import review_order

    settings = load_settings()
    _, chunk_text = _chunk_docs(settings)
    path = Path(questions_file)
    qs = review_order(read_questions(path))
    n = verify_loop(qs, chunk_text, lambda q: write_questions(q, path), reviewer)
    typer.echo(f"verified {n}; {sum(q.verified for q in qs)}/{len(qs)} verified in file")


@app.command("label")
def bench_label(
    run_name: str = typer.Option("dev"),
    questions_file: str = typer.Option("data/bench/dev_handwritten.jsonl"),
    n: int = typer.Option(80, help="Answers to label (spec: 80-100)."),
    labeler: str = typer.Option("developer"),
    which: str = typer.Option("all", "--questions", help=_WHICH_HELP),
) -> None:
    """Blind human labels for judge validation (keys: 1 = correct, 5 = partial, 0 = wrong)."""
    from fixgraph.bench.validate import (
        agreement,
        label_loop,
        load_run,
        read_labels,
        sample_for_labeling,
        write_labels,
    )

    settings = load_settings()
    run_dir = settings.paths.results / run_name
    questions = {q.qid: q for q in _load_questions(questions_file, which, None)}
    answers, judged = load_run(run_dir)
    labels_path = settings.paths.bench / f"judge_labels_{run_name}.jsonl"
    todo = sample_for_labeling(list(answers), questions, n)
    labels = label_loop(
        todo,
        questions,
        answers,
        read_labels(labels_path),
        lambda x: write_labels(x, labels_path),
        labeler,
    )
    if labels:
        typer.echo(agreement(labels, judged).model_dump_json(indent=2))


@app.command("kappa")
def bench_kappa(run_name: str = typer.Option("dev")) -> None:
    """Cohen's kappa between human labels and the LLM judge for a run."""
    from fixgraph.bench.validate import agreement, load_run, read_labels

    settings = load_settings()
    _, judged = load_run(settings.paths.results / run_name)
    labels = read_labels(settings.paths.bench / f"judge_labels_{run_name}.jsonl")
    typer.echo(agreement(labels, judged).model_dump_json(indent=2))


# --- judge calibration (D33) ----------------------------------------------------------------

_LABEL_RUN = "test"  # the run whose blind labels calibrate the judge
_SPLIT_FILE = "judge_label_split_test.json"
_CALIBRATION = Path("results/judge/calibration.json")


def _judge_variant(name: str) -> JudgeVariant:
    from fixgraph.bench.judge import VARIANTS

    if name not in VARIANTS:
        raise typer.BadParameter(f"judge variant must be one of {VARIANTS}")
    return cast(JudgeVariant, name)


def _judge_examples(settings: Settings) -> list[JudgeExample]:
    """v4 worked examples: fixed dev items of the calibration split (see pick_examples)."""
    from fixgraph.bench.validate import LabelSplit, load_run, pick_examples, read_labels

    paths = settings.paths
    labels = read_labels(paths.bench / f"judge_labels_{_LABEL_RUN}.jsonl")
    split = LabelSplit.model_validate_json((paths.bench / _SPLIT_FILE).read_text(encoding="utf-8"))
    answers, _ = load_run(paths.results / _LABEL_RUN)
    questions = {q.qid: q for q in read_questions(paths.bench / "generated.jsonl")}
    score = {(x.qid, x.system): x.score for x in labels}
    out = []
    for qid, system in pick_examples(labels, split.dev):
        q = questions[qid]
        out.append(
            JudgeExample(
                question=q.question,
                reference=q.gold_answer,
                key_facts=q.key_facts,
                answer=answers[(qid, system)],
                score=score[(qid, system)],
            )
        )
    return out


@app.command("label-split")
def label_split(seed: int = typer.Option(13)) -> None:
    """Fix which judge labels may be used to design prompts (dev, 1/3) vs held out (2/3)."""
    from fixgraph.bench.validate import read_labels, split_labels

    paths = load_settings().paths
    target = paths.bench / _SPLIT_FILE
    if target.exists():
        raise typer.BadParameter(f"{target} exists; the split is fixed once (D33)")
    split = split_labels(read_labels(paths.bench / f"judge_labels_{_LABEL_RUN}.jsonl"), seed=seed)
    target.write_text(split.model_dump_json(indent=1), encoding="utf-8")
    typer.echo(f"dev {len(split.dev)} / heldout {len(split.heldout)} -> {target}")


@app.command("judge-calibrate")
def judge_calibrate(
    variants: list[str] = typer.Option(..., "--variant", help="Repeatable: v1 v2 v3 v4."),
    split_name: str = typer.Option("dev", "--split", help="dev | heldout (heldout runs once)."),
) -> None:
    """Agreement (kappa) of judge prompt variants with the blind labels on one split; every
    run is appended to results/judge/calibration.json."""
    from fixgraph.bench.judge import judge_answer
    from fixgraph.bench.validate import LabelSplit, agreement, load_run, pick_examples, read_labels
    from fixgraph.llm.factory import build_llm_client

    settings = load_settings()
    paths = settings.paths
    if split_name not in ("dev", "heldout"):
        raise typer.BadParameter("--split must be dev or heldout")
    log = json.loads(_CALIBRATION.read_text(encoding="utf-8")) if _CALIBRATION.exists() else []
    if split_name == "heldout" and any(r["split"] == "heldout" for r in log):
        raise typer.BadParameter("the held-out split was already scored (D33: once only)")
    if split_name == "heldout" and len(variants) != 1:
        raise typer.BadParameter("score exactly one chosen variant on the held-out split")
    labels = read_labels(paths.bench / f"judge_labels_{_LABEL_RUN}.jsonl")
    split = LabelSplit.model_validate_json((paths.bench / _SPLIT_FILE).read_text(encoding="utf-8"))
    keys = set(split.dev if split_name == "dev" else split.heldout)
    answers, _ = load_run(paths.results / _LABEL_RUN)
    questions = {q.qid: q for q in read_questions(paths.bench / "generated.jsonl")}
    client = build_llm_client(settings)
    try:
        for name in variants:
            variant = _judge_variant(name)
            examples = _judge_examples(settings) if variant == "v4" else []
            excluded = set(pick_examples(labels, split.dev)) if variant == "v4" else set()
            scores: dict[tuple[str, str], float] = {}
            for qid, system in tqdm(sorted(keys - excluded), desc=f"judge {variant}"):
                text = answers[(qid, system)]
                out = judge_answer(
                    client, questions[qid], text, not text, settings.llm.judge_model, variant,
                    examples,
                )  # fmt: skip
                scores[(qid, system)] = out.score
            ag = agreement([x for x in labels if (x.qid, x.system) in scores], scores)
            entry = {
                "variant": variant,
                "split": split_name,
                "judge_model": settings.llm.judge_model,
                "excluded_examples": sorted([list(k) for k in excluded]),
                **ag.model_dump(),
            }
            log.append(entry)
            typer.echo(json.dumps(entry))
    finally:
        client.close()
        _unload(settings, settings.llm.judge_model)
    _CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    _CALIBRATION.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")


# --- bridge benchmark (D35) -------------------------------------------------------------------

_BRIDGE_FILE = "data/bench/bridge.jsonl"
_BRIDGE_FROZEN = Path("results/bridge/frozen.json")


@app.command("bridge-generate")
def bridge_generate(out_file: str = typer.Option(_BRIDGE_FILE)) -> None:
    """Matched bridge / direct question pairs from Apple's conditional links (D35 rule)."""
    from collections import defaultdict

    from fixgraph.bench.bridge import generate_pair, select_candidates
    from fixgraph.bench.schema import write_questions
    from fixgraph.ingest.links import read_links
    from fixgraph.llm.factory import build_llm_client

    settings = load_settings()
    paths = settings.paths
    if _BRIDGE_FROZEN.exists():
        raise typer.BadParameter("the bridge set is frozen (D35); it is not regenerated")
    _, chunk_text = _chunk_docs(settings)
    titles = {a.article_id: a.title for a in read_articles(paths.articles)}
    by_article: dict[str, list[str]] = defaultdict(list)
    for c in read_chunks(paths.chunks):
        by_article[c.article_id].append(c.chunk_id)
    cands = select_candidates(read_links(paths.corpus / "links.parquet"), titles, by_article,
                              seed=settings.seed)  # fmt: skip
    client = build_llm_client(settings)
    results = []
    try:
        for i, cand in enumerate(tqdm(cands, desc="bridge")):
            results.append(
                generate_pair(client, cand, chunk_text, settings.llm.judge_model, f"b{i:03d}")
            )
    finally:
        client.close()
        _unload(settings, settings.llm.judge_model)
    qs = [q for r in results for q in r.questions]
    write_questions(qs, Path(out_file))
    log = Path("results/bridge/generation.json")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        json.dumps(
            {
                "candidates": len(cands),
                "pairs_kept": sum(r.ok for r in results),
                "rejections": [
                    {
                        "src": r.candidate.link.src_article,
                        "dst": r.candidate.link.dst_article,
                        "reason": r.reason,
                    }
                    for r in results
                    if not r.ok
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    typer.echo(f"{sum(r.ok for r in results)}/{len(cands)} pairs -> {out_file}")


@app.command("bridge-freeze")
def bridge_freeze(questions_file: str = typer.Option(_BRIDGE_FILE)) -> None:
    """Freeze the reviewed bridge set: record its SHA-256 and counts before any system runs."""
    import hashlib

    if _BRIDGE_FROZEN.exists():
        raise typer.BadParameter(f"already frozen: {_BRIDGE_FROZEN}")
    raw = Path(questions_file).read_bytes()
    qs = read_questions(Path(questions_file))
    kept = [q for q in qs if q.verified]
    pairs = {q.qid[:-1] for q in kept}
    complete = {p for p in pairs if {f"{p}a", f"{p}d"} <= {q.qid for q in kept}}
    if len(complete) != len(pairs):
        raise typer.BadParameter("review must keep or drop pairs as a unit")
    _BRIDGE_FROZEN.parent.mkdir(parents=True, exist_ok=True)
    _BRIDGE_FROZEN.write_text(
        json.dumps(
            {
                "file": questions_file,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "questions": len(qs),
                "verified_pairs": len(complete),
                "verified_by": sorted({q.verified_by for q in kept}),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    typer.echo(f"frozen {len(complete)} verified pairs ({_BRIDGE_FROZEN})")


@app.command("bridge-report")
def bridge_report_cmd(
    run_name: str = typer.Option("bridge"),
    questions_file: str = typer.Option(_BRIDGE_FILE),
    out: str = typer.Option("results/bridge/bridge_report.json"),
) -> None:
    """Answer-chunk hit@8, correctness, hop cost and crossover tests on verified pairs (D35)."""
    from fixgraph.bench.bridge import bridge_report

    settings = load_settings()
    run_dir = settings.paths.results / run_name
    qs = [q for q in read_questions(Path(questions_file)) if q.verified]
    ranked = {
        (r["qid"], r["system"]): r["result"]["chunk_ids"]
        for r in R._read(run_dir / "retrieval.jsonl")
    }
    judged_file = run_dir / "judged.jsonl"  # optional: retrieval-only runs have no judge stage
    judged = (
        {(r["qid"], r["system"]): float(r["judge"]["score"]) for r in R._read(judged_file)}
        if judged_file.exists()
        else {}
    )
    report = bridge_report(qs, ranked, judged)
    meta = run_dir / "judge_meta.json"
    if meta.exists():
        report["judge"] = json.loads(meta.read_text(encoding="utf-8"))
    frozen = Path("results/bridge/frozen.json")
    if frozen.exists():
        report["frozen"] = json.loads(frozen.read_text(encoding="utf-8"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


@app.command("bridge-import")
def bridge_import(
    candidates_file: str = typer.Option(..., help="JSON list of BridgeCandidate (seeded rule)."),
    pairs_files: list[str] = typer.Option(..., "--pairs", help="JSONL pairs written by Claude."),
    out_file: str = typer.Option(_BRIDGE_FILE),
) -> None:
    """Run externally written pairs (D35 amendment) through the same mechanical checks."""
    from fixgraph.bench.bridge import BridgeCandidate, BridgePair, BridgeResult, check_pair
    from fixgraph.bench.schema import write_questions

    if _BRIDGE_FROZEN.exists():
        raise typer.BadParameter("the bridge set is frozen (D35)")
    settings = load_settings()
    _, chunk_text = _chunk_docs(settings)
    raw = json.loads(Path(candidates_file).read_text(encoding="utf-8"))
    cands = {f"b{i:03d}": BridgeCandidate.model_validate(c) for i, c in enumerate(raw)}
    written: dict[str, dict[str, object]] = {}
    for f in pairs_files:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                written[row["pid"]] = row
    results: list[BridgeResult] = []
    for pid, cand in cands.items():
        row = written.get(pid)
        if row is None or row.get("skip"):
            reason = "not written" if row is None else f"writer skipped: {row.get('skip_reason')}"
            results.append(BridgeResult(candidate=cand, ok=False, reason=reason))
            continue
        pair = BridgePair.model_validate({k: row[k] for k in BridgePair.model_fields})
        results.append(check_pair(pair, cand, chunk_text, pid, source="bridge-claude"))
    write_questions([q for r in results for q in r.questions], Path(out_file))
    log = Path("results/bridge/generation.json")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        json.dumps(
            {
                "generator": "claude-opus-5-5 (D35 amendment)",
                "candidates": len(cands),
                "pairs_passed_checks": sum(r.ok for r in results),
                "rejections": [
                    {"pid": pid, "reason": r.reason}
                    for pid, r in zip(cands, results, strict=True)
                    if not r.ok
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    typer.echo(f"{sum(r.ok for r in results)}/{len(cands)} pairs passed -> {out_file}")


# --- graph + hybrid combinations (D37+) ---------------------------------------------------------


@app.command("miss-analysis")
def miss_analysis_cmd(
    kg_dir: str = typer.Option("data/kg_verified"),
    out: str = typer.Option("results/combo/miss_analysis.json"),
) -> None:
    """Where S1 misses gold chunks, and which graph routes could reach them (D37 ceiling)."""
    from fixgraph.bench.miss import Expander, miss_analysis
    from fixgraph.ingest.links import read_links
    from fixgraph.kg.store import read_kg
    from fixgraph.retrieval.graph import build_graph_index

    settings = load_settings()
    paths = settings.paths
    graph = build_graph_index(read_kg(Path(kg_dir)))
    links = [(x.src_article, x.dst_article) for x in read_links(paths.corpus / "links.parquet")]
    exp = Expander(graph, links, [c.chunk_id for c in read_chunks(paths.chunks)])
    report: dict[str, object] = {"kg_dir": kg_dir}
    for name, qfile, run in (
        ("main93", "data/bench/generated.jsonl", "test_verified"),
        ("bridge156", _BRIDGE_FILE, "bridge"),
    ):
        qs = [q for q in read_questions(Path(qfile)) if q.verified]
        ranked = {
            r["qid"]: r["result"]["chunk_ids"]
            for r in R._read(paths.results / run / "retrieval.jsonl")
            if r["system"] == "S1"
        }
        report[name] = miss_analysis(qs, ranked, exp)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name in ("main93", "bridge156"):
        r = report[name]
        assert isinstance(r, dict)
        typer.echo(f"{name}: {r['n_missed_chunks']}/{r['n_gold_chunks']} gold chunks missed")
        for depth in ("expand_from_top3", "expand_from_top8"):
            routes = r[depth]["routes"]
            typer.echo(f"  {depth}: " + ", ".join(
                f"{k}={v['share']} (pool {v.get('median_pool_chunks', '-')})"
                for k, v in routes.items()))  # fmt: skip


_COMBO = Path("results/combo")
_COMBO_SETS = (("data/bench/generated.jsonl", "main"), (_BRIDGE_FILE, "bridge"))


def _combo_questions(split: str | None = None) -> list[Question]:
    qs = [q for f, _ in _COMBO_SETS for q in read_questions(Path(f)) if q.verified]
    if split is None:
        return qs
    assignment = json.loads((_COMBO / "split.json").read_text(encoding="utf-8"))["assignment"]
    return [q for q in qs if assignment[q.qid] == split]


@app.command("combo-split")
def combo_split(seed: int = typer.Option(13)) -> None:
    """Seeded 60/40 dev/test split of the 93 main + 156 bridge questions; hash-freezes test."""
    import hashlib

    from fixgraph.bench.combo import split_questions

    target = _COMBO / "split.json"
    if target.exists():
        raise typer.BadParameter(f"{target} exists; the split is fixed once (D37)")
    qs = _combo_questions()
    assignment = split_questions(qs, seed=seed)
    test_ids = sorted(q for q, s in assignment.items() if s == "test")
    counts: dict[str, dict[str, int]] = {}
    for q in qs:
        counts.setdefault(q.qtype, {"dev": 0, "test": 0})[assignment[q.qid]] += 1
    _COMBO.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "seed": seed,
                "test_sha256": hashlib.sha256("\n".join(test_ids).encode()).hexdigest(),
                "counts": counts,
                "assignment": dict(sorted(assignment.items())),
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    typer.echo(json.dumps(counts))


@app.command("combo-cache")
def combo_cache(split: str = typer.Option(..., help="dev | test")) -> None:
    """One GPU pass: fused candidates, cross-encoder scores for S1's pool and the Apple-link
    pool, and timings, cached for the split's questions."""
    from fixgraph.bench.combo import build_cache
    from fixgraph.embeddings import SentenceTransformerEmbedder
    from fixgraph.ingest.links import read_links
    from fixgraph.retrieval.hybrid import HybridRetriever
    from fixgraph.retrieval.index import load_bm25, open_client
    from fixgraph.retrieval.rerank import CrossEncoderReranker

    if split == "test" and not (_COMBO / "finalists.json").exists():
        raise typer.BadParameter("record the finalists (D38) before touching the test split")
    settings = load_settings()
    paths = settings.paths
    docs, _ = _chunk_docs(settings)
    links = [(x.src_article, x.dst_article) for x in read_links(paths.corpus / "links.parquet")]
    embedder, reranker = SentenceTransformerEmbedder(), CrossEncoderReranker()
    client = open_client(paths.index)
    try:
        hybrid = HybridRetriever(client, embedder, load_bm25(paths.index), reranker, docs)
        rows = build_cache(_combo_questions(split), hybrid, docs, links, list(docs))
    finally:
        client.close()
        reranker.release()
        embedder.release()
    out = paths.results / "combo" / f"cache_{split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")
    typer.echo(f"{len(rows)} questions cached -> {out}")


def _combo_rows(split: str) -> dict[str, CacheRow]:
    path = load_settings().paths.results / "combo" / f"cache_{split}.jsonl"
    rows = [CacheRow.model_validate_json(x) for x in path.read_text(encoding="utf-8").splitlines()]
    return {r.qid: r for r in rows}


@app.command("combo-dev")
def combo_dev() -> None:
    """Score S1 and every D37 grid config on the dev split; log all of them."""
    from statistics import median

    from fixgraph.bench.combo import Config, grid, per_question, rank, recovered_broken, summarize

    qs = _combo_questions("dev")
    rows = _combo_rows("dev")
    threshold = median(rows[q.qid].ce[rows[q.qid].s1[0]] for q in qs)
    s1 = {q.qid: rank(rows[q.qid], Config(kind="s1"), threshold) for q in qs}
    log = []
    for cfg in [Config(kind="s1"), *grid()]:
        r = {q.qid: rank(rows[q.qid], cfg, threshold) for q in qs}
        per = per_question(qs, r)
        rb = recovered_broken(qs, s1, r)
        log.append({
            "config": cfg.name,
            **{sub: summarize(qs, per, sub) for sub in ("main", "bridge", "bridge_direct")},
            "recovered": sum(v["recovered"] for v in rb.values()),
            "broken": sum(v["broken"] for v in rb.values()),
            "recovered_broken_by_type": rb,
        })  # fmt: skip
    (_COMBO / "dev_log.json").write_text(
        json.dumps({"route_threshold": threshold, "n_dev": len(qs), "configs": log}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    typer.echo(f"route threshold (dev median S1 top score) = {threshold:.3f}")
    for e in log:
        typer.echo(
            f"{e['config']:<22} main r@8 {e['main']['recall@8']}"
            f"  bridge hit {e['bridge']['answer_hit@8']}"
            f"  direct hit {e['bridge_direct']['answer_hit@8']}  +{e['recovered']}/-{e['broken']}"
        )


@app.command("combo-test")
def combo_test(out: str = typer.Option("results/combo/test_report.json")) -> None:
    """The single locked-test run of S1 vs the recorded finalists (D38). Refuses to rerun."""
    import hashlib

    from fixgraph.bench.combo import Config, test_report

    if Path(out).exists():
        raise typer.BadParameter(f"{out} exists; the test split is evaluated once (D38)")
    fin = json.loads((_COMBO / "finalists.json").read_text(encoding="utf-8"))
    split = json.loads((_COMBO / "split.json").read_text(encoding="utf-8"))
    qs = _combo_questions("test")
    ids = "\n".join(sorted(q.qid for q in qs)).encode()
    if hashlib.sha256(ids).hexdigest() != split["test_sha256"]:
        raise typer.BadParameter("test questions do not match the frozen split hash")
    report = test_report(qs, _combo_rows("test"), [Config(**c) for c in fin["finalists"]],
                         fin["route_threshold"])  # fmt: skip
    report["finalists"] = fin
    Path(out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps({k: report[k] for k in ("table", "tests_vs_S1", "recovered_broken",
                                                   "latency_s")}, indent=1))  # fmt: skip
