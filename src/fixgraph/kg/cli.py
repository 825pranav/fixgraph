"""`fixgraph kg extract | build | stats | eval | gold-sample | annotate | verify-edges |
edge-sample | edge-eval`.

extract runs kg.run_extract over chunks.parquet; build turns the extractions into the parquet KG
(kg.build, optional LLM adjudication from kg.resolve) and writes it with kg.store; stats / eval
report graph statistics and gold-set P/R/F1 (kg.quality); gold-sample / annotate drive the
gold-set tooling (kg.annotate); verify-edges / edge-sample / edge-eval check every edge against
its source text and measure the hallucinated-edge rate (kg.verify).
Used by: cli.py (mounted as `kg`).
Uses: core.config, core.ontology, ingest.store, llm.factory, llm.ollama, embeddings.
"""

import json
import logging
from pathlib import Path

import typer

from fixgraph.core.config import load_settings
from fixgraph.core.ontology import load_ontology
from fixgraph.core.paths import DataPaths
from fixgraph.ingest.store import read_articles, read_chunks
from fixgraph.kg.build import build_kg
from fixgraph.kg.quality import GoldChunk, evaluate_extractions, graph_stats
from fixgraph.kg.resolve import llm_adjudicator
from fixgraph.kg.run_extract import output_path, read_records, run_extraction
from fixgraph.kg.store import read_kg, write_kg
from fixgraph.llm.factory import build_llm_client
from fixgraph.llm.ollama import OllamaClient

logger = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=True, help="Knowledge-graph construction.")


@app.command()
def extract(
    model: str | None = typer.Option(None, help="Defaults to llm.extraction_model."),
    limit: int | None = typer.Option(None, help="Process at most N pending chunks."),
    concurrency: int = typer.Option(2, help="Concurrent requests (match OLLAMA_NUM_PARALLEL)."),
    chunk_ids_file: str | None = typer.Option(None, help="Only chunks listed (one id per line)."),
    dry_run: bool = typer.Option(False, help="Print what would run; no LLM calls."),
    unload: bool = typer.Option(True, help="Unload the model from VRAM when done."),
) -> None:
    """Extract typed entities/relations from chunks with the local LLM (cached, resumable)."""
    settings = load_settings()
    paths = settings.paths
    model = model or settings.llm.extraction_model
    chunks = read_chunks(paths.chunks)
    if chunk_ids_file:
        lines = Path(chunk_ids_file).read_text(encoding="utf-8").splitlines()
        wanted = {line.strip() for line in lines if line.strip()}
        chunks = [c for c in chunks if c.chunk_id in wanted]
    titles = {a.article_id: a.title for a in read_articles(paths.articles)}
    out = output_path(paths.extractions, model)
    if dry_run:
        typer.echo(f"{len(chunks)} chunks in scope, model {model}, output {out}")
        return
    client = build_llm_client(settings)
    try:
        stats = run_extraction(
            client,
            chunks,
            titles,
            model,
            out,
            ontology=load_ontology(),
            num_ctx=settings.llm.num_ctx,
            concurrency=concurrency,
            limit=limit,
        )
    finally:
        client.close()
    if unload and settings.llm.backend == "ollama":
        ollama = OllamaClient(settings.llm.base_url)
        ollama.unload(model)
        ollama.close()
    typer.echo(json.dumps(stats, indent=2))


@app.command()
def build(
    model: str | None = typer.Option(None, help="Extraction model whose output to use."),
    canonicalize: bool = typer.Option(True, help="--no-canonicalize = ablation graph."),
    adjudicate: bool = typer.Option(False, help="LLM adjudication of borderline merges."),
    out_dir: str | None = typer.Option(None, help="Defaults to data/kg (or data/kg_nocanon)."),
) -> None:
    """Resolve entities and write the parquet KG (nodes, edges with provenance, mentions)."""
    from fixgraph.embeddings import SentenceTransformerEmbedder

    settings = load_settings()
    paths = settings.paths
    model = model or settings.llm.extraction_model
    in_corpus = {c.chunk_id for c in read_chunks(paths.chunks)}
    records = [
        r for r in read_records(output_path(paths.extractions, model)) if r.chunk_id in in_corpus
    ]
    if not records:
        raise typer.BadParameter(f"no extractions for {model}; run `fixgraph kg extract` first")
    missing = len(in_corpus) - len({r.chunk_id for r in records if r.ok})
    if missing:
        logger.warning("%d corpus chunks have no successful extraction yet", missing)
    target = Path(out_dir) if out_dir else (paths.kg if canonicalize else paths.root / "kg_nocanon")
    embedder = SentenceTransformerEmbedder()
    client = build_llm_client(settings) if adjudicate else None
    try:
        adj = llm_adjudicator(client, settings.llm.extraction_model) if client else None
        kg, report = build_kg(
            records,
            read_articles(paths.articles),
            load_ontology(),
            embedder,
            adjudicate=adj,
            canonicalize=canonicalize,
        )
    finally:
        embedder.release()
        if client:
            client.close()
    write_kg(kg, target)
    with (target / "merges.jsonl").open("w", encoding="utf-8") as f:
        for m in report.merges:
            f.write(m.model_dump_json() + "\n")
    stats = graph_stats(kg)
    stats["records"] = report.n_records
    stats["failed_records"] = report.n_failed_records
    stats["dropped_mentions"] = dict(report.dropped_mentions.most_common())
    stats["merge_groups"] = len(report.merges)
    (target / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    typer.echo(json.dumps(stats, indent=2))


@app.command()
def stats(kg_dir: str | None = typer.Option(None)) -> None:
    """Print graph statistics for a built KG."""
    paths = load_settings().paths
    typer.echo(json.dumps(graph_stats(read_kg(Path(kg_dir) if kg_dir else paths.kg)), indent=2))


@app.command("eval")
def eval_(
    models: list[str] | None = typer.Option(None, "--model", help="Repeatable; default 4B."),
    gold_file: str | None = typer.Option(None, help="Defaults to data/gold/extraction_gold.jsonl"),
    status: str = typer.Option("all", help="all | reviewed (only human-reviewed gold chunks)."),
    out: str | None = typer.Option(None, help="Also write the JSON report to this file."),
) -> None:
    """Entity/relation P/R/F1 per type against the gold set, per extraction model."""
    paths = load_settings().paths
    gold_path = Path(gold_file) if gold_file else paths.gold / "extraction_gold.jsonl"
    gold = [
        GoldChunk.model_validate_json(line)
        for line in gold_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if status not in ("all", "reviewed"):
        raise typer.BadParameter("--status must be 'all' or 'reviewed'")
    n_reviewed = sum(g.status == "reviewed" for g in gold)
    if status == "reviewed":
        gold = [g for g in gold if g.status == "reviewed"]
    report: dict[str, object] = {
        "gold_chunks": len(gold),
        "gold_status": {"reviewed": n_reviewed, "draft": len(gold) - n_reviewed}
        if status == "all"
        else {"reviewed": len(gold), "draft": 0},
    }
    for model in models or ["qwen3:4b"]:
        recs = {r.chunk_id: r for r in read_records(output_path(paths.extractions, model))}
        missing = [g.chunk_id for g in gold if g.chunk_id not in recs]
        preds = {cid: r.validated for cid, r in recs.items() if r.ok and r.validated}
        ent, rel = evaluate_extractions(preds, gold)
        secs = [recs[g.chunk_id].seconds for g in gold if g.chunk_id in recs]
        report[model] = {
            "missing_predictions": len(missing),
            "seconds_per_chunk": round(sum(secs) / len(secs), 2) if secs else None,
            "entities": [prf.row(k) for k, prf in sorted(ent.items())],
            "relations": [prf.row(k) for k, prf in sorted(rel.items())],
        }
    text = json.dumps(report, indent=2)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n", encoding="utf-8")
    typer.echo(text)


@app.command("gold-sample")
def gold_sample(n: int = typer.Option(50), seed: int = typer.Option(13)) -> None:
    """Pick N chunks for the gold set and write their ids to data/gold/gold_chunk_ids.txt."""
    from fixgraph.kg.annotate import sample_gold_chunks

    paths = load_settings().paths
    titles = {a.article_id: a.title for a in read_articles(paths.articles)}
    picked = sample_gold_chunks(read_chunks(paths.chunks), titles, n=n, seed=seed)
    out = paths.gold / "gold_chunk_ids.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(c.chunk_id + "\n" for c in picked), encoding="utf-8")
    typer.echo(f"{len(picked)} chunk ids -> {out}")


@app.command()
def annotate(reviewer: str = typer.Option("developer")) -> None:
    """Review draft gold annotations (keys: a = accept, e = edit in Notepad, s = skip, q = quit)."""
    from fixgraph.kg.annotate import read_gold, review_loop, write_gold

    paths = load_settings().paths
    gold_path = paths.gold / "extraction_gold.jsonl"
    gold = read_gold(gold_path)
    chunks = {c.chunk_id: c for c in read_chunks(paths.chunks)}
    titles = {a.article_id: a.title for a in read_articles(paths.articles)}
    done = review_loop(gold, chunks, titles, lambda g: write_gold(g, gold_path), reviewer)
    remaining = sum(g.status != "reviewed" for g in gold)
    typer.echo(f"reviewed {done} this session; {remaining} drafts remaining")


def _passages(paths: DataPaths) -> dict[str, str]:
    """chunk_id -> passage shown to the verifier (title + heading + text)."""
    titles = {a.article_id: a.title for a in read_articles(paths.articles)}
    chunks = read_chunks(paths.chunks)
    passages = {
        c.chunk_id: f"Article: {titles.get(c.article_id, '')}\nSection: {c.heading}\n{c.text}"
        for c in chunks
    }
    return passages


@app.command("verify-edges")
def verify_edges(
    kg_dir: str | None = typer.Option(None, help="Input KG; defaults to data/kg."),
    out_dir: str | None = typer.Option(None, help="Verified KG; defaults to data/kg_verified."),
    model: str | None = typer.Option(None, help="Defaults to llm.judge_model."),
    concurrency: int = typer.Option(2),
) -> None:
    """Keep only edges a source chunk states (LLM verdict + quote check); write the verified KG
    and per-(edge, chunk) verdicts (D32)."""
    from fixgraph.kg.verify import edge_claims, run_verification, verified_kg, write_verdicts

    settings = load_settings()
    paths = settings.paths
    model = model or settings.llm.judge_model
    src = Path(kg_dir) if kg_dir else paths.kg
    target = Path(out_dir) if out_dir else paths.root / "kg_verified"
    kg = read_kg(src)
    passages = _passages(paths)
    claims = edge_claims(kg)
    client = build_llm_client(settings)
    try:
        verdicts = run_verification(client, claims, passages, model, concurrency)
    finally:
        client.close()
        if settings.llm.backend == "ollama":
            ollama = OllamaClient(settings.llm.base_url)
            ollama.unload(model)
            ollama.close()
    write_verdicts(verdicts, target / "edge_verdicts.jsonl")
    vkg = verified_kg(kg, verdicts)
    write_kg(vkg, target)
    stats = graph_stats(vkg)
    stats["verification"] = {
        "model": model,
        "claims": len(verdicts),
        "claims_supported": sum(v.supported for v in verdicts),
        "claims_llm_yes_quote_missing": sum(v.llm_supported and not v.supported for v in verdicts),
        "claims_failed_calls": sum(v.error is not None for v in verdicts),
    }
    (target / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    typer.echo(json.dumps(stats["verification"], indent=2))


@app.command("edge-sample")
def edge_sample_cmd(
    n: int = typer.Option(200), min_per: int = typer.Option(5), seed: int = typer.Option(13)
) -> None:
    """Stratified random sample of extracted edges for blind support labelling (D32)."""
    from fixgraph.kg.verify import edge_sample, write_jsonl

    paths = load_settings().paths
    sample = edge_sample(read_kg(paths.kg), n=n, min_per=min_per, seed=seed)
    out = paths.gold / "edge_sample.jsonl"
    write_jsonl(sample, out)
    typer.echo(f"{len(sample)} edges -> {out}")


@app.command("edge-eval")
def edge_eval(
    labels_file: str | None = typer.Option(None, help="Defaults to data/gold/edge_labels.jsonl"),
    verified_dir: str | None = typer.Option(None, help="Defaults to data/kg_verified."),
    out: str = typer.Option("results/kg/edge_verification.json"),
) -> None:
    """Hallucinated-edge rate before/after verification and verifier precision/recall."""
    from fixgraph.kg.verify import (
        EdgeLabel,
        SampledEdge,
        evaluate_edges,
        read_jsonl,
        read_verdicts,
        stratum_sizes,
    )

    paths = load_settings().paths
    vdir = Path(verified_dir) if verified_dir else paths.root / "kg_verified"
    sample = read_jsonl(paths.gold / "edge_sample.jsonl", SampledEdge)
    labels = read_jsonl(
        Path(labels_file) if labels_file else paths.gold / "edge_labels.jsonl", EdgeLabel
    )
    kept = stratum_sizes(read_kg(vdir))
    report = evaluate_edges(
        sample,
        labels,
        read_verdicts(vdir / "edge_verdicts.jsonl"),
        stratum_sizes(read_kg(paths.kg)),
        kept,
    )
    report["labelers"] = sorted({x.labeler for x in labels})
    text = json.dumps(report, indent=2)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text + "\n", encoding="utf-8")
    typer.echo(text)
