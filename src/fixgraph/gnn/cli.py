"""`fixgraph gnn train | gaps` (spec §9).

train: article-held-out split, then cosine / Adamic-Adar / DistMult / GraphSAGE over several
seeds; writes aggregated filtered-ranking metrics. gaps: trains on all known edges and writes
top-K predicted Symptom->Fix links (gap_candidates.jsonl, predicted_edges.parquet), which
api/app.py serves at /links/suggestions.
Used by: cli.py (mounted as `gnn`).
Uses: kg.store.read_kg, embeddings, gnn.data, gnn.splits, gnn.models, gnn.evaluate.
"""

# Imports: typer for the CLI, polars to write predicted edges, and our gnn modules.
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import torch
import typer

from fixgraph.core.config import load_settings
from fixgraph.gnn.data import GraphData, build_graph_data
from fixgraph.gnn.evaluate import aggregate, evaluate, known_fixes
from fixgraph.gnn.models import adamic_adar_scores, cosine_scores, train_distmult, train_sage
from fixgraph.gnn.splits import article_held_out_split
from fixgraph.kg.store import EDGE_SCHEMA, read_kg

# Typer app mounted as `fixgraph gnn`, and the four models we compare.
logger = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=True, help="GNN link prediction (Symptom -> Fix).")

MODELS = ("cosine", "adamic_adar", "distmult", "hetero_sage")


# Load the KG, embed node text into a PyG graph, and make sure the output folder exists.
def _load_graph() -> tuple[GraphData, Path]:
    from fixgraph.embeddings import SentenceTransformerEmbedder

    paths = load_settings().paths
    kg = read_kg(paths.kg)
    embedder = SentenceTransformerEmbedder()
    try:
        g = build_graph_data(kg, embedder)
    finally:
        embedder.release()
    out = paths.results / "gnn"
    out.mkdir(parents=True, exist_ok=True)
    return g, out


# One seed: make the article-held-out split, train each model, and score them on test edges.
def run_seed(g: GraphData, seed: int, epochs: int, device: str) -> dict[str, dict[str, float]]:
    split = article_held_out_split(g, holdout_frac=0.2, seed=seed)
    known = known_fixes(g.target_edge_index)
    # Baselines and DistMult see all non-test edges; GraphSAGE trains on its disjoint message graph.
    ev = split.eval_message
    dm = train_distmult(ev, epochs=epochs * 2, seed=seed, device=device)
    sage = train_sage(split.message, split.train_pos, epochs=epochs, seed=seed, device=device)
    return {
        "cosine": evaluate(lambda s: cosine_scores(ev, s), split.test_pos, known),
        "adamic_adar": evaluate(lambda s: adamic_adar_scores(ev, s), split.test_pos, known),
        "distmult": evaluate(lambda s: dm.scores(ev, s), split.test_pos, known),
        "hetero_sage": evaluate(lambda s: sage.scores(ev, s), split.test_pos, known),
    }


# Format the aggregated metrics as a Markdown table (mean +- std over seeds).
def results_markdown(agg: dict[str, dict[str, dict[str, float]]], n_test: list[int]) -> str:
    lines = [
        "| Model | MRR | Hits@1 | Hits@3 | Hits@10 | AUROC |",
        "|---|---|---|---|---|---|",
    ]
    for model in MODELS:
        m = agg[model]
        cells = [
            f"{m[k]['mean']:.3f} ± {m[k]['std']:.3f}"
            for k in ("mrr", "hits@1", "hits@3", "hits@10", "auroc")
        ]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        f"Article-held-out split, filtered ranking against all Fix nodes, 3 seeds; "
        f"held-out edges per seed: {n_test}."
    )
    return "\n".join(lines) + "\n"


# CLI `gnn train`: run every model over several seeds and save results.json and results.md.
@app.command()
def train(
    epochs: int = typer.Option(100),
    seeds: int = typer.Option(3),
) -> None:
    """Baselines + hetero GraphSAGE on the article-held-out split; mean ± std over seeds."""
    g, out = _load_graph()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    per_seed: list[dict[str, dict[str, float]]] = []
    n_test: list[int] = []
    # Run each seed and record how many test edges it held out.
    for seed in range(seeds):
        n_test.append(int(article_held_out_split(g, seed=seed).test_pos.size(1)))
        per_seed.append(run_seed(g, seed, epochs, device))
        logger.info("seed %d: %s", seed, json.dumps(per_seed[-1]))
    # Average across seeds, then write the JSON and Markdown reports.
    agg = {m: aggregate([r[m] for r in per_seed]) for m in MODELS}
    report = {
        "split": "article_held_out",
        "target_edges": int(g.target_edge_index.size(1)),
        "num_symptoms": g.num_nodes("Symptom"),
        "num_fixes": g.num_nodes("Fix"),
        "held_out_edges_per_seed": n_test,
        "per_seed": per_seed,
        "aggregate": agg,
    }
    (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = results_markdown(agg, n_test)
    (out / "results.md").write_text(md, encoding="utf-8")
    typer.echo(md)


# CLI `gnn gaps`: train on all known edges and write the top predicted missing Symptom->Fix links.
@app.command()
def gaps(
    top_k: int = typer.Option(50),
    epochs: int = typer.Option(100),
) -> None:
    """Top-K predicted missing Symptom->Fix links from a model trained on all known edges."""
    g, out = _load_graph()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # holdout_frac=0 means no test edges: the model learns from everything we know.
    split = article_held_out_split(g, holdout_frac=0.0, seed=0)
    model = train_sage(split.message, split.train_pos, epochs=epochs, seed=0, device=device)
    known = known_fixes(g.target_edge_index)
    n_sym = g.num_nodes("Symptom")
    candidates: list[tuple[float, int, int]] = []
    # Score symptoms in batches of 512; hide known fixes so only new links are proposed.
    for start in range(0, n_sym, 512):
        syms = torch.arange(start, min(start + 512, n_sym))
        scores = model.scores(split.eval_message, syms)
        for row, s in enumerate(syms.tolist()):
            r = scores[row].clone()
            if known.get(s):
                r[list(known[s])] = float("-inf")
            vals, idx = torch.topk(r, min(top_k, r.numel()))
            candidates += [(float(v), s, int(f)) for v, f in zip(vals, idx, strict=True)]
    # Keep the global top-K by score.
    candidates.sort(key=lambda x: -x[0])
    now = datetime.now(UTC).isoformat(timespec="seconds")
    # Write each candidate to JSONL (for review) and as a "predicted" edge row for the API.
    rows = []
    with (out / "gap_candidates.jsonl").open("w", encoding="utf-8") as f:
        for score, s, fx in candidates[:top_k]:
            rec = {
                "symptom_id": g.node_ids["Symptom"][s],
                "symptom": g.texts["Symptom"][s],
                "fix_id": g.node_ids["Fix"][fx],
                "fix": g.texts["Fix"][fx],
                "score": score,
                "origin": "predicted",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            src, dst = rec["symptom_id"], rec["fix_id"]
            rows.append(
                {
                    "edge_id": hashlib.sha1(f"{src}|RESOLVED_BY|{dst}|pred".encode()).hexdigest()[
                        :16
                    ],
                    "src": src,
                    "rel": "RESOLVED_BY",
                    "dst": dst,
                    "source_chunk_ids": [],
                    "extraction_model": "gnn-hetero-sage",
                    "extraction_confidence": float(torch.sigmoid(torch.tensor(score))),
                    "extracted_at": now,
                    "origin": "predicted",
                    "n_support": 0,
                }
            )
    # Save predicted edges as parquet with the same schema as real KG edges.
    pl.DataFrame(rows, schema=EDGE_SCHEMA).write_parquet(out / "predicted_edges.parquet")
    typer.echo(f"{len(rows)} predicted links -> {out / 'gap_candidates.jsonl'}")
