"""Parquet KG storage (spec §7.1): nodes, edges (with provenance), node->chunk mentions."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

NODE_SCHEMA = {
    "node_id": pl.String,
    "label": pl.String,
    "canonical_text": pl.String,
    "props_json": pl.String,
}
EDGE_SCHEMA = {
    "edge_id": pl.String,
    "src": pl.String,
    "rel": pl.String,
    "dst": pl.String,
    "source_chunk_ids": pl.List(pl.String),
    "extraction_model": pl.String,
    "extraction_confidence": pl.Float64,
    "extracted_at": pl.String,
    "origin": pl.String,  # extracted | predicted
    "n_support": pl.Int64,
}
MENTION_SCHEMA = {"node_id": pl.String, "chunk_id": pl.String, "surface": pl.String}


@dataclass
class KG:
    nodes: pl.DataFrame
    edges: pl.DataFrame
    mentions: pl.DataFrame
    _props: dict[str, dict[str, Any]] = field(default_factory=lambda: dict[str, dict[str, Any]]())

    def props(self, node_id: str) -> dict[str, Any]:
        if not self._props:
            self._props = {
                r["node_id"]: json.loads(r["props_json"]) for r in self.nodes.iter_rows(named=True)
            }
        return self._props.get(node_id, {})

    def extracted_edges(self) -> pl.DataFrame:
        return self.edges.filter(pl.col("origin") == "extracted")


def kg_paths(kg_dir: Path) -> tuple[Path, Path, Path]:
    return kg_dir / "nodes.parquet", kg_dir / "edges.parquet", kg_dir / "mentions.parquet"


def write_kg(kg: KG, kg_dir: Path) -> None:
    kg_dir.mkdir(parents=True, exist_ok=True)
    n, e, m = kg_paths(kg_dir)
    kg.nodes.write_parquet(n)
    kg.edges.write_parquet(e)
    kg.mentions.write_parquet(m)


def read_kg(kg_dir: Path) -> KG:
    n, e, m = kg_paths(kg_dir)
    return KG(pl.read_parquet(n), pl.read_parquet(e), pl.read_parquet(m))
