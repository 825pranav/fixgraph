"""Qdrant local-mode index (spec §4): dense (Qwen3-Embedding) + sparse (BM25) per chunk, and a
dense index over KG node names/aliases for entity linking (M4).

Local mode (`QdrantClient(path=...)`) is embedded; no server. Verified to support named dense +
sparse vectors with the IDF modifier and RRF fusion via prefetch (DECISIONS.md D21).

Used by: `fixgraph index build` (bench/cli.py) to build both collections; retrieval.hybrid and
retrieval.linking query them; bench/cli.py and api/app.py open the client.
Uses: core.models.Chunk, embeddings, retrieval.bm25.
"""

import json
import logging
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from fixgraph.core.models import Chunk
from fixgraph.embeddings import Embedder
from fixgraph.retrieval.bm25 import BM25Encoder

logger = logging.getLogger(__name__)

CHUNKS = "chunks"
NODES = "nodes"


def chunk_document(chunk: Chunk, title: str) -> str:
    """Text that is embedded/indexed for a chunk: article title gives context to later sections."""
    return f"{title}\n{chunk.text}" if not chunk.text.startswith(title) else chunk.text


def open_client(index_dir: Path) -> QdrantClient:
    index_dir.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(index_dir / "qdrant"))


def build_chunk_index(
    client: QdrantClient,
    chunks: list[Chunk],
    titles: dict[str, str],
    embedder: Embedder,
    index_dir: Path,
    batch: int = 256,
) -> None:
    docs = [chunk_document(c, titles.get(c.article_id, "")) for c in chunks]
    bm25 = BM25Encoder().fit(docs)
    (index_dir / "bm25.json").write_text(
        json.dumps({"k1": bm25.k1, "b": bm25.b, "avgdl": bm25.avgdl}), encoding="utf-8"
    )
    if client.collection_exists(CHUNKS):
        client.delete_collection(CHUNKS)
    client.create_collection(
        CHUNKS,
        vectors_config={"dense": qm.VectorParams(size=embedder.dim, distance=qm.Distance.COSINE)},
        sparse_vectors_config={"bm25": qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
    )
    for start in range(0, len(chunks), batch):
        part = chunks[start : start + batch]
        part_docs = docs[start : start + batch]
        dense = embedder.encode(part_docs)
        points = []
        for j, (c, d) in enumerate(zip(part, part_docs, strict=True)):
            ids, vals = bm25.encode_doc(d)
            points.append(
                qm.PointStruct(
                    id=start + j,
                    vector={
                        "dense": dense[j].tolist(),
                        "bm25": qm.SparseVector(indices=ids, values=vals),
                    },
                    payload={"chunk_id": c.chunk_id, "article_id": c.article_id},
                )
            )
        client.upsert(CHUNKS, points)
    logger.info("indexed %d chunks", len(chunks))


def load_bm25(index_dir: Path) -> BM25Encoder:
    cfg = json.loads((index_dir / "bm25.json").read_text(encoding="utf-8"))
    return BM25Encoder(k1=cfg["k1"], b=cfg["b"], avgdl=cfg["avgdl"])


def build_node_index(
    client: QdrantClient,
    node_names: list[tuple[str, str, str]],  # (node_id, label, text) - one row per name/alias
    embedder: Embedder,
    batch: int = 512,
) -> None:
    if client.collection_exists(NODES):
        client.delete_collection(NODES)
    client.create_collection(
        NODES,
        vectors_config={"dense": qm.VectorParams(size=embedder.dim, distance=qm.Distance.COSINE)},
    )
    for start in range(0, len(node_names), batch):
        part = node_names[start : start + batch]
        dense = embedder.encode([t for _, _, t in part])
        client.upsert(
            NODES,
            [
                qm.PointStruct(
                    id=start + j,
                    vector={"dense": dense[j].tolist()},
                    payload={"node_id": nid, "label": label, "text": text},
                )
                for j, (nid, label, text) in enumerate(part)
            ],
        )
    logger.info("indexed %d node names", len(node_names))
