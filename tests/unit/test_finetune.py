import numpy as np
import torch

from fixgraph.core.models import Chunk
from fixgraph.retrieval.finetune import listwise_loss, mine_negatives, sample_chunks


def _chunk(art: str, i: int, n_tokens: int = 100) -> Chunk:
    return Chunk(chunk_id=f"{art}:0:{i}", article_id=art, section_idx=0, chunk_idx=i, heading="h",
                 text="t", char_start=0, char_end=1, n_tokens=n_tokens)  # fmt: skip


def test_sample_chunks_excludes_articles_caps_per_article_and_is_seeded() -> None:
    chunks = [_chunk(a, i) for a in ("A", "B", "C") for i in range(5)] + [_chunk("D", 0, 10)]
    got = sample_chunks(chunks, {"C"}, n=10, seed=3)
    assert got == sample_chunks(chunks, {"C"}, n=10, seed=3)
    arts = [c.article_id for c in got]
    assert "C" not in arts and "D" not in arts  # excluded article, too-short chunk
    assert arts.count("A") == 2 and arts.count("B") == 2


def test_mine_negatives_skips_same_article_and_top_matches() -> None:
    doc = np.eye(5, dtype=np.float32)
    q = np.array([5, 4, 3, 2, 1], dtype=np.float32)
    ids = ["p", "x1", "x2", "x3", "x4"]
    arts = ["P", "X", "X", "Y", "Y"]
    assert mine_negatives(q, doc, ids, arts, "P", n=2, skip=1) == ["x2", "x3"]


def test_listwise_loss_prefers_positive_first() -> None:
    good = listwise_loss(torch.tensor([[5.0, 0.0, 0.0]]))
    bad = listwise_loss(torch.tensor([[0.0, 5.0, 0.0]]))
    assert float(good) < float(bad)
