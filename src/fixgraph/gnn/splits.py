"""Leakage-free splits for Symptom -> Fix link prediction (spec §9.2).

Article-held-out (headline): a seeded ~20% of articles is held out; target edges whose
provenance comes ONLY from held-out articles are test edges and are removed from message
passing in both directions. Training supervision edges are disjoint from training message
edges (disjoint_train_ratio), so the model never sees a supervised edge in its input graph.

Transductive: PyG RandomLinkSplit with rev_edge_types (secondary).
"""

import random
from dataclasses import dataclass

import torch
import torch_geometric.transforms as T
from torch import Tensor
from torch_geometric.data import HeteroData

from fixgraph.gnn.data import REV_TARGET, TARGET, GraphData, message_graph


@dataclass
class LinkSplit:
    message: HeteroData  # training-time message graph (no supervision or test edges)
    eval_message: HeteroData  # evaluation-time message graph (all non-test edges)
    train_pos: Tensor  # [2, N] supervision edges (Symptom idx, Fix idx)
    test_pos: Tensor  # [2, M] held-out edges
    held_out_articles: frozenset[str]


def article_held_out_split(
    g: GraphData, holdout_frac: float = 0.2, seed: int = 0, supervision_frac: float = 0.3
) -> LinkSplit:
    articles = sorted(set().union(*g.target_articles)) if g.target_articles else []
    rng = random.Random(seed)
    n_held = round(len(articles) * holdout_frac)
    held = frozenset(rng.sample(articles, n_held)) if n_held else frozenset()
    is_test = torch.tensor(
        [bool(a) and a <= held for a in g.target_articles], dtype=torch.bool
    ).reshape(-1)

    train_idx = (~is_test).nonzero().view(-1)
    gen = torch.Generator().manual_seed(seed)
    perm = train_idx[torch.randperm(len(train_idx), generator=gen)]
    if len(perm) <= 1:  # nothing to split: supervise on what exists, no target messages
        sup, msg = perm, perm[:0]
    else:
        n_sup = max(1, int(len(perm) * supervision_frac))
        sup, msg = perm[:n_sup], perm[n_sup:]

    keep_msg = torch.zeros(len(g.target_articles), dtype=torch.bool)
    keep_msg[msg] = True
    ei = g.target_edge_index
    return LinkSplit(
        message=message_graph(g, keep_msg),
        eval_message=message_graph(g, ~is_test),
        train_pos=ei[:, sup],
        test_pos=ei[:, is_test],
        held_out_articles=held,
    )


def transductive_split(g: GraphData, seed: int = 0) -> tuple[HeteroData, HeteroData, HeteroData]:
    """Train/val/test HeteroData from RandomLinkSplit on the target relation."""
    torch.manual_seed(seed)
    data = T.ToUndirected()(g.data.clone())
    split = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.2,
        disjoint_train_ratio=0.3,
        add_negative_train_samples=False,
        edge_types=TARGET,
        rev_edge_types=REV_TARGET,
    )
    train, val, test = split(data)
    return train, val, test
