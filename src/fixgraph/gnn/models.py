"""Link-prediction models (spec §9.3).

Baselines: cosine similarity of text embeddings (no graph), Adamic-Adar on the undirected
homogeneous graph, DistMult (plain PyTorch, all relations). Main: to_hetero(GraphSAGE, 2 layers)
with a dot-product decoder, trained full-batch with BCE on sampled negative Fix nodes.

Every scorer returns a [len(symptoms), num_fix] score matrix so evaluation is shared.

Used by: gnn/cli.py (`gnn train | gaps`); scored by gnn.evaluate.
Uses: gnn.data (TARGET edge type).
"""

# Imports: scipy sparse for Adamic-Adar, torch and PyG for DistMult and GraphSAGE.
import logging

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix, diags
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero

from fixgraph.gnn.data import TARGET

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


# Baseline 1 (no graph): cosine similarity between symptom and fix text embeddings.
def cosine_scores(data: HeteroData, symptoms: Tensor) -> Tensor:
    s = nn.functional.normalize(data["Symptom"].x[symptoms], dim=-1)
    f = nn.functional.normalize(data["Fix"].x, dim=-1)
    return s @ f.t()


# Give each node type a starting offset so all nodes fit in one global index.
def _offsets(data: HeteroData) -> tuple[dict[str, int], int]:
    offsets: dict[str, int] = {}
    total = 0
    for nt in data.node_types:
        offsets[nt] = total
        total += int(data[nt].num_nodes or 0)
    return offsets, total


# Flatten the typed graph into one undirected 0/1 adjacency matrix for Adamic-Adar.
def _homogeneous_adjacency(data: HeteroData) -> tuple[csr_matrix, dict[str, int]]:
    offsets, n = _offsets(data)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    # Add every edge in both directions using global node indices.
    for src, _rel, dst in data.edge_types:
        ei = data[(src, _rel, dst)].edge_index.cpu().numpy()
        rows += [ei[0] + offsets[src], ei[1] + offsets[dst]]
        cols += [ei[1] + offsets[dst], ei[0] + offsets[src]]
    r = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    c = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    # Build the sparse matrix, drop self-loops, and make all weights 1.
    adj = csr_matrix(coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)))
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:] = 1.0  # binary
    return adj, offsets


# Baseline 2: Adamic-Adar, which rewards symptom/fix pairs sharing rare (low-degree) neighbours.
def adamic_adar_scores(data: HeteroData, symptoms: Tensor) -> Tensor:
    """AA(u, v) = sum over common neighbours w of 1 / log(deg w), via sparse A W A."""
    # Weight each node by 1 / log(degree); nodes with degree <= 1 get weight 0.
    adj, offsets = _homogeneous_adjacency(data)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    w = np.zeros_like(deg)
    w[deg > 1] = 1.0 / np.log(deg[deg > 1])
    # Score = symptom rows x neighbour weights x fix columns, as one sparse product.
    s_rows = symptoms.cpu().numpy() + offsets["Symptom"]
    n_fix = int(data["Fix"].num_nodes or 0)
    f_cols = np.arange(n_fix) + offsets["Fix"]
    scores = csr_matrix(adj[s_rows] @ diags(w) @ adj[:, f_cols])
    return torch.from_numpy(np.asarray(scores.todense(), dtype=np.float32))


# Baseline 3: DistMult, which learns one vector per node and per relation type.
class DistMult(nn.Module):
    def __init__(self, num_nodes: int, num_rels: int, dim: int = 64) -> None:
        super().__init__()
        self.ent = nn.Embedding(num_nodes, dim)
        self.rel = nn.Embedding(num_rels, dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)

    # Score of a triple (head, relation, tail) = sum of the three vectors multiplied element-wise.
    def forward(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        return (self.ent(h) * self.rel(r) * self.ent(t)).sum(-1)


# Wraps a trained DistMult so it can score symptoms against all fixes like the other models.
class TrainedDistMult:
    def __init__(
        self, model: DistMult, offsets: dict[str, int], rel_ids: dict[tuple[str, str, str], int]
    ) -> None:
        self.model, self.offsets, self.rel_ids = model, offsets, rel_ids

    # Score the given symptoms against every Fix via the resolved_by relation vector.
    @torch.no_grad()
    def scores(self, data: HeteroData, symptoms: Tensor) -> Tensor:
        self.model.eval()
        dev = self.model.ent.weight.device
        n_fix = int(data["Fix"].num_nodes or 0)
        h = self.model.ent(symptoms.to(dev) + self.offsets["Symptom"])
        r = self.model.rel(torch.tensor(self.rel_ids[TARGET], device=dev))
        t = self.model.ent(torch.arange(n_fix, device=dev) + self.offsets["Fix"])
        return ((h * r) @ t.t()).cpu()


# Train DistMult on every edge in the graph, using random tails as negatives.
def train_distmult(
    data: HeteroData,
    epochs: int = 200,
    dim: int = 64,
    lr: float = 0.01,
    seed: int = 0,
    device: str = "cpu",
) -> TrainedDistMult:
    torch.manual_seed(seed)
    # Give each relation type an id and collect all edges as (head, relation, tail) in global ids.
    offsets, n = _offsets(data)
    edge_types = sorted(set(data.edge_types) | {TARGET})
    rel_ids = {et: i for i, et in enumerate(edge_types)}
    hs, rs, ts = [], [], []
    for et in data.edge_types:
        ei = data[et].edge_index
        hs.append(ei[0] + offsets[et[0]])
        ts.append(ei[1] + offsets[et[2]])
        rs.append(torch.full((ei.size(1),), rel_ids[et], dtype=torch.long))
    # Build the model; if there are no edges, return it untrained.
    model = DistMult(max(n, 1), len(rel_ids), dim).to(device)
    if not hs or sum(len(x) for x in hs) == 0:
        return TrainedDistMult(model, offsets, rel_ids)
    h, r, t = (torch.cat(x).to(device) for x in (hs, rs, ts))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    # Training loop: real triples labelled 1, random-tail triples labelled 0, BCE loss.
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        neg_t = torch.randint(0, n, t.shape, device=device)
        logits = torch.cat([model(h, r, t), model(h, r, neg_t)])
        labels = torch.cat(
            [torch.ones_like(t, dtype=torch.float), torch.zeros_like(t, dtype=torch.float)]
        )
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()
    return TrainedDistMult(model, offsets, rel_ids)


# ---------------------------------------------------------------------------
# Heterogeneous GraphSAGE
# ---------------------------------------------------------------------------


# Two-layer GraphSAGE used per node/edge type after to_hetero converts it.
class _SAGE(nn.Module):
    def __init__(self, hidden: int, out: int) -> None:
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden)
        self.conv2 = SAGEConv((-1, -1), out)

    # Two rounds of neighbour aggregation with a ReLU in between.
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.conv2(self.conv1(x, edge_index).relu(), edge_index)


# Main model: heterogeneous GraphSAGE encoder plus a text-similarity skip term.
class HeteroSAGE(nn.Module):
    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden: int = 128,
        out: int = 64,
    ) -> None:
        super().__init__()
        # to_hetero copies the SAGE layers for every node and edge type in the graph.
        self.encoder = to_hetero(_SAGE(hidden, out), metadata, aggr="sum")
        # Skip term: learnable weight on the cosine of the input text features, so structure
        # is learned as a correction on top of text similarity (DECISIONS.md D24).
        self.text_weight = nn.Parameter(torch.tensor(5.0))
        self._x: dict[str, Tensor] = {}

    # Encode all nodes; also cache the normalised Symptom and Fix text features for decode.
    def forward(self, data: HeteroData) -> dict[str, Tensor]:
        self._x = {t: nn.functional.normalize(data[t].x, dim=-1) for t in ("Symptom", "Fix")}
        return self.encoder(data.x_dict, data.edge_index_dict)

    # Edge score = dot product of learned embeddings + weighted cosine of the text features.
    def decode(self, z: dict[str, Tensor], symptoms: Tensor, fixes: Tensor) -> Tensor:
        structural = (z["Symptom"][symptoms] * z["Fix"][fixes]).sum(-1)
        text = (self._x["Symptom"][symptoms] * self._x["Fix"][fixes]).sum(-1)
        return structural + self.text_weight * text

    # Score symptoms against every Fix, same formula as decode but as a full matrix.
    @torch.no_grad()
    def scores(self, data: HeteroData, symptoms: Tensor) -> Tensor:
        self.eval()
        dev = next(self.parameters()).device
        z = self(data.to(dev))
        s = symptoms.to(dev)
        structural = z["Symptom"][s] @ z["Fix"].t()
        text = self._x["Symptom"][s] @ self._x["Fix"].t()
        return (structural + self.text_weight * text).cpu()


# Train GraphSAGE: message edges as input, separate supervision edges as positives.
def train_sage(
    message: HeteroData,
    train_pos: Tensor,
    epochs: int = 100,
    lr: float = 0.005,
    seed: int = 0,
    device: str = "cpu",
) -> HeteroSAGE:
    """Full-batch training: supervision edges `train_pos` are NOT in `message` (disjoint)."""
    torch.manual_seed(seed)
    model = HeteroSAGE(message.metadata()).to(device)
    msg = message.to(device)
    # Run once so PyG's lazy layers know their input sizes before the optimizer is built.
    with torch.no_grad():
        model(msg)  # materialize lazy layers before building the optimizer
    if train_pos.size(1) == 0:
        return model
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    pos = train_pos.to(device)
    n_fix = int(msg["Fix"].num_nodes or 0)
    # Each epoch: encode, score positives and random-Fix negatives, BCE loss, one Adam step.
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        z = model(msg)
        neg = torch.randint(0, n_fix, (pos.size(1),), device=device)
        logits = torch.cat([model.decode(z, pos[0], pos[1]), model.decode(z, pos[0], neg)])
        labels = torch.cat([torch.ones(pos.size(1)), torch.zeros(pos.size(1))]).to(device)
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()
        if epoch % 25 == 0:
            logger.debug("sage epoch %d loss %.4f", epoch, loss.item())
    return model
