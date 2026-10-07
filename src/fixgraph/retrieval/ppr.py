"""Personalized PageRank by power iteration on scipy.sparse (spec §10.2).

Matches networkx.pagerank(G, alpha=damping, personalization=seeds, weight="weight") semantics:
dangling nodes redistribute their mass according to the personalization vector.

Used by: retrieval.graphrag (S2). No fixgraph imports.
"""

# Imports: numpy for score vectors and scipy.sparse for the graph adjacency matrix.
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, diags


# Runs Personalized PageRank from seed nodes over the sparse graph; graphrag S2 uses the scores
# to rank nodes and then their chunks. Returns one score per node, summing to 1.
def personalized_pagerank(
    adj: csr_matrix,
    seeds: dict[int, float],
    damping: float = 0.5,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> NDArray[np.float64]:
    """PPR scores (sum to 1). `adj[i, j]` is the weight of edge i->j (symmetric for undirected).

    damping is the probability of following an edge; HippoRAG uses 0.5.
    """
    # Empty graph: nothing to rank.
    n = int(adj.get_shape()[0])
    if n == 0:
        return np.zeros(0)
    # Build the restart vector from the seed weights (the question's linked entities).
    p = np.zeros(n)
    for i, w in seeds.items():
        if w > 0:
            p[i] += w
    # No usable seeds falls back to uniform restart; otherwise normalize seeds to sum to 1.
    if p.sum() == 0:
        p[:] = 1.0 / n  # no seeds: plain PageRank
    else:
        p /= p.sum()
    # Find nodes with no outgoing edges (dangling) and the inverse out-degree for the rest.
    out_w = np.asarray(adj.sum(axis=1)).ravel()
    dangling = out_w == 0
    inv = np.zeros(n)
    inv[~dangling] = 1.0 / out_w[~dangling]
    # Column-stochastic over non-dangling nodes: P^T with P = D^-1 A.
    transition_t = csr_matrix(diags(inv) @ adj).transpose().tocsr()
    # Power iteration: follow an edge with prob. damping, else jump back to the seeds; dangling mass
    # is also sent back to the seeds. Stop once the change is below the tolerance.
    x = p.copy()
    for _ in range(max_iter):
        x_new = damping * (transition_t @ x + x[dangling].sum() * p) + (1 - damping) * p
        if np.abs(x_new - x).sum() < tol * n:
            x = x_new
            break
        x = x_new
    # Renormalize so the final scores sum to 1.
    return x / x.sum()
