"""Personalized PageRank matches networkx (property-based), incl. dangling nodes and no seeds.

Covers retrieval/ppr.py.
"""

import networkx as nx
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.sparse import csr_matrix

from fixgraph.retrieval.ppr import personalized_pagerank


def _to_nx(adj: np.ndarray) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(range(adj.shape[0]))
    for i, j in zip(*np.nonzero(adj), strict=True):
        g.add_edge(int(i), int(j), weight=float(adj[i, j]))
    return g


@settings(max_examples=60, deadline=None)
@given(
    st.integers(2, 25).flatmap(
        lambda n: st.tuples(
            st.just(n),
            st.lists(
                st.tuples(st.integers(0, n - 1), st.integers(0, n - 1), st.floats(0.1, 3)),
                max_size=60,
            ),
            st.dictionaries(st.integers(0, n - 1), st.floats(0.1, 2), min_size=1, max_size=4),
            st.sampled_from([0.5, 0.85]),
        )
    )
)
def test_matches_networkx(
    case: tuple[int, list[tuple[int, int, float]], dict[int, float], float],
) -> None:
    n, edges, seeds, damping = case
    dense = np.zeros((n, n))
    for i, j, w in edges:
        if i != j:
            dense[i, j] += w
            dense[j, i] += w  # undirected, as used in retrieval
    ours = personalized_pagerank(
        csr_matrix(dense), seeds, damping=damping, tol=1e-12, max_iter=1000
    )
    theirs = nx.pagerank(
        _to_nx(dense),
        alpha=damping,
        personalization=seeds,
        weight="weight",
        tol=1e-12,
        max_iter=1000,
    )
    np.testing.assert_allclose(ours, [theirs[i] for i in range(n)], atol=1e-6)


def test_directed_with_dangling_matches_networkx() -> None:
    dense = np.zeros((4, 4))
    dense[0, 1] = 1.0
    dense[1, 2] = 2.0  # node 2 and 3 are dangling
    seeds = {0: 1.0, 3: 0.5}
    ours = personalized_pagerank(csr_matrix(dense), seeds, damping=0.85, tol=1e-12, max_iter=1000)
    theirs = nx.pagerank(
        _to_nx(dense), alpha=0.85, personalization=seeds, weight="weight", tol=1e-12
    )
    np.testing.assert_allclose(ours, [theirs[i] for i in range(4)], atol=1e-6)


def test_seed_neighbourhood_gets_mass() -> None:
    # path graph 0-1-2-3-4, seed at 0: mass decreases with distance
    dense = np.zeros((5, 5))
    for i in range(4):
        dense[i, i + 1] = dense[i + 1, i] = 1.0
    scores = personalized_pagerank(csr_matrix(dense), {0: 1.0})
    assert list(np.argsort(-scores)) == [0, 1, 2, 3, 4]
    assert np.isclose(scores.sum(), 1.0)


def test_no_seeds_is_plain_pagerank() -> None:
    dense = np.ones((3, 3)) - np.eye(3)
    np.testing.assert_allclose(personalized_pagerank(csr_matrix(dense), {}), [1 / 3] * 3)
