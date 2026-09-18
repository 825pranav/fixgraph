from fixgraph.bench.combo import CacheRow, Config, grid, rank, recovered_broken, split_questions
from fixgraph.bench.schema import QType, Question


def _q(qid: str, qtype: QType, gold: list[str]) -> Question:
    return Question(qid=qid, question="q", qtype=qtype, gold_chunk_ids=gold)


def test_split_is_seeded_stratified_and_keeps_bridge_pairs_together() -> None:
    qs = [_q(f"t{i:03d}", "single_hop", ["c"]) for i in range(10)]
    for i in range(10):
        qs += [_q(f"b{i:03d}a", "bridge", ["c"]), _q(f"b{i:03d}d", "bridge_direct", ["c"])]
    a = split_questions(qs, seed=3)
    assert a == split_questions(qs, seed=3)
    assert sum(v == "test" for k, v in a.items() if k.startswith("t")) == 4
    assert all(a[f"b{i:03d}a"] == a[f"b{i:03d}d"] for i in range(10))


def _row() -> CacheRow:
    fused = [f"x{i}" for i in range(30)]
    ce = {c: 0.9 - i * 0.01 for i, c in enumerate(fused)}
    ce["L1"] = 0.5  # link-reached, outside S1's pool
    return CacheRow(qid="q", fused=fused, ce=ce, s1=fused[:8], link_pool={"1": ["L1"], "3": ["L1"]},
                    link_reached={"1": ["L1"], "3": ["L1"]}, s1_seconds=0.1,
                    extra_ce_seconds={"1": 0.01, "3": 0.02})  # fmt: skip


def test_rank_variants_return_k_chunks_and_respect_routing() -> None:
    row = _row()
    assert rank(row, Config(kind="s1"), 1.0) == row.s1
    assert rank(row, Config(kind="union", d=1), 1.0) == row.s1  # 0.5 never beats the top 8
    promoted = rank(row, Config(kind="prior", d=1, lam=0.5), 1.0)
    assert promoted[0] == "L1" and len(promoted) == 8
    assert rank(row, Config(kind="prior", d=1, lam=0.5, routing=True), 0.5) == row.s1  # confident
    protected = rank(row, Config(kind="protected", d=1), 1.0)
    assert protected[:6] == row.s1[:6] and "L1" in protected and len(protected) == 8
    assert len(grid()) == 16


def test_recovered_and_broken_counts() -> None:
    qs = [_q("q", "single_hop", ["a", "b"])]
    rb = recovered_broken(qs, {"q": ["a", "x"]}, {"q": ["b", "x"]})
    assert rb == {"single_hop": {"recovered": 1, "broken": 1}}
