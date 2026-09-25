import numpy as np
from scipy.sparse import csr_matrix

from fixgraph.bench.bridge import (
    BridgeCandidate,
    BridgePair,
    anchor_leak,
    check_pair,
    select_candidates,
)
from fixgraph.ingest.links import Link, extract_links
from fixgraph.retrieval.graph import GraphIndex, add_document_layer

HTML = """<html><body><div id="content">
<p class="gb-paragraph">Update your Mac. If your computer doesn't recognize your device,
<a href="/en-us/118106">learn how to use recovery mode</a>.</p>
<ul class="gb-list"><li><a href="#top">Top</a></li><li>See <a href="https://support.apple.com/en-us/999999">other</a>.</li></ul>
</div></body></html>"""


def _link(src: str, dst: str, sentence: str, chunk: str = "c") -> Link:
    return Link(src_article=src, dst_article=dst, anchor="recovery mode", sentence=sentence,
                src_chunk_id=chunk)  # fmt: skip


def test_extract_links_keeps_corpus_links_with_their_sentence() -> None:
    (link,) = extract_links(HTML, "1", {"118106"})  # 999999 is not in the corpus
    assert link.dst_article == "118106" and link.anchor == "learn how to use recovery mode"
    assert link.sentence.startswith("If your computer doesn't recognize")


def test_candidates_are_conditional_deduplicated_capped_and_seeded() -> None:
    links = [
        _link("A", "C1", "If it fails, use recovery mode."),
        _link("A", "C1", "When stuck, use recovery mode."),  # same pair: dropped
        _link("A", "C2", "If X, see this."),
        _link("A", "C3", "If Y, see this."),  # third link from A: capped
        _link("B", "C1", "Back up your device."),  # not conditional
        _link("B", "C2", "If Z, see this.", chunk=""),  # not in a chunk
    ]
    chunks = {"C1": ["C1:0:0"], "C2": ["C2:0:0"], "C3": ["C3:0:0"]}
    got = select_candidates(links, {}, chunks, seed=1)
    assert len(got) == 2 and {c.link.src_article for c in got} == {"A"}
    assert got == select_candidates(links, {}, chunks, seed=1)


def test_anchor_leak_ignores_generic_and_source_title_words() -> None:
    leaked = anchor_leak("Should I use recovery mode?", "learn how to use recovery mode", "")
    assert "recovery" in leaked and len(leaked) == 2  # "use" / "learn" / "how" are generic
    assert not anchor_leak("My update is stuck, what now?", "learn how to use recovery mode", "")
    assert not anchor_leak("Backup failed", "back up your iPhone", "How to back up your iPhone")


def _cand() -> BridgeCandidate:
    return BridgeCandidate(
        link=_link("A", "C", "If it is stuck, use recovery mode.", chunk="A:0:0"),
        source_title="If your iPhone won't update",
        target_title="If you can't update or restore",
        target_chunk_ids=["C:0:0", "C:1:0"],
    )


TEXT = {"A:0:0": "stuck", "C:0:0": "Intro.", "C:1:0": "Press and hold the side button."}


def test_check_pair_builds_a_matched_pair_with_the_quoted_chunk_as_gold() -> None:
    pair = BridgePair(bridge_question="My iPhone update froze on the logo, what do I do?",
                 direct_question="How do I put my iPhone in recovery mode?",
                 answer="Press and hold the side button.", key_facts=["hold side button"],
                 answer_quote="Press and hold the side button.")  # fmt: skip
    r = check_pair(pair, _cand(), TEXT, "b001")
    assert r.ok
    a, d = r.questions
    assert (a.qid, a.qtype, a.gold_chunk_ids) == ("b001a", "bridge", ["A:0:0", "C:1:0"])
    assert (d.qid, d.qtype, d.gold_chunk_ids) == ("b001d", "bridge_direct", ["C:1:0"])


def test_check_pair_rejects_named_bridge_and_missing_quote() -> None:
    def pair(bridge: str, quote: str) -> BridgePair:
        answer = "Press and hold the side button."
        return BridgePair(bridge_question=bridge, direct_question="d", answer=answer, key_facts=[],
                     answer_quote=quote)  # fmt: skip

    named = pair("Should I use recovery mode?", "Press and hold the side button.")
    assert "names the link" in check_pair(named, _cand(), TEXT, "x").reason
    invented = pair("It froze.", "Unplug everything.")
    assert check_pair(invented, _cand(), TEXT, "x").reason == "answer quote not found in C"


def test_document_layer_links_articles_and_maps_them_to_their_chunks() -> None:
    adj = csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    g = GraphIndex(node_ids=["n0", "n1"], labels=["Symptom", "Fix"], texts=["s", "f"], adj=adj,
                   index={"n0": 0, "n1": 1}, node_chunks={0: {"A:0:0"}, 1: {"C:0:0"}},
                   chunk_nodes={"A:0:0": {0}, "C:0:0": {1}}, edges=[])  # fmt: skip
    d = add_document_layer(g, [("A", "C", "A:0:0")], lambda c: c.split(":")[0])
    a, c = d.index["doc:A"], d.index["doc:C"]
    assert d.n == 4 and d.adj[a, c] == 0.8 and d.adj[0, a] == 0.2
    assert d.node_chunks[c] == {"C:0:0"} and d.edges[-1][1:3] == ("LINKS_TO", c)
    assert g.n == 2  # original untouched


def test_bridge_report_hop_cost_and_crossover() -> None:
    from fixgraph.bench.bridge import bridge_report
    from fixgraph.bench.schema import Question

    qs, ranked, corr = [], {}, {}
    for i in range(6):
        p = f"b{i:03d}"
        qs += [
            Question(qid=f"{p}a", question="q", qtype="bridge", gold_chunk_ids=[f"A{i}", f"C{i}"]),
            Question(qid=f"{p}d", question="q", qtype="bridge_direct", gold_chunk_ids=[f"C{i}"]),
        ]
        ranked[(f"{p}d", "S1")] = [f"C{i}"]  # S1 finds C when asked directly...
        ranked[(f"{p}a", "S1")] = ["x"]  # ...but not across the bridge
        ranked[(f"{p}d", "S2L")] = [f"C{i}"]
        ranked[(f"{p}a", "S2L")] = [f"C{i}"]  # S2L crosses it
        for side in "ad":
            corr[(f"{p}{side}", "S1")] = corr[(f"{p}{side}", "S2L")] = 1.0
    rep = bridge_report(qs, ranked, corr)
    hit = rep["answer_chunk_hit@8"]
    assert rep["n_pairs"] == 6
    assert (
        hit["bridge"]["mean"]["S1"]["mean"] == 0.0 and hit["bridge"]["mean"]["S2L"]["mean"] == 1.0
    )
    assert (
        hit["hop_cost"]["mean"]["S1"]["mean"] == 1.0
        and hit["hop_cost"]["mean"]["S2L"]["mean"] == 0.0
    )
    (cross,) = hit["hop_cost"]["crossover_tests_vs_S1"]
    assert cross["system"] == "S2L" and cross["diff"] == -1.0
