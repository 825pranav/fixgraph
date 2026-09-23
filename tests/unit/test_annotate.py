from pathlib import Path

from fixgraph.core.models import Chunk
from fixgraph.kg.annotate import read_gold, review_loop, sample_gold_chunks, write_gold
from fixgraph.kg.quality import GoldChunk, GoldEntity


def _chunk(aid: str, i: int = 0, tokens: int = 100) -> Chunk:
    return Chunk(
        chunk_id=f"{aid}:{i}:0",
        article_id=aid,
        section_idx=i,
        chunk_idx=0,
        heading="H",
        text="If it won't pair, restart it.",
        char_start=0,
        char_end=10,
        n_tokens=tokens,
    )


def test_sample_is_stratified_deterministic_one_per_article() -> None:
    chunks = [_chunk(str(a), i) for a in range(1, 41) for i in range(3)] + [_chunk("99", 0, 5)]
    titles = {str(a): ("If x" if a % 2 else "About x") for a in range(1, 41)} | {"99": "If tiny"}
    s1 = sample_gold_chunks(chunks, titles, n=10, seed=1)
    s2 = sample_gold_chunks(chunks, titles, n=10, seed=1)
    assert s1 == s2 and len(s1) == 10
    assert len({c.article_id for c in s1}) == 10
    assert sum(titles[c.article_id].startswith("If") for c in s1) == 5
    assert all(c.article_id != "99" for c in s1)  # too small


def test_review_loop_accept_edit_skip_quit(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    gold = [
        GoldChunk(chunk_id=f"{a}:0:0", annotator="ai-draft", entities=[]) for a in ("1", "2", "3")
    ]
    write_gold(gold, path)
    keys = iter(["a", "e", "a", "s", "q"])

    def editor(g: GoldChunk) -> GoldChunk:
        return g.model_copy(update={"entities": [GoldEntity(type="Fix", text="restart it")]})

    shown: list[str] = []
    n = review_loop(
        read_gold(path),
        {c.chunk_id: c for c in (_chunk("1"), _chunk("2"), _chunk("3"))},
        {},
        lambda g: write_gold(g, path),
        reviewer="dev",
        ask=lambda _: next(keys),
        show=shown.append,
        editor=editor,
    )
    assert n == 2
    saved = read_gold(path)
    assert [g.status for g in saved] == ["reviewed", "reviewed", "draft"]
    assert saved[0].annotator == "ai-draft+dev"
    assert saved[1].entities[0].text == "restart it"
