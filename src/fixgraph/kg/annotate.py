"""Gold-set tooling (spec §8.3): sample chunks and review/correct annotations from the terminal.

Annotations live in data/gold/extraction_gold.jsonl (committed; contains short entity strings,
not article text). Each record has `annotator` and `status` (draft | reviewed) so drafts are never
mistaken for reviewed gold.
"""

import random
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from fixgraph.core.models import Chunk
from fixgraph.kg.quality import GoldChunk


def sample_gold_chunks(
    chunks: list[Chunk], titles: dict[str, str], n: int = 50, seed: int = 13
) -> list[Chunk]:
    """Stratified sample: half from troubleshooting articles ("If ..." titles), half from the
    rest, one chunk per article, skipping tiny chunks. Deterministic for a given seed."""
    rng = random.Random(seed)
    by_article: dict[str, list[Chunk]] = {}
    for c in chunks:
        if c.n_tokens >= 60:
            by_article.setdefault(c.article_id, []).append(c)
    trouble = sorted(a for a in by_article if titles.get(a, "").lower().startswith("if "))
    other = sorted(a for a in by_article if a not in set(trouble))
    rng.shuffle(trouble)
    rng.shuffle(other)
    picked = trouble[: n // 2] + other[: n - min(n // 2, len(trouble))]
    return sorted((rng.choice(by_article[a]) for a in picked[:n]), key=lambda c: c.chunk_id)


def read_gold(path: Path) -> list[GoldChunk]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [GoldChunk.model_validate_json(x) for x in lines if x.strip()]


def write_gold(gold: list[GoldChunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(g.model_dump_json() + "\n" for g in sorted(gold, key=lambda g: g.chunk_id))
    path.write_text(body, encoding="utf-8")


def render(g: GoldChunk, chunk: Chunk, title: str) -> str:
    lines = [f"=== {g.chunk_id}  [{g.status} by {g.annotator or '?'}]", f"Article: {title}"]
    lines += ["", chunk.text, "", "Entities:"]
    lines += [f"  {i:2d}. {e.type:10s} {e.text}" for i, e in enumerate(g.entities)]
    lines += ["Relations:"]
    lines += [f"  - {r.head} --{r.rel}--> {r.tail}" for r in g.relations]
    return "\n".join(lines)


def edit_in_notepad(g: GoldChunk) -> GoldChunk:
    """Open the record as pretty JSON in Notepad (blocks until closed), then re-validate."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write(g.model_dump_json(indent=2))
        tmp = Path(f.name)
    try:
        subprocess.run(["notepad.exe", str(tmp)], check=False)
        return GoldChunk.model_validate_json(tmp.read_text(encoding="utf-8"))
    finally:
        tmp.unlink(missing_ok=True)


def review_loop(
    gold: list[GoldChunk],
    chunks: dict[str, Chunk],
    titles: dict[str, str],
    save: Callable[[list[GoldChunk]], None],
    reviewer: str,
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
    editor: Callable[[GoldChunk], GoldChunk] = edit_in_notepad,
) -> int:
    """a=accept, e=edit then accept, s=skip, q=quit. Saves after every change."""
    reviewed = 0
    for i, g in enumerate(gold):
        if g.status == "reviewed":
            continue
        chunk = chunks[g.chunk_id]
        show(render(g, chunk, titles.get(chunk.article_id, "")))
        while True:
            key = ask("[a]ccept  [e]dit  [s]kip  [q]uit > ").strip().lower()[:1]
            if key == "q":
                return reviewed
            if key == "s":
                break
            if key == "e":
                try:
                    g = editor(g)
                except ValueError as exc:  # invalid JSON: show and re-prompt
                    show(f"invalid edit: {exc}")
                    continue
                show(render(g, chunk, titles.get(chunk.article_id, "")))
                continue
            if key == "a":
                annotator = g.annotator if reviewer in g.annotator else f"{g.annotator}+{reviewer}"
                gold[i] = g.model_copy(
                    update={"status": "reviewed", "annotator": annotator.strip("+")}
                )
                save(gold)
                reviewed += 1
                break
    return reviewed
