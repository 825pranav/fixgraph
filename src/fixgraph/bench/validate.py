"""Judge validation (spec §11.4): a human scores a sample of answers with the same 0 / 0.5 / 1
rubric the LLM judge uses; Cohen's kappa measures agreement.

Labels are stored as (qid, system, score) only; answers are re-read from the run's results,
so no generated text is duplicated into the committed labels file.
"""

import json
import random
from collections import defaultdict
from collections.abc import Callable
from itertools import zip_longest
from pathlib import Path

from pydantic import BaseModel

from fixgraph.bench.schema import Question
from fixgraph.bench.stats import cohens_kappa

_KEYS = {"0": 0.0, "5": 0.5, "1": 1.0}


class HumanLabel(BaseModel):
    qid: str
    system: str
    score: float
    labeler: str


def sample_for_labeling(
    keys: list[tuple[str, str]], questions: dict[str, Question], n: int, seed: int = 13
) -> list[tuple[str, str]]:
    """Answerable (qid, system) pairs, balanced across systems (round-robin over per-system
    shuffles) so kappa is not dominated by one retriever. Unanswerable questions are scored
    mechanically (abstained or not), so they need no human label."""
    rng = random.Random(seed)
    by_system: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for k in sorted(k for k in keys if k[0] in questions and questions[k[0]].answerable):
        by_system[k[1]].append(k)
    for pool in by_system.values():
        rng.shuffle(pool)
    out: list[tuple[str, str]] = []
    for batch in zip_longest(*(by_system[s] for s in sorted(by_system))):
        out += [k for k in batch if k is not None]
    return out[:n]


def read_labels(path: Path) -> list[HumanLabel]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [HumanLabel.model_validate_json(x) for x in lines if x.strip()]


def label_loop(
    todo: list[tuple[str, str]],
    questions: dict[str, Question],
    answer_text: dict[tuple[str, str], str],
    existing: list[HumanLabel],
    save: Callable[[list[HumanLabel]], None],
    labeler: str,
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
) -> list[HumanLabel]:
    """Keys: 1 = correct, 5 = partially correct (0.5), 0 = wrong, s = skip, q = quit.
    The system name is hidden so labels are blind to which retriever produced the answer."""
    labels = list(existing)
    done = {(x.qid, x.system) for x in labels}
    for i, key in enumerate(todo):
        if key in done:
            continue
        q = questions[key[0]]
        facts = "\n".join(f"  - {f}" for f in q.key_facts)
        show(
            f"\n[{i + 1}/{len(todo)}] {q.question}\nREFERENCE: {q.gold_answer}\nKEY FACTS:\n{facts}"
            f"\nANSWER:\n{answer_text.get(key) or '(abstained / empty)'}"
        )
        while True:
            k = ask("[1] correct  [5] partial  [0] wrong  [s]kip  [q]uit > ").strip().lower()[:1]
            if k in _KEYS:
                labels.append(
                    HumanLabel(qid=key[0], system=key[1], score=_KEYS[k], labeler=labeler)
                )
                save(labels)
                break
            if k == "s":
                break
            if k == "q":
                return labels
    return labels


class Agreement(BaseModel):
    n: int
    kappa: float
    exact_agreement: float
    judge_mean: float
    human_mean: float


def agreement(labels: list[HumanLabel], judge_scores: dict[tuple[str, str], float]) -> Agreement:
    pairs = [
        (x.score, judge_scores[(x.qid, x.system)])
        for x in labels
        if (x.qid, x.system) in judge_scores
    ]
    if not pairs:
        raise ValueError("no overlapping labels")
    human = [str(h) for h, _ in pairs]
    judge = [str(j) for _, j in pairs]
    return Agreement(
        n=len(pairs),
        kappa=round(cohens_kappa(human, judge), 3),
        exact_agreement=round(sum(h == j for h, j in pairs) / len(pairs), 3),
        judge_mean=round(sum(j for _, j in pairs) / len(pairs), 3),
        human_mean=round(sum(h for h, _ in pairs) / len(pairs), 3),
    )


def write_labels(labels: list[HumanLabel], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(x.model_dump_json() + "\n" for x in labels), encoding="utf-8")


def load_run(run_dir: Path) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], float]]:
    """(qid, system) -> answer text, and -> judge score, from a benchmark run directory."""
    answers: dict[tuple[str, str], str] = {}
    for line in (run_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        a = r["answer"]
        answers[(r["qid"], r["system"])] = (
            "" if a["abstained"] else " ".join(s["text"] for s in a["sentences"])
        )
    judged: dict[tuple[str, str], float] = {}
    for line in (run_dir / "judged.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        judged[(r["qid"], r["system"])] = float(r["judge"]["score"])
    return answers, judged
