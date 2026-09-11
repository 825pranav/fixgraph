"""TroubleshootQA question format (spec §11): one JSON object per line in data/bench/*.jsonl.

Used by: bench/generate.py (writes questions), bench/screen.py (adds `screen`), bench/run.py and
bench/validate.py (read them), `fixgraph bench *` commands in bench/cli.py.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

QType = Literal[
    "single_hop",
    "cross_device",
    "version_conditional",
    "error_code",
    "multi_constraint",
    "unanswerable",
]


def article_of(chunk_id: str) -> str:
    """Chunk ids are `{article_id}:{section_idx}:{chunk_idx}` (core/models.py)."""
    return chunk_id.split(":", 1)[0]


class ScreenResult(BaseModel):
    """Automatic pre-screen verdict (bench/screen.py). Advisory only: it orders and annotates
    the human review queue, it never sets `Question.verified`."""

    passed: bool
    reason: str
    answerable: bool
    answer_supported: bool
    leaks_answer: bool
    # Verbatim evidence sentence the screen model cited for the answer, and whether it really
    # occurs in the evidence (checked with fuzzy matching, not by the model).
    support_quote: str = ""
    quote_found: bool = False
    # None for single-article questions; False means one article alone answers the question.
    needs_multiple_articles: bool | None = None
    model: str = ""


class Question(BaseModel):
    qid: str
    question: str
    qtype: QType
    answerable: bool = True
    gold_answer: str = ""
    key_facts: list[str] = Field(default_factory=lambda: list[str]())
    gold_chunk_ids: list[str] = Field(default_factory=lambda: list[str]())
    gold_seed_nodes: list[str] = Field(default_factory=lambda: list[str]())
    # KG node ids of the answer (generated questions only); lets the screen detect answer leaks.
    gold_answer_nodes: list[str] = Field(default_factory=lambda: list[str]())
    source: str = "handwritten"  # handwritten | generated | generated-informal
    split: Literal["dev", "test"] = "dev"
    verified: bool = False
    verified_by: str = ""
    screen: ScreenResult | None = None

    @property
    def article_ids(self) -> list[str]:
        """Articles the gold evidence comes from (chunk ids start with the article id)."""
        return sorted({article_of(c) for c in self.gold_chunk_ids})


QuestionFilter = Literal["all", "screened", "verified"]


def select_questions(questions: list[Question], which: QuestionFilter) -> list[Question]:
    """all = everything; screened = auto-screen passed or human-verified; verified = human only.
    Human-rejected questions are already removed from the file by `bench verify`."""
    if which == "verified":
        return [q for q in questions if q.verified]
    if which == "screened":
        return [q for q in questions if q.verified or (q.screen is not None and q.screen.passed)]
    return questions


def read_questions(path: Path) -> list[Question]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Question.model_validate_json(x) for x in lines if x.strip()]


def write_questions(questions: list[Question], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(q.model_dump_json() + "\n" for q in questions), encoding="utf-8")
