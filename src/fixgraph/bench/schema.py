"""TroubleshootQA question format (spec §11)."""

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


class Question(BaseModel):
    qid: str
    question: str
    qtype: QType
    answerable: bool = True
    gold_answer: str = ""
    key_facts: list[str] = Field(default_factory=lambda: list[str]())
    gold_chunk_ids: list[str] = Field(default_factory=lambda: list[str]())
    gold_seed_nodes: list[str] = Field(default_factory=lambda: list[str]())
    source: str = "handwritten"  # handwritten | generated | generated-informal
    split: Literal["dev", "test"] = "dev"
    verified: bool = False
    verified_by: str = ""


def read_questions(path: Path) -> list[Question]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Question.model_validate_json(x) for x in lines if x.strip()]


def write_questions(questions: list[Question], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(q.model_dump_json() + "\n" for q in questions), encoding="utf-8")
