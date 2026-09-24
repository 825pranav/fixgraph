"""Correctness judge (spec §11.3): rubric score 0 / 0.5 / 1 against the gold answer + key-fact
coverage. Run with the judge model (qwen3:8b); validated against human labels (§11.4).

Used by: bench/run.py (judge stage).
Uses: bench.schema.Question, llm.structured.complete_structured.
"""

from pydantic import BaseModel, Field

from fixgraph.bench.schema import Question
from fixgraph.llm.base import ChatMessage, LLMClient, LLMRequest
from fixgraph.llm.structured import StructuredOutputError, complete_structured


class JudgeOutput(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="0, 0.5 or 1")
    key_facts_covered: list[bool] = Field(default_factory=lambda: list[bool]())


SYSTEM_JUDGE = """You grade answers to Apple troubleshooting questions against a reference.
score 1: the answer gives the reference's main solution/explanation and nothing contradicting it.
score 0.5: partially correct (some key points, or correct but missing the main point).
score 0: wrong, irrelevant, contradicts the reference, or refuses although the reference answers.
Extra correct details are fine. For each key fact, say whether the answer covers it."""


def judge_request(q: Question, answer_text: str, model: str) -> LLMRequest:
    facts = "\n".join(f"{i}. {f}" for i, f in enumerate(q.key_facts)) or "(none)"
    return LLMRequest(
        model=model,
        messages=[
            ChatMessage(role="system", content=SYSTEM_JUDGE),
            ChatMessage(
                role="user",
                content=f"Question: {q.question}\nReference answer: {q.gold_answer}\n"
                f"Key facts:\n{facts}\n\nAnswer to grade:\n{answer_text or '(no answer)'}\n\n"
                f"Return score and key_facts_covered with exactly {len(q.key_facts)} booleans.",
            ),
        ],
        temperature=0.0,
        num_ctx=4096,
        max_tokens=150,
        think=False,
        json_schema=JudgeOutput.model_json_schema(),
    )


def judge_answer(
    client: LLMClient, q: Question, answer_text: str, abstained: bool, model: str
) -> JudgeOutput:
    if not q.answerable:
        return JudgeOutput(score=1.0 if abstained else 0.0)
    if abstained or not answer_text.strip():
        return JudgeOutput(score=0.0, key_facts_covered=[False] * len(q.key_facts))
    try:
        out = complete_structured(client, judge_request(q, answer_text, model), JudgeOutput)
    except (StructuredOutputError, RuntimeError):
        return JudgeOutput(score=0.0, key_facts_covered=[False] * len(q.key_facts))
    covered = (out.key_facts_covered + [False] * len(q.key_facts))[: len(q.key_facts)]
    score = min((0.0, 0.5, 1.0), key=lambda s: abs(s - out.score))  # snap to rubric
    return JudgeOutput(score=score, key_facts_covered=covered)
