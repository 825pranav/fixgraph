"""Correctness judge (spec §11.3): rubric score 0 / 0.5 / 1 against the gold answer + key-fact
coverage. Run with the judge model (qwen3:8b); validated against blind reference labels (§11.4).

Prompt variants (DECISIONS.md D33), fixed before calibration; `v1` is the original judge and its
request is unchanged, so earlier cached judgements stay valid:
- v1  three-line rubric.
- v2  tighter rubric: defines "main solution" and what does not lower a score.
- v3  v2 wording, but the model reports the main solution as yes / partly / no plus a
      contradiction flag, and code derives the score (no free-form number).
- v4  v2 plus worked examples drawn only from the calibration (dev) labels.

Used by: bench/run.py (judge stage), bench/cli.py (`bench judge-calibrate`).
Uses: bench.schema.Question, llm.structured.complete_structured.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from fixgraph.bench.schema import Question
from fixgraph.llm.base import ChatMessage, LLMClient, LLMRequest
from fixgraph.llm.structured import StructuredOutputError, complete_structured

JudgeVariant = Literal["v1", "v2", "v3", "v4"]
VARIANTS: tuple[JudgeVariant, ...] = ("v1", "v2", "v3", "v4")


class JudgeOutput(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="0, 0.5 or 1")
    key_facts_covered: list[bool] = Field(default_factory=lambda: list[bool]())


class _Derived(BaseModel):
    """v3 output: the model classifies, code scores."""

    key_facts_covered: list[bool] = Field(default_factory=lambda: list[bool]())
    main_solution: Literal["yes", "partly", "no"]
    contradicts_reference: bool


class JudgeExample(BaseModel):
    """A labelled calibration item used as a worked example (v4)."""

    question: str
    reference: str
    key_facts: list[str]
    answer: str
    score: float


SYSTEM_JUDGE = """You grade answers to Apple troubleshooting questions against a reference.
score 1: the answer gives the reference's main solution/explanation and nothing contradicting it.
score 0.5: partially correct (some key points, or correct but missing the main point).
score 0: wrong, irrelevant, contradicts the reference, or refuses although the reference answers.
Extra correct details are fine. For each key fact, say whether the answer covers it."""

RUBRIC_V2 = """You grade an answer to an Apple troubleshooting question against a reference answer.
First identify the reference's main solution: the action(s) it tells the user to take, or its
central explanation.
score 1: the answer tells the user to do the main solution, in any wording, even as one step
  among several or phrased as a check, and nothing in it contradicts the reference. Extra
  steps, extra detail, a different order, or missing minor details (exact menu paths, light
  colours, timings) do NOT lower the score.
score 0.5: the answer gives only part of the main solution, e.g. one of two required actions,
  or the right action for only one of two problems the question asks about.
score 0: the main solution is missing, the answer contradicts the reference, is off-topic, or
  refuses/abstains although the reference answers.
For each key fact, say whether the answer covers it."""

RUBRIC_V3 = """You grade an answer to an Apple troubleshooting question against a reference answer.
First identify the reference's main solution: the action(s) it tells the user to take, or its
central explanation.
main_solution:
  yes    the answer tells the user to do the main solution, in any wording, even as one step
         among several or phrased as a check. Extra steps, extra detail, a different order or
         missing minor details (exact menu paths, light colours, timings) still count as yes.
  partly the answer gives only part of it, e.g. one of two required actions, or the right
         action for only one of two problems the question asks about.
  no     the main solution is missing, or the answer is off-topic or refuses.
contradicts_reference: true only if the answer tells the user something the reference says is
  wrong or the opposite of the reference.
For each key fact, say whether the answer covers it."""

_DERIVED_SCORE = {"yes": 1.0, "partly": 0.5, "no": 0.0}


def _user(q: Question, answer_text: str) -> str:
    facts = "\n".join(f"{i}. {f}" for i, f in enumerate(q.key_facts)) or "(none)"
    return (
        f"Question: {q.question}\nReference answer: {q.gold_answer}\n"
        f"Key facts:\n{facts}\n\nAnswer to grade:\n{answer_text or '(no answer)'}\n\n"
    )


def _example_messages(examples: Sequence[JudgeExample]) -> list[ChatMessage]:
    out: list[ChatMessage] = []
    for ex in examples:
        q = Question(
            qid="example",
            question=ex.question,
            qtype="single_hop",
            gold_answer=ex.reference,
            key_facts=ex.key_facts,
        )
        out.append(
            ChatMessage(
                role="user",
                content=_user(q, ex.answer) + "Return score and key_facts_covered.",
            )
        )
        out.append(ChatMessage(role="assistant", content=f'{{"score": {ex.score}}}'))
    return out


def judge_request(
    q: Question,
    answer_text: str,
    model: str,
    variant: JudgeVariant = "v1",
    examples: Sequence[JudgeExample] = (),
) -> LLMRequest:
    n = len(q.key_facts)
    if variant == "v3":
        system, schema = RUBRIC_V3, _Derived.model_json_schema()
        tail = f"Return main_solution, contradicts_reference and key_facts_covered ({n} booleans)."
    else:
        system = SYSTEM_JUDGE if variant == "v1" else RUBRIC_V2
        schema = JudgeOutput.model_json_schema()
        tail = f"Return score and key_facts_covered with exactly {n} booleans."
    messages = [ChatMessage(role="system", content=system)]
    if variant == "v4":
        messages += _example_messages(examples)
    messages.append(ChatMessage(role="user", content=_user(q, answer_text) + tail))
    return LLMRequest(
        model=model,
        messages=messages,
        temperature=0.0,
        num_ctx=4096 if variant != "v4" else 8192,
        max_tokens=150,
        think=False,
        json_schema=schema,
    )


def judge_answer(
    client: LLMClient,
    q: Question,
    answer_text: str,
    abstained: bool,
    model: str,
    variant: JudgeVariant = "v1",
    examples: Sequence[JudgeExample] = (),
) -> JudgeOutput:
    if not q.answerable:
        return JudgeOutput(score=1.0 if abstained else 0.0)
    if abstained or not answer_text.strip():
        return JudgeOutput(score=0.0, key_facts_covered=[False] * len(q.key_facts))
    request = judge_request(q, answer_text, model, variant, examples)
    try:
        if variant == "v3":
            d = complete_structured(client, request, _Derived)
            covered_raw = d.key_facts_covered
            score = 0.0 if d.contradicts_reference else _DERIVED_SCORE[d.main_solution]
        else:
            out = complete_structured(client, request, JudgeOutput)
            covered_raw = out.key_facts_covered
            score = min((0.0, 0.5, 1.0), key=lambda s: abs(s - out.score))  # snap to rubric
    except (StructuredOutputError, RuntimeError):
        return JudgeOutput(score=0.0, key_facts_covered=[False] * len(q.key_facts))
    covered = (covered_raw + [False] * len(q.key_facts))[: len(q.key_facts)]
    return JudgeOutput(score=score, key_facts_covered=covered)
