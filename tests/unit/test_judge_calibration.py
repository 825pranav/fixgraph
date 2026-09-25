import json

from fixgraph.bench.judge import (
    SYSTEM_JUDGE,
    JudgeExample,
    JudgeOutput,
    judge_answer,
    judge_request,
)
from fixgraph.bench.schema import Question
from fixgraph.bench.stats import weighted_kappa
from fixgraph.bench.validate import HumanLabel, pick_examples, split_labels
from fixgraph.llm.base import ChatMessage, LLMRequest
from fixgraph.llm.fake import FakeLLMClient

Q = Question(qid="q", question="AirPods won't pair?", qtype="single_hop", gold_answer="Reset them.",
             key_facts=["reset AirPods", "hold button 15 s"])  # fmt: skip


def test_v1_request_is_byte_identical_to_the_original_judge() -> None:
    """Earlier cached judgements must stay valid."""
    original = LLMRequest(
        model="m",
        messages=[
            ChatMessage(role="system", content=SYSTEM_JUDGE),
            ChatMessage(
                role="user",
                content="Question: AirPods won't pair?\nReference answer: Reset them.\n"
                "Key facts:\n0. reset AirPods\n1. hold button 15 s\n\nAnswer to grade:\nReset.\n\n"
                "Return score and key_facts_covered with exactly 2 booleans.",
            ),
        ],
        temperature=0.0,
        num_ctx=4096,
        max_tokens=150,
        think=False,
        json_schema=JudgeOutput.model_json_schema(),
    )
    assert judge_request(Q, "Reset.", "m") == original


def test_v3_score_is_derived_by_code() -> None:
    reply = {"key_facts_covered": [True], "main_solution": "partly", "contradicts_reference": False}
    client = FakeLLMClient(scripted=[json.dumps(reply)])
    out = judge_answer(client, Q, "Reset them.", False, "m", "v3")
    assert out.score == 0.5 and out.key_facts_covered == [True, False]
    reply["contradicts_reference"] = True
    client = FakeLLMClient(scripted=[json.dumps(reply)])
    assert judge_answer(client, Q, "x", False, "m", "v3").score == 0.0


def test_v4_puts_examples_before_the_item() -> None:
    ex = JudgeExample(question="Q?", reference="R", key_facts=["k"], answer="A", score=0.5)
    req = judge_request(Q, "Reset.", "m", "v4", [ex])
    assert [m.role for m in req.messages] == ["system", "user", "assistant", "user"]
    assert req.messages[2].content == '{"score": 0.5}'


def test_weighted_kappa_penalises_two_step_disagreement_more() -> None:
    a = [0.0, 0.5, 1.0, 1.0]
    assert weighted_kappa(a, a) == 1.0
    one_step = weighted_kappa(a, [0.0, 0.5, 0.5, 1.0])
    two_step = weighted_kappa(a, [0.0, 0.5, 0.0, 1.0])
    assert one_step > two_step


def test_split_is_seeded_disjoint_and_covers_every_system() -> None:
    labels = [HumanLabel(qid=f"q{i}", system=s, score=float(i % 3) / 2, labeler="t")
              for i in range(12) for s in ("S1", "S2", "S3")]  # fmt: skip
    a, b = split_labels(labels, seed=5), split_labels(labels, seed=5)
    assert a == b and not set(a.dev) & set(a.heldout)
    assert len(a.dev) + len(a.heldout) == len(labels)
    assert {s for _, s in a.dev} == {"S1", "S2", "S3"}
    ex = pick_examples(labels, a.dev)
    assert set(ex) <= set(a.dev) and len(ex) == 3
