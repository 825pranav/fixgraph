"""Post-hoc claim verifier (spec §10.4).

Claims are the answer's sentences (the answer model already writes one fact per sentence).
The judge model checks each claim against the chunks it cites and labels it supported /
partial / unsupported. Unsupported claims are dropped; partial ones are kept but flagged.
Uncited claims are unsupported by definition (no judge call needed).

Used by: bench/run.py (judge stage, feeds the unsupported-claim-rate metric).
Uses: answer.grounded (GroundedAnswer), llm.structured.complete_structured.
"""

from typing import Literal

from pydantic import BaseModel, Field

from fixgraph.answer.grounded import AnswerSentence, GroundedAnswer
from fixgraph.llm.base import ChatMessage, LLMClient, LLMRequest
from fixgraph.llm.structured import StructuredOutputError, complete_structured

Verdict = Literal["supported", "partial", "unsupported"]


class ClaimVerdict(BaseModel):
    claim: int
    verdict: Verdict


class VerifierOutput(BaseModel):
    verdicts: list[ClaimVerdict] = Field(default_factory=lambda: list[ClaimVerdict]())


class VerifiedAnswer(BaseModel):
    verdicts: list[Verdict]  # one per original sentence
    kept: list[AnswerSentence]  # post-verifier answer (unsupported claims dropped)
    abstained: bool
    judge_calls: int = 0
    error: str | None = None


SYSTEM_VERIFIER = """You check whether claims are supported by their cited sources.
For each numbered claim decide:
- supported: the cited sources state it (paraphrase is fine).
- partial: some of it is stated, some is not.
- unsupported: the cited sources do not state it, or contradict it.
Judge only against the sources shown, not your own knowledge."""


def build_verifier_request(
    sentences: list[AnswerSentence], chunk_text: dict[str, str], model: str, num_ctx: int = 8192
) -> LLMRequest:
    cited = sorted({c for s in sentences for c in s.citations})
    sources = "\n\n".join(f"[{c}]\n{chunk_text.get(c, '')}" for c in cited)
    claims = "\n".join(
        f"{i}. {s.text}  (cites: {', '.join(s.citations)})" for i, s in enumerate(sentences)
    )
    return LLMRequest(
        model=model,
        messages=[
            ChatMessage(role="system", content=SYSTEM_VERIFIER),
            ChatMessage(
                role="user",
                content=f"Sources:\n{sources}\n\nClaims:\n{claims}\n\nReturn a verdict for "
                f"every claim number 0..{len(sentences) - 1}.",
            ),
        ],
        temperature=0.0,
        num_ctx=num_ctx,
        max_tokens=400,
        think=False,
        json_schema=VerifierOutput.model_json_schema(),
    )


def verify_answer(
    client: LLMClient, answer: GroundedAnswer, chunk_text: dict[str, str], model: str
) -> VerifiedAnswer:
    if answer.abstained or not answer.sentences:
        return VerifiedAnswer(verdicts=[], kept=[], abstained=True)
    verdicts: list[Verdict] = ["unsupported"] * len(answer.sentences)
    cited_idx = [i for i, s in enumerate(answer.sentences) if s.citations]
    calls = 0
    error = None
    if cited_idx:
        cited_sentences = [answer.sentences[i] for i in cited_idx]
        request = build_verifier_request(cited_sentences, chunk_text, model)
        calls = 1
        try:
            out = complete_structured(client, request, VerifierOutput)
            for v in out.verdicts:
                if 0 <= v.claim < len(cited_idx):
                    verdicts[cited_idx[v.claim]] = v.verdict
        except (StructuredOutputError, RuntimeError) as exc:
            error = str(exc)[:300]  # leave cited claims unjudged -> conservative "unsupported"
    kept = [s for s, v in zip(answer.sentences, verdicts, strict=True) if v != "unsupported"]
    return VerifiedAnswer(
        verdicts=verdicts, kept=kept, abstained=not kept, judge_calls=calls, error=error
    )
