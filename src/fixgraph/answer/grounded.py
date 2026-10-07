"""Grounded answer generation shared by all systems (spec §10.4).

The answer model returns JSON: a list of sentences, each with the chunk ids it cites, or a
structured abstention. Citations to chunks that were not in the context are stripped, so an
answer can never cite evidence it wasn't shown. Every system uses the same model, prompt and
context token budget; only the retrieved chunks differ.

Used by: bench/run.py (answer stage), api/app.py (`POST /answer`), answer/verifier.py (types).
Uses: llm.structured.complete_structured for the JSON output; ingest.chunk.count_tokens to
enforce the context budget.
"""

# Imports: timing, pydantic for the answer schema, a token counter, and the LLM call helpers.
import time

from pydantic import BaseModel, Field

from fixgraph.ingest.chunk import count_tokens
from fixgraph.llm.base import ChatMessage, LLMClient, LLMRequest
from fixgraph.llm.structured import StructuredOutputError, complete_structured

# Max tokens of chunk text packed into the prompt; the same budget for every system.
CONTEXT_BUDGET_TOKENS = 6000


# One answer sentence plus up to 4 chunk ids it cites. Citations are data, not free text.
class AnswerSentence(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=lambda: list[str](), max_length=4)


# The JSON shape the model must return: either abstain (with a reason) or up to 8 sentences.
class AnswerOutput(BaseModel):
    abstain: bool
    abstain_reason: str | None = None
    sentences: list[AnswerSentence] = Field(
        default_factory=lambda: list[AnswerSentence](), max_length=8
    )


# Final result handed back to the API / bench: cleaned sentences, which chunks were shown,
# timing, and an error string if the model call failed.
class GroundedAnswer(BaseModel):
    question: str
    abstained: bool
    sentences: list[AnswerSentence]
    context_chunk_ids: list[str]
    latency_s: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None

    # Whole answer as one plain string, built by joining the sentences.
    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)

    # All cited chunk ids, de-duplicated but kept in first-seen order.
    @property
    def cited_chunk_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for s in self.sentences:
            for c in s.citations:
                seen.setdefault(c, None)
        return list(seen)


# System prompt for grounded systems: answer only from the numbered sources, cite them, or abstain.
SYSTEM_GROUNDED = """You are an Apple support assistant. Answer the question using ONLY the
numbered sources. Rules:
- Each sentence states one fact and cites the source ids that support it, e.g. ["123:1:0"].
- Never state anything the sources do not support. No outside knowledge.
- If the sources do not contain the answer, set abstain=true and give a short abstain_reason;
  do not guess.
- Be concise: at most 6 sentences, practical steps first."""

# System prompt for S0 (closed book): no sources, so the model answers from memory.
SYSTEM_CLOSED_BOOK = """You are an Apple support assistant. Answer the question from your own
knowledge in at most 6 short sentences. You have no sources, so leave citations empty. If you
do not know, set abstain=true with a short abstain_reason."""


# Pack ranked chunks into the prompt: in rank order, stop once the token budget is reached.
# Input: ranked ids + id->text map. Output: list of (chunk_id, text) pairs.
def build_context(
    chunk_ids: list[str],
    chunk_text: dict[str, str],
    budget_tokens: int = CONTEXT_BUDGET_TOKENS,
) -> list[tuple[str, str]]:
    """Take ranked chunks until the token budget is spent (the last one may be truncated)."""
    out: list[tuple[str, str]] = []
    used = 0
    # Walk chunks best-first, skipping ids we have no text for.
    for cid in chunk_ids:
        text = chunk_text.get(cid)
        if text is None:
            continue
        n = count_tokens(text)
        # Budget would overflow: if enough room is left, add a cut-down version, then stop.
        if used + n > budget_tokens:
            remaining = budget_tokens - used
            if remaining > 80:
                words = text.split(" ")
                out.append((cid, " ".join(words[: int(remaining * 0.7)]) + " ..."))
            break
        # Chunk fits whole, so add it and count its tokens.
        out.append((cid, text))
        used += n
    return out


# Format the context as "[chunk_id]" followed by its text, so the model can cite by id.
def render_sources(context: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"[{cid}]\n{text}" for cid, text in context)


# Build the chat request for the answer model, with the AnswerOutput JSON schema attached.
def build_answer_request(
    question: str,
    context: list[tuple[str, str]],
    model: str,
    num_ctx: int = 8192,
    closed_book: bool = False,
) -> LLMRequest:
    # Closed book (S0): just the question. Otherwise: labelled sources followed by the question.
    if closed_book:
        messages = [
            ChatMessage(role="system", content=SYSTEM_CLOSED_BOOK),
            ChatMessage(role="user", content=f"Question: {question}"),
        ]
    else:
        messages = [
            ChatMessage(role="system", content=SYSTEM_GROUNDED),
            ChatMessage(
                role="user",
                content=f"Sources:\n{render_sources(context)}\n\nQuestion: {question}",
            ),
        ]
    # Deterministic settings (temperature 0, no thinking) and a cap on reply length.
    return LLMRequest(
        model=model,
        messages=messages,
        temperature=0.0,
        num_ctx=num_ctx,
        max_tokens=700,
        think=False,
        json_schema=AnswerOutput.model_json_schema(),
    )


# Main answer entry point used by POST /answer and the bench answer stage.
# Question + ranked chunk ids in -> GroundedAnswer with only valid citations out.
def answer_question(
    client: LLMClient,
    question: str,
    ranked_chunk_ids: list[str],
    chunk_text: dict[str, str],
    model: str,
    num_ctx: int = 8192,
    closed_book: bool = False,
    budget_tokens: int = CONTEXT_BUDGET_TOKENS,
) -> GroundedAnswer:
    # Build the context and remember exactly which chunk ids the model is allowed to cite.
    context = [] if closed_book else build_context(ranked_chunk_ids, chunk_text, budget_tokens)
    allowed = {cid for cid, _ in context}
    request = build_answer_request(question, context, model, num_ctx, closed_book)
    t0 = time.perf_counter()
    # Call the model for schema JSON; if it fails twice or Ollama is down, abstain, don't crash.
    try:
        out = complete_structured(client, request, AnswerOutput)
    except (StructuredOutputError, RuntimeError) as exc:
        return GroundedAnswer(
            question=question,
            abstained=True,
            sentences=[],
            context_chunk_ids=[c for c, _ in context],
            latency_s=time.perf_counter() - t0,
            error=str(exc)[:300],
        )
    # Clean the reply: drop empty sentences and delete any citation id that was not in the context.
    sentences = [
        AnswerSentence(
            text=s.text.strip(),
            citations=[c.strip("[] ") for c in s.citations if c.strip("[] ") in allowed],
        )
        for s in out.sentences
        if s.text.strip()
    ]
    # Treat "model abstained" and "nothing usable left" the same way: return an abstention.
    abstained = out.abstain or not sentences
    return GroundedAnswer(
        question=question,
        abstained=abstained,
        sentences=[] if abstained else sentences,
        context_chunk_ids=[c for c, _ in context],
        latency_s=time.perf_counter() - t0,
    )
