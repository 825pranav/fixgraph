"""Automatic pre-screen for generated questions, run before human verification.

Checks, cheapest and most objective first:
1. Answer leak (deterministic): if most content words of the answer node's text already appear
   in the question, the question gives the answer away.
2. Full-evidence check (judge model, qwen3:8b, temperature 0, cached): is the question
   answerable from its gold evidence, and is the gold answer supported? The model must copy the
   supporting sentence verbatim; the code then checks that the quote really occurs in the
   evidence, so a rubber-stamp "supported" without a real quote fails.
3. Single-article check (only for questions whose evidence spans >= 2 articles): can ANY one
   article's evidence answer the question on its own? If yes, the question is not really
   multi-hop and `needs_multiple_articles` is False.

The verdict is advisory. It is stored on `Question.screen`, shown during `fixgraph bench verify`,
and used to order the review queue; only a human sets `Question.verified`. Screened-but-
unverified questions are still reported as "auto-screened", never as "verified".

Used by: `fixgraph bench screen` (bench/cli.py).
Uses: llm/structured.py (schema-constrained output), bench/schema.py (Question, ScreenResult).
"""

import logging
import re
from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel

from fixgraph.bench.schema import Question, ScreenResult, article_of
from fixgraph.kg.validation import partial_ratio
from fixgraph.llm.base import ChatMessage, LLMClient, LLMRequest
from fixgraph.llm.structured import StructuredOutputError, complete_structured

logger = logging.getLogger(__name__)
_MAX_CHUNK_CHARS = 1500
QUOTE_MIN_RATIO = 0.85  # fuzzy match for "the quote occurs in the evidence" (tolerates list joins)
LEAK_MIN_OVERLAP = 0.6  # share of the answer's content words already present in the question
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "my",
        "of",
        "on",
        "or",
        "so",
        "that",
        "the",
        "then",
        "this",
        "to",
        "up",
        "with",
        "you",
        "your",
    }
)


def _stem(word: str) -> str:
    """Crude suffix strip so "updating" / "updated" / "updates" all match "update"."""
    for suffix in ("ing", "ed", "es", "s", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _content_words(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


# Device, OS and app names are the question's topic, not its answer ("Unpair your Apple Watch"
# does not leak because the question mentions an Apple Watch).
_TOPIC_WORDS = frozenset(
    {
        "apple", "watch", "iphone", "ipad", "ipod", "mac", "macbook", "airpods", "pro", "max",
        "vision", "ios", "ipados", "macos", "watchos", "tvos", "visionos", "icloud", "imessage",
        "facetime", "mail", "safari", "music", "app", "apps", "device", "devices", "settings",
    }
)  # fmt: skip


def leak_overlap(question: str, answer_texts: list[str], topic_texts: Sequence[str] = ()) -> float:
    """Max share of an answer text's content words that already appear in the question, after
    dropping topic words (device/app names and words of the question's own seed nodes)."""
    q_words = _content_words(question)
    topic = {_stem(w) for w in _TOPIC_WORDS}.union(*(_content_words(t) for t in topic_texts))
    best = 0.0
    for text in answer_texts:
        words = _content_words(text) - topic
        if words:
            best = max(best, len(words & q_words) / len(words))
    return best


class _EvidenceCheck(BaseModel):
    answerable: bool
    answer_supported: bool
    support_quote: str
    reason: str


class _SingleSourceCheck(BaseModel):
    fully_answerable: bool


_SYSTEM = "You audit benchmark questions for a troubleshooting QA dataset. Be strict and literal."

_EVIDENCE_PROMPT = """Question: {question}
Reference answer: {answer}

Evidence:
{evidence}

Decide:
- answerable: can the question be answered using ONLY the evidence?
- answer_supported: is every main point of the reference answer stated in the evidence? Claims
  or reasons in the reference answer that the evidence does not state make this false.
- support_quote: copy, word for word, the one evidence sentence that states the reference
  answer's main step ("" if there is none).
- reason: one short sentence explaining any problem (or "ok")."""

_SINGLE_PROMPT = """Question: {question}
Reference answer: {answer}

Text:
{evidence}

Can the reference answer be fully confirmed from this text alone, with nothing else? A text
that covers only part of the question (e.g. only one of two problems) is NOT enough."""


def _evidence(chunk_ids: list[str], chunk_text: dict[str, str]) -> str:
    return "\n\n".join(f"[{c}] {chunk_text.get(c, '')[:_MAX_CHUNK_CHARS]}" for c in chunk_ids)


def _request(model: str, prompt: str, schema: type[BaseModel]) -> LLMRequest:
    return LLMRequest(
        model=model,
        messages=[
            ChatMessage(role="system", content=_SYSTEM),
            ChatMessage(role="user", content=prompt),
        ],
        temperature=0.0,
        num_ctx=8192,
        max_tokens=200,
        think=False,
        json_schema=schema.model_json_schema(),
    )


def screen_question(
    client: LLMClient,
    q: Question,
    chunk_text: dict[str, str],
    model: str,
    node_text: dict[str, str] | None = None,
) -> ScreenResult:
    """`node_text` maps KG node ids to text, for the leak check on `gold_answer_nodes`."""
    text_of = node_text or {}
    answer_texts = [text_of.get(n, "") for n in q.gold_answer_nodes]
    seed_texts = [text_of.get(n, "") for n in q.gold_seed_nodes]
    # Only the main answer node: version nodes ("iOS 17") are legitimately named in questions.
    leaks = leak_overlap(q.question, answer_texts[:1], seed_texts) >= LEAK_MIN_OVERLAP
    evidence = _evidence(q.gold_chunk_ids, chunk_text)
    try:
        ev = complete_structured(
            client,
            _request(
                model,
                _EVIDENCE_PROMPT.format(
                    question=q.question,
                    answer=q.gold_answer,
                    evidence=evidence,
                ),
                _EvidenceCheck,
            ),
            _EvidenceCheck,
        )
    except (StructuredOutputError, RuntimeError) as exc:
        logger.warning("screen failed for %s: %s", q.qid, exc)
        return ScreenResult(
            passed=False,
            reason="screen model output invalid",
            answerable=False,
            answer_supported=False,
            leaks_answer=leaks,
            model=model,
        )

    needs_multi: bool | None = None
    if len(q.article_ids) > 1:
        by_article: dict[str, list[str]] = defaultdict(list)
        for c in q.gold_chunk_ids:
            by_article[article_of(c)].append(c)
        needs_multi = True
        for chunks in by_article.values():
            prompt = _SINGLE_PROMPT.format(
                question=q.question, answer=q.gold_answer, evidence=_evidence(chunks, chunk_text)
            )
            try:
                single = complete_structured(
                    client, _request(model, prompt, _SingleSourceCheck), _SingleSourceCheck
                )
            except (StructuredOutputError, RuntimeError):
                continue  # unknown: don't claim single-source
            if single.fully_answerable:
                needs_multi = False
                break

    quote = ev.support_quote.strip()
    quote_found = len(quote) >= 15 and partial_ratio(quote, evidence) >= QUOTE_MIN_RATIO
    passed = ev.answerable and ev.answer_supported and quote_found and not leaks
    problems = [
        msg
        for bad, msg in (
            (leaks, "question names the answer"),
            (not quote_found, "no verbatim supporting quote in evidence"),
        )
        if bad
    ]
    reason = "; ".join([*problems, ev.reason.strip()]) if problems else ev.reason.strip() or "ok"
    return ScreenResult(
        passed=passed,
        reason=reason,
        answerable=ev.answerable,
        answer_supported=ev.answer_supported,
        leaks_answer=leaks,
        support_quote=quote,
        quote_found=quote_found,
        needs_multiple_articles=needs_multi,
        model=model,
    )


def review_order(questions: list[Question]) -> list[Question]:
    """Screen-passed questions first (multi-article before single), flagged ones last, so a
    time-boxed human review spends its time on the most likely keepers."""

    def key(q: Question) -> tuple[int, int, str]:
        passed = q.screen is not None and q.screen.passed
        multi = q.screen is not None and bool(q.screen.needs_multiple_articles)
        return (0 if passed else 1, 0 if multi else 1, q.qid)

    return sorted(questions, key=key)
