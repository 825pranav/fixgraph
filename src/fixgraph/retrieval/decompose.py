"""Query decomposition for multi-problem questions (DECISIONS.md D42).

Multi-article questions name two problems ("my Mac won't turn on and the camera isn't
working"); a single query embedding and a single cross-encoder pass favour whichever problem
dominates the text. The local LLM splits the question into one self-contained search per
problem; each sub-query gets its own fused candidates and cross-encoder scores, and the lists
are merged.

Merges (`merge`):
- `orig`: the union of every query's candidate pool, reranked against the original question.
- `max`: each chunk keeps its best cross-encoder score over the queries whose pool it is in.
- `rr`: round-robin over the per-query reranked lists, original question first.

Used by: `fixgraph bench rr-decomp` (bench/cli.py) and retrieval.hybrid.DecomposingRetriever.
Uses: llm.structured.
"""

# Imports: pydantic for the reply shape, and the LLM client plus schema-checked completion helper.
from pydantic import BaseModel

from fixgraph.llm.base import ChatMessage, LLMClient, LLMRequest
from fixgraph.llm.structured import StructuredOutputError, complete_structured

# Prompt asking the local LLM to split a multi-problem question into at most 3 search queries.
DECOMPOSE_PROMPT = """Split a troubleshooting question into separate searches.

If the question describes more than one problem (or the same problem on different devices),
write one short, self-contained search query per problem, keeping the device names. If it
describes a single problem, return the question unchanged as the only query. At most 3 queries.

Question: {question}

Return JSON: {{"queries": ["...", "..."]}}"""


# Shape of the JSON reply the model must return: just a list of query strings.
class _Decomposed(BaseModel):
    queries: list[str]


# Question in, 2-3 sub-queries out; [] if it is a single problem or the LLM call fails.
def decompose(client: LLMClient, model: str, question: str) -> list[str]:
    """Sub-queries for `question` (deduplicated, at most 3, never containing the original).
    Returns [] when the model sees a single problem or fails."""
    # Build a small, short-context request with the question filled into the prompt.
    req = LLMRequest(
        model=model,
        messages=[ChatMessage(role="user", content=DECOMPOSE_PROMPT.format(question=question))],
        num_ctx=2048,
        max_tokens=200,
    )
    # Ask the model for schema-valid JSON; any failure falls back to no decomposition.
    try:
        out = complete_structured(client, req, _Decomposed).queries
    except (StructuredOutputError, RuntimeError):
        return []
    # Clean up the reply: drop blanks, duplicates and anything equal to the original question.
    seen = {question.strip().lower()}
    subs: list[str] = []
    for q in out:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            subs.append(q)
    # Only a real split (two or more distinct queries) counts; cap the result at 3.
    return subs[:3] if len(subs) >= 2 else []


# Combines the reranked lists from the original question and each sub-query into one top-k list.
def merge(
    lists: list[list[tuple[str, float]]],
    orig_scores: dict[str, float],
    how: str,
    k: int,
) -> list[str]:
    """Merge per-query reranked lists (index 0 = original question), each best first.
    `orig_scores` holds the original question's score for every pooled chunk (for `orig`)."""
    # "orig": pool every candidate, then rank by its score against the original question.
    if how == "orig":
        pool = {c for lst in lists for c, _ in lst}
        return sorted(pool, key=lambda c: -orig_scores[c])[:k]
    # "max": each chunk keeps its best score across all queries.
    if how == "max":
        best: dict[str, float] = {}
        for lst in lists:
            for c, s in lst:
                best[c] = max(best.get(c, float("-inf")), s)
        return sorted(best, key=lambda c: -best[c])[:k]
    # "rr": take the 1st item of each list in turn, then the 2nd, and so on, skipping repeats.
    if how == "rr":
        out: list[str] = []
        for i in range(max(len(lst) for lst in lists)):
            for lst in lists:
                if i < len(lst) and lst[i][0] not in out:
                    out.append(lst[i][0])
        return out[:k]
    # Unknown merge name is a caller error.
    raise ValueError(how)
