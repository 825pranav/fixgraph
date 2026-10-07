"""Deterministic in-memory LLMClient for tests and the fixture pipeline (no GPU, no network).

Selected by llm.factory when the backend is `fake`; api/app.py uses it for fake mode.
Uses: llm.base.
"""

# Imports: only the shared request/response types, since this client never talks to a server.
from collections.abc import Callable, Iterable

from fixgraph.llm.base import LLMRequest, LLMResponse

# A responder is any function that takes the request and returns the reply text.
Responder = Callable[[LLMRequest], str]


# Stand-in model client used by tests and the API's fake mode, so the pipeline runs without a GPU.
class FakeLLMClient:
    """Answers from a responder function, or pops scripted replies in order.

    Every request is recorded in `calls` so tests can assert on prompts.
    """

    # Store a responder function or a list of canned replies, and start an empty call log.
    def __init__(
        self, responder: Responder | None = None, scripted: Iterable[str] | None = None
    ) -> None:
        self._responder = responder
        self._scripted = list(scripted or [])
        self.calls: list[LLMRequest] = []

    # Record the request, then reply with the next scripted text, the responder's text, or an
    # empty default ("{}" when JSON was asked for so schema parsing still has something to read).
    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if self._scripted:
            text = self._scripted.pop(0)
        elif self._responder is not None:
            text = self._responder(request)
        else:
            text = "{}" if request.json_schema is not None else ""
        return LLMResponse(text=text, model=request.model)

    # Nothing to release for the fake client; exists so it matches the LLMClient protocol.
    def close(self) -> None:
        pass
