"""Deterministic in-memory LLMClient for tests and the fixture pipeline (no GPU, no network)."""

from collections.abc import Callable, Iterable

from fixgraph.llm.base import LLMRequest, LLMResponse

Responder = Callable[[LLMRequest], str]


class FakeLLMClient:
    """Answers from a responder function, or pops scripted replies in order.

    Every request is recorded in `calls` so tests can assert on prompts.
    """

    def __init__(
        self, responder: Responder | None = None, scripted: Iterable[str] | None = None
    ) -> None:
        self._responder = responder
        self._scripted = list(scripted or [])
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if self._scripted:
            text = self._scripted.pop(0)
        elif self._responder is not None:
            text = self._responder(request)
        else:
            text = "{}" if request.json_schema is not None else ""
        return LLMResponse(text=text, model=request.model)

    def close(self) -> None:
        pass
