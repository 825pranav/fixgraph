import pytest
from pydantic import BaseModel

from fixgraph.llm import (
    ChatMessage,
    FakeLLMClient,
    LLMRequest,
    StructuredOutputError,
    complete_structured,
)


class Out(BaseModel):
    product: str
    count: int


REQ = LLMRequest(model="m", messages=[ChatMessage(role="user", content="extract")])


def test_valid_first_try_passes_schema() -> None:
    fake = FakeLLMClient(scripted=['{"product": "AirPods Pro", "count": 2}'])
    out = complete_structured(fake, REQ, Out)
    assert out == Out(product="AirPods Pro", count=2)
    assert fake.calls[0].json_schema == Out.model_json_schema()


def test_retries_once_with_error_appended() -> None:
    fake = FakeLLMClient(scripted=['{"product": "iPhone"}', '{"product": "iPhone", "count": 1}'])
    out = complete_structured(fake, REQ, Out)
    assert out.count == 1
    assert len(fake.calls) == 2
    retry_msgs = fake.calls[1].messages
    assert retry_msgs[-2].role == "assistant"
    assert "did not validate" in retry_msgs[-1].content


def test_raises_after_second_failure() -> None:
    fake = FakeLLMClient(scripted=["not json", "still not json"])
    with pytest.raises(StructuredOutputError):
        complete_structured(fake, REQ, Out)
    assert len(fake.calls) == 2
