"""Schema-constrained generation: Pydantic model -> JSON Schema -> validated object (spec §5.5)."""

import logging

from pydantic import BaseModel, ValidationError

from fixgraph.llm.base import ChatMessage, LLMClient, LLMRequest

logger = logging.getLogger(__name__)


class StructuredOutputError(ValueError):
    """The model's output failed validation even after the retry."""


def complete_structured[T: BaseModel](
    client: LLMClient, request: LLMRequest, output_model: type[T]
) -> T:
    """Request JSON matching `output_model`; on invalid output, retry once with the error appended.

    Callers decide whether to log-and-skip on StructuredOutputError.
    """
    schema_request = request.model_copy(update={"json_schema": output_model.model_json_schema()})
    response = client.complete(schema_request)
    try:
        return output_model.model_validate_json(response.text)
    except ValidationError as first_error:
        logger.warning("Structured output invalid, retrying once: %s", first_error)
        retry_messages = [
            *schema_request.messages,
            ChatMessage(role="assistant", content=response.text),
            ChatMessage(
                role="user",
                content=(
                    "Your previous output did not validate against the schema:\n"
                    f"{first_error}\nReturn corrected JSON only."
                ),
            ),
        ]
        retry = client.complete(schema_request.model_copy(update={"messages": retry_messages}))
        try:
            return output_model.model_validate_json(retry.text)
        except ValidationError as second_error:
            raise StructuredOutputError(str(second_error)) from second_error
