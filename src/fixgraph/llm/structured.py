"""Schema-constrained generation: Pydantic model -> JSON Schema -> validated object (spec §5.5).

`complete_structured` attaches the JSON Schema to the request, validates the reply and retries
once with the validation error appended. Used by every structured LLM call: kg.run_extract,
kg.resolve, retrieval.linking, answer.grounded, answer.verifier, bench.judge, bench.generate,
and `fixgraph llm smoke`. Uses: llm.base.
"""

# Imports: logging for the retry warning, pydantic for validation, and the shared LLM types.
import logging

from pydantic import BaseModel, ValidationError

from fixgraph.llm.base import ChatMessage, LLMClient, LLMRequest

logger = logging.getLogger(__name__)


# Raised when the model's JSON still fails the schema after one retry; callers choose to skip.
class StructuredOutputError(ValueError):
    """The model's output failed validation even after the retry."""


# Core helper for all structured calls: Pydantic model in, validated Pydantic object out.
# Flow: attach schema -> call model -> validate -> on failure retry once with the error shown.
def complete_structured[T: BaseModel](
    client: LLMClient, request: LLMRequest, output_model: type[T]
) -> T:
    """Request JSON matching `output_model`; on invalid output, retry once with the error appended.

    Callers decide whether to log-and-skip on StructuredOutputError.
    """
    # Attach the model's JSON schema so the backend (e.g. Ollama `format`) constrains the output.
    schema_request = request.model_copy(update={"json_schema": output_model.model_json_schema()})
    response = client.complete(schema_request)
    # First attempt: parse the raw reply text straight into the output model.
    try:
        return output_model.model_validate_json(response.text)
    except ValidationError as first_error:
        logger.warning("Structured output invalid, retrying once: %s", first_error)
        # Build a retry chat: original messages, the bad reply, then the validation error text.
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
        # Second and last attempt; if it still fails, raise so the caller can log and move on.
        retry = client.complete(schema_request.model_copy(update={"messages": retry_messages}))
        try:
            return output_model.model_validate_json(retry.text)
        except ValidationError as second_error:
            raise StructuredOutputError(str(second_error)) from second_error
