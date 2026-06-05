"""Pre-routing tokenization of incoming OpenAI requests.

:class:`RouteTokenizer` turns a raw ChatCompletions / Completions request body
into prompt token IDs by calling the selected vLLM replica's ``/tokenize``
endpoint through the LLMServer deployment handle. The replica applies the chat
template for chat requests, so the resulting token IDs match what the engine
will actually prefill.

The token IDs feed KV-aware routing (a request router scores replicas on them).
Tokenization is best-effort: any failure returns ``None`` so the caller falls
back to normal (body-unaware) routing.
"""

import json
from typing import List, Optional

from ray.llm._internal.serve.core.configs.openai_api_models import (
    ErrorResponse,
    TokenizeChatRequest,
    TokenizeCompletionRequest,
)
from ray.llm._internal.serve.observability.logging import get_logger
from ray.serve.handle import DeploymentHandle

logger = get_logger(__name__)

# choose_replica kwarg carrying the prompt token IDs to KV-aware routers.
# Mirrors the ``token_ids`` of Dynamo's internal PreprocessedRequest.
REQUEST_TOKEN_IDS_KWARG = "request_token_ids"


class RouteTokenizer:
    """Tokenizes incoming requests via the replica's ``/tokenize`` endpoint.

    Args:
        handle: A handle to the LLMServer deployment. Its ``tokenize`` method is
            an async generator yielding exactly one ``TokenizeResponse`` (or an
            ``ErrorResponse``).
    """

    def __init__(self, handle: DeploymentHandle):
        self._handle = handle

    async def tokenize(
        self, request_body: bytes, body_truncated: bool
    ) -> Optional[List[int]]:
        """Tokenize ``request_body`` into prompt token IDs.

        Returns the token IDs, or ``None`` (graceful fallback to normal routing)
        when the body is truncated/empty, an unsupported shape, or tokenization
        fails.
        """
        try:
            if body_truncated or not request_body:
                return None

            payload = json.loads(request_body)
            if not isinstance(payload, dict):
                return None

            model = payload.get("model")
            tok_req: object
            if "messages" in payload:
                tok_req = TokenizeChatRequest.model_validate(
                    {
                        "model": model,
                        "messages": payload["messages"],
                        "add_generation_prompt": True,
                    }
                )
            elif "prompt" in payload:
                prompt = payload["prompt"]
                if not isinstance(prompt, str):
                    # Multi-prompt (list) tokenization is out of scope; fall
                    # back to normal routing.
                    return None
                tok_req = TokenizeCompletionRequest.model_validate(
                    {
                        "model": model,
                        "prompt": prompt,
                        "add_special_tokens": True,
                    }
                )
            else:
                return None

            gen = self._handle.options(stream=True).tokenize.remote(tok_req, None)
            resp = await gen.__anext__()
            if isinstance(resp, ErrorResponse):
                return None
            return list(resp.tokens)
        except Exception as e:
            logger.debug("Pre-routing tokenization failed, falling back: %s", e)
            return None
