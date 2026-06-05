import sys
from contextlib import asynccontextmanager
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.datastructures import Headers

from ray.llm._internal.serve.core.configs.openai_api_models import (
    ErrorInfo,
    ErrorResponse,
    TokenizeChatRequest,
    TokenizeCompletionRequest,
    TokenizeResponse,
)
from ray.llm._internal.serve.core.ingress.route_tokenizer import (
    REQUEST_TOKEN_IDS_KWARG,
    RouteTokenizer,
)
from ray.llm._internal.serve.core.ingress.router import LLMRouter


class _FakeRequest:
    """Minimal Starlette-request stand-in (mirrors test_router.py)."""

    def __init__(self, body: bytes, headers: Optional[dict] = None):
        self._body = body
        self.headers = Headers(headers or {})

    async def body(self) -> bytes:
        return self._body


def _make_handle_returning(resp):
    """Build a mock deployment handle whose ``.tokenize.remote`` yields ``resp``.

    Returns ``(handle, captured)`` where ``captured`` records the positional
    args passed into ``tokenize.remote`` (so tests can inspect the built
    Tokenize* request).
    """
    captured = {}

    async def _gen(req, raw_request_info):
        captured["req"] = req
        captured["raw_request_info"] = raw_request_info
        yield resp

    handle = MagicMock()
    # options(stream=True) returns a handle exposing tokenize.remote.
    streamed = MagicMock()
    streamed.tokenize.remote = _gen
    handle.options.return_value = streamed
    return handle, captured


class TestRouteTokenizer:
    @pytest.mark.asyncio
    async def test_chat_path(self):
        resp = TokenizeResponse(count=3, max_model_len=2048, tokens=[1, 2, 3])
        handle, captured = _make_handle_returning(resp)
        tokenizer = RouteTokenizer(handle)

        body = b'{"model": "m", "messages": [{"role": "user", "content": "hi"}]}'
        result = await tokenizer.tokenize(body, body_truncated=False)

        assert result == [1, 2, 3]
        # Inspect the request actually passed to tokenize.remote.
        handle.options.assert_called_once_with(stream=True)
        tok_req = captured["req"]
        assert isinstance(tok_req, TokenizeChatRequest)
        assert tok_req.model == "m"
        assert tok_req.messages == [{"role": "user", "content": "hi"}]
        assert tok_req.add_generation_prompt is True
        assert captured["raw_request_info"] is None

    @pytest.mark.asyncio
    async def test_completion_path(self):
        resp = TokenizeResponse(count=2, max_model_len=2048, tokens=[10, 11])
        handle, captured = _make_handle_returning(resp)
        tokenizer = RouteTokenizer(handle)

        body = b'{"model": "m", "prompt": "Hello"}'
        result = await tokenizer.tokenize(body, body_truncated=False)

        assert result == [10, 11]
        tok_req = captured["req"]
        assert isinstance(tok_req, TokenizeCompletionRequest)
        assert tok_req.prompt == "Hello"
        assert tok_req.add_special_tokens is True

    @pytest.mark.asyncio
    async def test_truncated_body_returns_none_and_skips_handle(self):
        resp = TokenizeResponse(count=1, max_model_len=2048, tokens=[1])
        handle, _ = _make_handle_returning(resp)
        tokenizer = RouteTokenizer(handle)

        body = b'{"model": "m", "prompt": "Hello"}'
        result = await tokenizer.tokenize(body, body_truncated=True)

        assert result is None
        handle.options.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_body_returns_none(self):
        handle, _ = _make_handle_returning(None)
        tokenizer = RouteTokenizer(handle)
        assert await tokenizer.tokenize(b"", body_truncated=False) is None
        handle.options.assert_not_called()

    @pytest.mark.asyncio
    async def test_multi_prompt_list_returns_none(self):
        resp = TokenizeResponse(count=1, max_model_len=2048, tokens=[1])
        handle, _ = _make_handle_returning(resp)
        tokenizer = RouteTokenizer(handle)

        body = b'{"model": "m", "prompt": ["a", "b"]}'
        result = await tokenizer.tokenize(body, body_truncated=False)

        assert result is None
        handle.options.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_response_returns_none(self):
        err = ErrorResponse(
            error=ErrorInfo(message="boom", type="BadRequest", code=400)
        )
        handle, _ = _make_handle_returning(err)
        tokenizer = RouteTokenizer(handle)

        body = b'{"model": "m", "prompt": "Hello"}'
        result = await tokenizer.tokenize(body, body_truncated=False)
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_json_returns_none(self):
        handle, _ = _make_handle_returning(None)
        tokenizer = RouteTokenizer(handle)
        assert await tokenizer.tokenize(b"not json {", body_truncated=False) is None
        handle.options.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_dict_json_returns_none(self):
        handle, _ = _make_handle_returning(None)
        tokenizer = RouteTokenizer(handle)
        assert await tokenizer.tokenize(b"[1, 2, 3]", body_truncated=False) is None
        handle.options.assert_not_called()

    @pytest.mark.asyncio
    async def test_neither_messages_nor_prompt_returns_none(self):
        handle, _ = _make_handle_returning(None)
        tokenizer = RouteTokenizer(handle)
        assert await tokenizer.tokenize(b'{"model": "m"}', body_truncated=False) is None
        handle.options.assert_not_called()


class _FakeRouteTokenizer:
    """Records calls and returns preset token IDs."""

    def __init__(self, result: Optional[List[int]]):
        self._result = result
        self.calls = []

    async def tokenize(self, request_body, body_truncated):
        self.calls.append((request_body, body_truncated))
        return self._result


class TestLLMRouterRoute:
    @pytest.mark.asyncio
    async def test_route_tokenizes_and_passes_token_ids_into_pick(self):
        # Bypass the async __init__ (mirrors test_router.py's _new_direct_router).
        router = LLMRouter.__new__(LLMRouter)
        router._handle = MagicMock()

        fake_tokenizer = _FakeRouteTokenizer([11, 22, 33])
        router._route_tokenizer = fake_tokenizer
        router._pick_replica = AsyncMock(return_value=("h", 1, "rid"))

        body = b'{"model": "m", "messages": [{"role": "user", "content": "hi"}]}'
        result = await router.route(_FakeRequest(body))

        assert result == {"host": "h", "port": 1, "replica_id": "rid"}
        assert fake_tokenizer.calls == [(body, False)]
        # The prompt token IDs are forwarded to _pick_replica for KV-aware routing.
        router._pick_replica.assert_called_once_with(
            handle=router._handle,
            request_body=body,
            body_truncated=False,
            request_token_ids=[11, 22, 33],
        )

    @pytest.mark.asyncio
    async def test_route_resilient_when_tokenization_returns_none(self):
        """Tokenization failure (None) must not break routing."""
        router = LLMRouter.__new__(LLMRouter)
        router._handle = MagicMock()

        fake_tokenizer = _FakeRouteTokenizer(None)
        router._route_tokenizer = fake_tokenizer
        router._pick_replica = AsyncMock(return_value=("h", 2, "rid2"))

        body = b'{"model": "m", "prompt": ["a", "b"]}'
        result = await router.route(_FakeRequest(body))

        assert result == {"host": "h", "port": 2, "replica_id": "rid2"}
        assert fake_tokenizer.calls == [(body, False)]
        # None is still forwarded; _pick_replica omits the kwarg from choose_replica.
        router._pick_replica.assert_called_once_with(
            handle=router._handle,
            request_body=body,
            body_truncated=False,
            request_token_ids=None,
        )

    @pytest.mark.asyncio
    async def test_pick_replica_forwards_token_ids_to_choose_replica(self):
        captured = {}

        @asynccontextmanager
        async def fake_choose_replica(*args, **kwargs):
            captured.update(kwargs)
            yield MagicMock(
                replica_id="rid",
                _replica=MagicMock(
                    backend_http_endpoint=("h", 9),
                    replica_id=MagicMock(to_full_id_str=lambda: "d#rid"),
                ),
            )

        handle = MagicMock()
        handle.choose_replica = fake_choose_replica
        router = LLMRouter.__new__(LLMRouter)
        router._handle = handle

        await router._pick_replica(
            handle=handle, request_body=b"{}", request_token_ids=[5, 6, 7]
        )
        assert captured == {
            "request_body": b"{}",
            "body_truncated": False,
            "_reserve": False,
            REQUEST_TOKEN_IDS_KWARG: [5, 6, 7],
        }


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
