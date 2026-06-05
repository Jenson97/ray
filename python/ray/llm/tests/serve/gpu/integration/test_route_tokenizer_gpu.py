"""GPU integration test for pre-routing tokenization.

Deploys a direct-streaming LLM app backed by a real (tiny) vLLM engine and
exercises the REAL ``RouteTokenizer`` path: it calls the deployed replica's
``/tokenize`` endpoint through the deployment handle and we assert the produced
token IDs match an independent HuggingFace tokenizer ground truth.

The implementation under test must NOT use HuggingFace directly; the HF
tokenizer here is only used as the test's independent ground truth.
"""

import os
import sys
from collections.abc import Mapping

import pytest

# Direct streaming must be enabled before importing the serve builders, since
# the flag is read into a module-level constant at import time.
os.environ["RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING"] = "1"

from ray import serve  # noqa: E402
from ray.llm._internal.serve.core.ingress.builder import (  # noqa: E402
    _build_direct_streaming_llm_deployment,
)
from ray.llm._internal.serve.core.ingress.route_tokenizer import (  # noqa: E402
    RouteTokenizer,
)
from ray.serve.llm import LLMConfig, ModelLoadingConfig  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


class TestRouteTokenizerGPU:
    @pytest.fixture(scope="class")
    def deployed_handle(self):
        """Deploy a direct-streaming LLMServer once for the whole class.

        Returns a DeploymentHandle to the LLMServer (ingress) deployment, which
        is exactly the handle ``RouteTokenizer`` calls ``tokenize`` on.
        """
        # Make sure no prior serve app is holding GPU memory.
        serve.shutdown()

        llm_config = LLMConfig(
            model_loading_config=ModelLoadingConfig(
                model_id=MODEL_ID,
                model_source=MODEL_ID,
            ),
            deployment_config=dict(
                autoscaling_config=dict(min_replicas=1, max_replicas=1),
            ),
            engine_kwargs=dict(
                max_model_len=2048,
                enforce_eager=True,
                gpu_memory_utilization=0.4,
                use_tqdm_on_load=False,
            ),
            # Single GPU; no specific accelerator type required.
            placement_group_config={"bundles": [{"GPU": 1}]},
            runtime_env=dict(env_vars={"VLLM_DISABLE_COMPILE_CACHE": "1"}),
            log_engine_metrics=False,
        )

        # _build_direct_streaming_llm_deployment wraps LLMServer with
        # serve.ingress() (the direct-streaming server). Running it directly
        # (without the LLMRouter peer) gives a handle to LLMServer, avoiding the
        # HAProxy requirement of the full router-peer wiring while still
        # exercising the real /tokenize path.
        app = _build_direct_streaming_llm_deployment(llm_config)
        handle = serve.run(app, name="route_tokenizer_gpu_test")
        yield handle
        serve.shutdown()

    @pytest.fixture(scope="class")
    def hf_tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(MODEL_ID)

    @pytest.mark.asyncio
    async def test_chat_tokenization_matches_hf_ground_truth(
        self, deployed_handle, hf_tokenizer
    ):
        messages = [{"role": "user", "content": "Hello, world!"}]
        body = (
            b'{"model": "' + MODEL_ID.encode() + b'", "messages": '
            b'[{"role": "user", "content": "Hello, world!"}]}'
        )

        tokenizer = RouteTokenizer(deployed_handle)
        result = await tokenizer.tokenize(body, body_truncated=False)

        assert result is not None

        # Independent ground truth via HuggingFace (test-only). With
        # tokenize=True the tokenizer may return a bare list of ids or a
        # mapping-like BatchEncoding (which is a UserDict, not a dict subclass),
        # so detect the token ids via the input_ids key when present.
        reference = hf_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        if isinstance(reference, Mapping):
            reference = reference["input_ids"]
        assert result == list(reference)

    @pytest.mark.asyncio
    async def test_completion_tokenization_matches_hf_ground_truth(
        self, deployed_handle, hf_tokenizer
    ):
        prompt = "Hello, world!"
        body = b'{"model": "' + MODEL_ID.encode() + b'", "prompt": "Hello, world!"}'

        tokenizer = RouteTokenizer(deployed_handle)
        result = await tokenizer.tokenize(body, body_truncated=False)

        assert result is not None

        # vLLM /tokenize for completions uses add_special_tokens=True, matching
        # RouteTokenizer's TokenizeCompletionRequest.
        reference = hf_tokenizer(prompt, add_special_tokens=True)["input_ids"]
        assert result == list(reference)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))
