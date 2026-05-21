from __future__ import annotations

import asyncio
import time
import uuid
from typing import Optional

from .backend_base import BackendBase, GenerationResult


class VLLMBackend(BackendBase):
    """vLLM backend with two modes:

    - use_async_ttft=True (default): uses AsyncLLMEngine and times the first
      generated token (TTFT) by consuming the streamed RequestOutput iterator.
      End-to-end latency is also measured. Runs on the SAME compute node as the
      SLURM job; no network, no server, no extra process.

    - use_async_ttft=False: falls back to the offline blocking LLM.generate
      API; TTFT is reported as None.

    Prefix caching is controlled by `enable_prefix_caching` and is passed
    through to vLLM as-is.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.llm = None
        self.async_engine = None
        self.sampling_params = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self) -> None:
        from vllm import SamplingParams
        self.sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        if self.use_async_ttft:
            from vllm import AsyncEngineArgs, AsyncLLMEngine
            engine_args = AsyncEngineArgs(
                model=self.model_name,
                enable_prefix_caching=self.enable_prefix_caching,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=self.trust_remote_code,
                disable_log_requests=True,
            )
            self.async_engine = AsyncLLMEngine.from_engine_args(engine_args)
            self._loop = asyncio.new_event_loop()
        else:
            from vllm import LLM
            self.llm = LLM(
                model=self.model_name,
                enable_prefix_caching=self.enable_prefix_caching,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=self.trust_remote_code,
            )
        self._started = True

    def stop(self) -> None:
        if self._loop is not None:
            try:
                self._loop.close()
            except Exception:
                pass
            self._loop = None
        self.async_engine = None
        self.llm = None
        self.sampling_params = None
        self._started = False

    def generate(self, prompt: str, request_id: Optional[str] = None) -> GenerationResult:
        if not self._started or self.sampling_params is None:
            raise RuntimeError('Backend not started. Call start() first.')
        if self.use_async_ttft:
            return self._generate_async(prompt, request_id)
        return self._generate_offline(prompt, request_id)

    def _generate_offline(self, prompt: str, request_id: Optional[str]) -> GenerationResult:
        t0 = time.perf_counter()
        outputs = self.llm.generate([prompt], self.sampling_params)
        latency = time.perf_counter() - t0
        out = outputs[0]
        text = out.outputs[0].text if out.outputs else ''
        prompt_tokens = len(out.prompt_token_ids) if getattr(out, 'prompt_token_ids', None) else None
        output_tokens = (
            len(out.outputs[0].token_ids)
            if out.outputs and getattr(out.outputs[0], 'token_ids', None)
            else None
        )
        return GenerationResult(
            text=text,
            ttft_seconds=None,
            latency_seconds=latency,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            raw={'request_id': request_id, 'mode': 'offline'},
        )

    def _generate_async(self, prompt: str, request_id: Optional[str]) -> GenerationResult:
        rid = request_id or uuid.uuid4().hex
        return self._loop.run_until_complete(self._run_one_request(prompt, rid))

    async def _run_one_request(self, prompt: str, request_id: str) -> GenerationResult:
        t0 = time.perf_counter()
        ttft: Optional[float] = None
        last_output = None
        prev_token_count = 0
        async for request_output in self.async_engine.generate(
            prompt=prompt,
            sampling_params=self.sampling_params,
            request_id=request_id,
        ):
            if ttft is None and request_output.outputs:
                cur_tokens = len(request_output.outputs[0].token_ids or [])
                if cur_tokens > prev_token_count:
                    ttft = time.perf_counter() - t0
                prev_token_count = cur_tokens
            last_output = request_output
        latency = time.perf_counter() - t0

        if last_output is None or not last_output.outputs:
            return GenerationResult(
                text='', ttft_seconds=ttft, latency_seconds=latency,
                raw={'request_id': request_id, 'mode': 'async', 'empty': True},
            )
        text = last_output.outputs[0].text
        prompt_tokens = (
            len(last_output.prompt_token_ids)
            if getattr(last_output, 'prompt_token_ids', None) else None
        )
        output_tokens = len(last_output.outputs[0].token_ids or [])
        return GenerationResult(
            text=text,
            ttft_seconds=ttft,
            latency_seconds=latency,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            raw={'request_id': request_id, 'mode': 'async'},
        )

    def backend_name(self) -> str:
        return 'vllm'
