from __future__ import annotations

import asyncio
import dataclasses
import time
import uuid
from typing import Any, Dict, Optional

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
        # Discovered at start() time: a zero-arg callable that flushes the
        # engine's prefix KV cache, or None if no working path was found in
        # this vLLM version. We probe once so we don't waste time per case.
        self._reset_prefix_cache_fn: Optional[Any] = None
        self._reset_prefix_cache_is_async: bool = False

    def start(self) -> None:
        from vllm import SamplingParams
        self.sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        if self.use_async_ttft:
            from vllm import AsyncEngineArgs, AsyncLLMEngine
            # AsyncEngineArgs is a dataclass whose fields drift across vLLM versions.
            # Build only the kwargs that actually exist in this installed wheel.
            try:
                field_names = {f.name for f in dataclasses.fields(AsyncEngineArgs)}
            except TypeError:
                field_names = set()
            kwargs: Dict[str, Any] = {
                'model': self.model_name,
                'enable_prefix_caching': self.enable_prefix_caching,
                'gpu_memory_utilization': self.gpu_memory_utilization,
                'trust_remote_code': self.trust_remote_code,
            }
            if self.max_model_len is not None:
                kwargs['max_model_len'] = self.max_model_len
            # quiet per-request logging
            if 'disable_log_requests' in field_names:
                kwargs['disable_log_requests'] = True   # vLLM < 0.10
            elif 'enable_log_requests' in field_names:
                kwargs['enable_log_requests'] = False   # vLLM >= 0.10 (renamed + inverted)
            engine_args = AsyncEngineArgs(**kwargs)
            self.async_engine = AsyncLLMEngine.from_engine_args(engine_args)
            self._loop = asyncio.new_event_loop()
        else:
            from vllm import LLM
            llm_kwargs: Dict[str, Any] = {
                'model': self.model_name,
                'enable_prefix_caching': self.enable_prefix_caching,
                'gpu_memory_utilization': self.gpu_memory_utilization,
                'trust_remote_code': self.trust_remote_code,
            }
            if self.max_model_len is not None:
                llm_kwargs['max_model_len'] = self.max_model_len
            self.llm = LLM(**llm_kwargs)
        self._load_chat_tokenizer()
        self._discover_reset_prefix_cache()
        self._started = True

    def _discover_reset_prefix_cache(self) -> None:
        """Find the right `reset_prefix_cache` entry point in this vLLM build.

        vLLM has moved this method around between versions:
          - sync LLM:  llm.llm_engine.reset_prefix_cache()
          - sync LLM (older): llm.reset_prefix_cache()
          - AsyncLLMEngine v0:  async_engine.engine.reset_prefix_cache()
          - AsyncLLMEngine v1:  async_engine.reset_prefix_cache()  (coroutine)

        We probe candidate paths in order, keep the first that exists, and
        record whether it is a coroutine so the runtime call site can await
        it correctly. If nothing matches, the field stays None and
        reset_prefix_cache() becomes a logged no-op.
        """
        import inspect

        candidates = []
        if self.async_engine is not None:
            candidates += [
                ('async_engine.reset_prefix_cache', self.async_engine, 'reset_prefix_cache'),
                ('async_engine.engine.reset_prefix_cache',
                 getattr(self.async_engine, 'engine', None), 'reset_prefix_cache'),
            ]
        if self.llm is not None:
            candidates += [
                ('llm.llm_engine.reset_prefix_cache',
                 getattr(self.llm, 'llm_engine', None), 'reset_prefix_cache'),
                ('llm.reset_prefix_cache', self.llm, 'reset_prefix_cache'),
            ]

        for label, owner, attr in candidates:
            if owner is None:
                continue
            fn = getattr(owner, attr, None)
            if callable(fn):
                self._reset_prefix_cache_fn = fn
                self._reset_prefix_cache_is_async = inspect.iscoroutinefunction(fn)
                print(f"[VLLMBackend] reset_prefix_cache via {label} "
                      f"(async={self._reset_prefix_cache_is_async})")
                return

        print(
            "[VLLMBackend] WARNING: no reset_prefix_cache entry point found in "
            "this vLLM build; cross-case cache contamination will NOT be cleared. "
            "Consider upgrading vLLM, or expect a non-zero unrelated_control noise floor."
        )

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
        # Wrap here, at the last point before the engine, so the assistant turn
        # marker is always the final content of the prompt no matter what the
        # layout strategy produced.
        prompt = self.format_prompt(prompt)
        if self.use_async_ttft:
            return self._generate_async(prompt, request_id)
        return self._generate_offline(prompt, request_id)

    def _generate_offline(self, prompt: str, request_id: Optional[str]) -> GenerationResult:
        t0 = time.perf_counter()
        outputs = self.llm.generate([prompt], self.sampling_params)
        wall_clock = time.perf_counter() - t0
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
            wall_clock_seconds=wall_clock,
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
        wall_clock = time.perf_counter() - t0

        if last_output is None or not last_output.outputs:
            return GenerationResult(
                text='', ttft_seconds=ttft, wall_clock_seconds=wall_clock,
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
            wall_clock_seconds=wall_clock,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            raw={'request_id': request_id, 'mode': 'async'},
        )

    def reset_prefix_cache(self) -> bool:
        if self._reset_prefix_cache_fn is None:
            return False
        try:
            if self._reset_prefix_cache_is_async:
                if self._loop is None:
                    # Should not happen: async reset implies AsyncLLMEngine path.
                    return False
                self._loop.run_until_complete(self._reset_prefix_cache_fn())
            else:
                self._reset_prefix_cache_fn()
            return True
        except Exception as e:
            # Surface once, then disable so we don't spam the log every case.
            print(f"[VLLMBackend] reset_prefix_cache failed: {type(e).__name__}: {e}. "
                  "Disabling further reset attempts for this run.")
            self._reset_prefix_cache_fn = None
            return False

    def backend_name(self) -> str:
        return 'vllm'
