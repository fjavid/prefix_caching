from __future__ import annotations

import time
from typing import Optional

from .backend_base import BackendBase, GenerationResult


class VLLMBackend(BackendBase):
    """
    Offline vLLM backend using the Python API.

    Prefix caching is controlled by the `enable_prefix_caching` flag.
    TTFT is not directly exposed by the offline Python API here, so it is None.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.llm = None
        self.sampling_params = None

    def start(self) -> None:
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=self.model_name,
            enable_prefix_caching=self.enable_prefix_caching,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=self.trust_remote_code,
        )
        self.sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        self._started = True

    def stop(self) -> None:
        self.llm = None
        self.sampling_params = None
        self._started = False

    def generate(self, prompt: str, request_id: Optional[str] = None) -> GenerationResult:
        if not self._started or self.llm is None or self.sampling_params is None:
            raise RuntimeError('Backend not started. Call start() first.')

        t0 = time.perf_counter()
        outputs = self.llm.generate([prompt], self.sampling_params)
        latency = time.perf_counter() - t0

        out = outputs[0]
        text = out.outputs[0].text if out.outputs else ''
        prompt_tokens = len(out.prompt_token_ids) if getattr(out, 'prompt_token_ids', None) else None
        output_tokens = len(out.outputs[0].token_ids) if out.outputs and getattr(out.outputs[0], 'token_ids', None) else None

        return GenerationResult(
            text=text,
            ttft_seconds=None,
            latency_seconds=latency,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            raw={'request_id': request_id},
        )

    def backend_name(self) -> str:
        return 'vllm'
