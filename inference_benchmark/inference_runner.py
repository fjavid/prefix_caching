from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from .backend_base import BackendBase
from .case_builder import BenchmarkCase
from .metrics_utils import build_request_metrics


@dataclass
class CaseRunResult:
    case: Dict[str, Any]
    base_result: Dict[str, Any]
    followup_result: Dict[str, Any]
    base_metrics: Dict[str, Any]
    followup_metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InferenceRunner:
    """Sequential single-request runner.

    Sequential is intentional: prefix-cache measurements depend on the order in
    which requests arrive. The first request after backend.start() carries
    engine warmup cost (CUDA graph capture, KV allocation, etc.) so we
    explicitly run a warmup pass before the measured loop and tag those
    metrics with phase='warmup' so the analysis layer can drop them.

    If reset_cache_between_cases=True (default) the runner calls
    backend.reset_prefix_cache() BEFORE every case, so each (base, followup)
    pair starts with an empty KV cache. Without this, every case after the
    first sees cache state from prior cases, which contaminates the
    unrelated_control noise floor in particular.
    """

    def __init__(
        self,
        backend: BackendBase,
        warmup_iters: int = 2,
        reset_cache_between_cases: bool = True,
    ) -> None:
        self.backend = backend
        self.warmup_iters = max(0, warmup_iters)
        self.reset_cache_between_cases = reset_cache_between_cases
        self._reset_supported: Optional[bool] = None  # discovered on first call

    def warmup(self, cases: List[BenchmarkCase]) -> List[Dict[str, Any]]:
        """Run a few prompts solely to absorb startup cost. Results are tagged
        with phase='warmup' and should be filtered out of any analysis."""
        if not cases or self.warmup_iters == 0:
            return []
        warmup_metrics: List[Dict[str, Any]] = []
        for i in range(min(self.warmup_iters, len(cases))):
            case = cases[i]
            res = self.backend.generate(case.base_prompt, request_id=f"warmup::{case.case_id}")
            warmup_metrics.append(
                build_request_metrics(
                    case_id=case.case_id,
                    relation='warmup',
                    backend_name=self.backend.backend_name(),
                    model_name=self.backend.model_name,
                    cache_enabled=self.backend.enable_prefix_caching,
                    phase='warmup',
                    ttft_seconds=res.ttft_seconds,
                    wall_clock_seconds=res.wall_clock_seconds,
                    prompt_tokens=res.prompt_tokens,
                    output_tokens=res.output_tokens,
                ).to_dict()
            )
        return warmup_metrics

    def run_case(self, case: BenchmarkCase) -> CaseRunResult:
        if self.reset_cache_between_cases:
            did = self.backend.reset_prefix_cache()
            if self._reset_supported is None:
                self._reset_supported = did
                if not did:
                    print(
                        "[InferenceRunner] reset_cache_between_cases is enabled but the "
                        "backend reported no working reset path; cases will share cache "
                        "state. Treat unrelated_control values with caution."
                    )
        base = self.backend.generate(case.base_prompt, request_id=case.case_id + '::base')
        followup = self.backend.generate(case.followup_prompt, request_id=case.case_id + '::followup')

        base_metrics = build_request_metrics(
            case_id=case.case_id,
            relation=case.relation,
            backend_name=self.backend.backend_name(),
            model_name=self.backend.model_name,
            cache_enabled=self.backend.enable_prefix_caching,
            phase='base',
            ttft_seconds=base.ttft_seconds,
            wall_clock_seconds=base.wall_clock_seconds,
            prompt_tokens=base.prompt_tokens,
            output_tokens=base.output_tokens,
        )
        followup_metrics = build_request_metrics(
            case_id=case.case_id,
            relation=case.relation,
            backend_name=self.backend.backend_name(),
            model_name=self.backend.model_name,
            cache_enabled=self.backend.enable_prefix_caching,
            phase='followup',
            ttft_seconds=followup.ttft_seconds,
            wall_clock_seconds=followup.wall_clock_seconds,
            prompt_tokens=followup.prompt_tokens,
            output_tokens=followup.output_tokens,
        )

        # Engine-visible overlap: model tokens after chat templating, i.e. what
        # the prefix cache actually matches on. Recorded here because this is
        # the only stage with the model's tokenizer loaded; the analysis stage
        # must stay GPU-free and model-free.
        case_dict = case.to_dict()
        token_overlap = self.backend.token_overlap(
            case.base_prompt, case.followup_prompt
        )
        if token_overlap:
            case_dict['metadata'] = dict(case_dict.get('metadata') or {})
            case_dict['metadata']['model_token_overlap'] = token_overlap

        return CaseRunResult(
            case=case_dict,
            base_result=base.to_dict(),
            followup_result=followup.to_dict(),
            base_metrics=base_metrics.to_dict(),
            followup_metrics=followup_metrics.to_dict(),
        )

    def run_cases(self, cases: List[BenchmarkCase], do_warmup: bool = True) -> List[CaseRunResult]:
        """Run every case sequentially and return the per-case result list.

        do_warmup: when True (default) the runner runs `warmup_iters` warmup
        requests at the start and resets the cache before the first measured
        case. When False, this preamble is skipped. Set False when this is
        the second or later batch on an already-warmed engine (e.g. the
        single-engine multi-layout sweep) so warmup cost is paid once per
        engine instead of once per layout.
        """
        if do_warmup:
            self.warmup(cases)
            if self.reset_cache_between_cases:
                self.backend.reset_prefix_cache()
        return [self.run_case(case) for case in cases]
