from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any

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
    def __init__(self, backend: BackendBase) -> None:
        self.backend = backend

    def run_case(self, case: BenchmarkCase) -> CaseRunResult:
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
            latency_seconds=base.latency_seconds,
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
            latency_seconds=followup.latency_seconds,
            prompt_tokens=followup.prompt_tokens,
            output_tokens=followup.output_tokens,
        )

        return CaseRunResult(
            case=case.to_dict(),
            base_result=base.to_dict(),
            followup_result=followup.to_dict(),
            base_metrics=base_metrics.to_dict(),
            followup_metrics=followup_metrics.to_dict(),
        )

    def run_cases(self, cases: List[BenchmarkCase]) -> List[CaseRunResult]:
        return [self.run_case(case) for case in cases]
