from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


def safe_div(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return num / den


@dataclass
class RequestMetrics:
    case_id: str
    relation: str
    backend_name: str
    model_name: str
    cache_enabled: bool
    phase: str
    # See GenerationResult for the precise definitions.
    ttft_seconds: Optional[float]
    wall_clock_seconds: float
    prompt_tokens: Optional[int]
    output_tokens: Optional[int]
    tokens_per_second: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_request_metrics(
    case_id: str,
    relation: str,
    backend_name: str,
    model_name: str,
    cache_enabled: bool,
    phase: str,
    ttft_seconds: Optional[float],
    wall_clock_seconds: float,
    prompt_tokens: Optional[int],
    output_tokens: Optional[int],
) -> RequestMetrics:
    return RequestMetrics(
        case_id=case_id,
        relation=relation,
        backend_name=backend_name,
        model_name=model_name,
        cache_enabled=cache_enabled,
        phase=phase,
        ttft_seconds=ttft_seconds,
        wall_clock_seconds=wall_clock_seconds,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        tokens_per_second=safe_div(output_tokens or 0, wall_clock_seconds),
    )
