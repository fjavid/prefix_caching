from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class GenerationResult:
    text: str
    ttft_seconds: Optional[float]
    latency_seconds: float
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BackendBase:
    def __init__(
        self,
        model_name: str,
        enable_prefix_caching: bool,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = False,
        use_async_ttft: bool = True,
    ) -> None:
        self.model_name = model_name
        self.enable_prefix_caching = enable_prefix_caching
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.use_async_ttft = use_async_ttft
        self._started = False

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def restart(self) -> None:
        self.stop()
        self.start()

    def generate(self, prompt: str, request_id: Optional[str] = None) -> GenerationResult:
        raise NotImplementedError

    def reset_prefix_cache(self) -> bool:
        """Flush the engine's prefix KV cache between cases.

        Subclasses should override. Returns True if the reset actually
        happened, False if the backend has no working reset path (in which
        case the caller may want to log a warning since experimental
        isolation between cases is then NOT guaranteed).
        """
        return False

    def backend_name(self) -> str:
        raise NotImplementedError
