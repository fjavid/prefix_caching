from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

_DEFAULT_BENCHMARK_OUTPUT_DIR = str(
    Path(__file__).resolve().parents[1] / "outputs" / "benchmark_results"
)


@dataclass
class BackendConfig:
    backend_name: str = "vllm"
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    enable_prefix_caching: bool = True
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    gpu_memory_utilization: float = 0.85
    trust_remote_code: bool = False
    use_async_ttft: bool = True
    warmup_iters: int = 2
    # Reset the prefix cache between cases so each (base, followup) pair is
    # measured against an empty cache. Prevents cross-case contamination, e.g.
    # unrelated_control followups hitting blocks left over from earlier cases.
    reset_cache_between_cases: bool = True


@dataclass
class LayoutTask:
    """One (layout_name, mutation_jsonl) pair to run in the engine sweep."""
    layout_name: str
    mutation_jsonl_path: str


@dataclass
class DatasetConfig:
    # The benchmark stage runs ONE engine per (cache_mode) and sweeps every
    # layout listed here back-to-back on that same engine. Each task writes a
    # separate output JSONL named via `OutputConfig.run_name_template`.
    layout_tasks: List[LayoutTask] = field(default_factory=list)
    max_cases: Optional[int] = None
    shard_index: int = 0
    num_shards: int = 1
    include_relations: Optional[List[str]] = None


@dataclass
class OutputConfig:
    output_dir: str = _DEFAULT_BENCHMARK_OUTPUT_DIR
    # Template containing `{layout}` (and optionally other placeholders the
    # caller fills in). One JSONL file per layout_task is written using
    # template.format(layout=layout_name).
    run_name_template: str = "prefix_cache_benchmark_{layout}"


@dataclass
class BenchmarkConfig:
    backend: BackendConfig
    dataset: DatasetConfig
    output: OutputConfig
