from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List


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


@dataclass
class DatasetConfig:
    mutation_jsonl_path: str = ""
    max_cases: Optional[int] = None
    shard_index: int = 0
    num_shards: int = 1
    include_relations: Optional[List[str]] = None


@dataclass
class OutputConfig:
    output_dir: str = "../data/benchmark_results"
    run_name: str = "prefix_cache_benchmark_run"


@dataclass
class BenchmarkConfig:
    backend: BackendConfig
    dataset: DatasetConfig
    output: OutputConfig
