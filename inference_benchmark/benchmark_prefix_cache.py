from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .benchmark_config import BackendConfig, DatasetConfig, OutputConfig, BenchmarkConfig
from .case_builder import load_mutation_records, build_cases
from .inference_runner import InferenceRunner
from .vllm_backend import VLLMBackend
from .sglang_backend import SGLangBackend


def make_backend(cfg: BackendConfig):
    if cfg.backend_name == 'vllm':
        return VLLMBackend(
            model_name=cfg.model_name,
            enable_prefix_caching=cfg.enable_prefix_caching,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            trust_remote_code=cfg.trust_remote_code,
            use_async_ttft=cfg.use_async_ttft,
        )
    if cfg.backend_name == 'sglang':
        return SGLangBackend(
            model_name=cfg.model_name,
            enable_prefix_caching=cfg.enable_prefix_caching,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            trust_remote_code=cfg.trust_remote_code,
        )
    raise ValueError(f'Unsupported backend: {cfg.backend_name}')


def run_benchmark(config: BenchmarkConfig) -> Path:
    records = load_mutation_records(config.dataset.mutation_jsonl_path)
    cases = build_cases(
        records,
        include_relations=config.dataset.include_relations,
        max_cases=config.dataset.max_cases,
        shard_index=config.dataset.shard_index,
        num_shards=config.dataset.num_shards,
    )

    backend = make_backend(config.backend)
    backend.start()
    runner = InferenceRunner(
        backend,
        warmup_iters=config.backend.warmup_iters,
        reset_cache_between_cases=config.backend.reset_cache_between_cases,
    )
    results = runner.run_cases(cases)
    backend.stop()

    out_dir = Path(config.output.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{config.output.run_name}.jsonl"
    summary_path = out_dir / f"{config.output.run_name}.summary.json"

    with out_path.open('w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res.to_dict(), ensure_ascii=False) + '\n')

    summary = {
        'num_cases': len(cases),
        'backend': asdict(config.backend),
        'dataset': asdict(config.dataset),
        'output': asdict(config.output),
        'output_path': str(out_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return out_path


def parse_args() -> BenchmarkConfig:
    p = argparse.ArgumentParser()
    p.add_argument('--backend-name', choices=['vllm', 'sglang'], default='vllm')
    p.add_argument('--model-name', required=True)
    p.add_argument('--enable-prefix-caching', action='store_true')
    p.add_argument('--disable-prefix-caching', action='store_true')
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    p.add_argument('--trust-remote-code', action='store_true')
    p.add_argument('--use-async-ttft', action='store_true', default=True,
                   help='Use AsyncLLMEngine to measure TTFT (default).')
    p.add_argument('--no-async-ttft', dest='use_async_ttft', action='store_false',
                   help='Fall back to offline LLM.generate; TTFT will be None.')
    p.add_argument('--warmup-iters', type=int, default=2,
                   help='Number of warmup requests run (and tagged) before the measured loop.')
    p.add_argument('--reset-cache-between-cases', action='store_true', default=True,
                   help='Reset the prefix KV cache before every (base, followup) pair '
                        '(default). Prevents cross-case contamination of unrelated_control.')
    p.add_argument('--no-reset-cache-between-cases', dest='reset_cache_between_cases',
                   action='store_false',
                   help='Disable per-case cache reset. Use only to reproduce the '
                        'pre-fix behavior where cases share cache state.')

    p.add_argument('--mutation-jsonl-path', required=True)
    p.add_argument('--max-cases', type=int, default=None)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--include-relations', nargs='*', default=['exact_reuse', 'partial_reuse', 'unrelated_control'])

    p.add_argument(
        '--output-dir',
        default=str(Path(__file__).resolve().parents[1] / 'outputs' / 'benchmark_results'),
    )
    p.add_argument('--run-name', default='prefix_cache_benchmark_run')

    args = p.parse_args()
    enable_prefix_caching = True
    if args.disable_prefix_caching:
        enable_prefix_caching = False
    elif args.enable_prefix_caching:
        enable_prefix_caching = True

    return BenchmarkConfig(
        backend=BackendConfig(
            backend_name=args.backend_name,
            model_name=args.model_name,
            enable_prefix_caching=enable_prefix_caching,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=args.trust_remote_code,
            use_async_ttft=args.use_async_ttft,
            warmup_iters=args.warmup_iters,
            reset_cache_between_cases=args.reset_cache_between_cases,
        ),
        dataset=DatasetConfig(
            mutation_jsonl_path=args.mutation_jsonl_path,
            max_cases=args.max_cases,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            include_relations=args.include_relations,
        ),
        output=OutputConfig(
            output_dir=args.output_dir,
            run_name=args.run_name,
        ),
    )


if __name__ == '__main__':
    cfg = parse_args()
    out_path = run_benchmark(cfg)
    print(f'Saved benchmark results to: {out_path}')
