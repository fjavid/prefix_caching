from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List

from .benchmark_config import (
    BackendConfig, DatasetConfig, OutputConfig, BenchmarkConfig, LayoutTask,
)
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
            max_model_len=cfg.max_model_len,
            apply_chat_template=cfg.apply_chat_template,
            tokenizer_path=cfg.tokenizer_path,
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
            max_model_len=cfg.max_model_len,
            apply_chat_template=cfg.apply_chat_template,
            tokenizer_path=cfg.tokenizer_path,
        )
    raise ValueError(f'Unsupported backend: {cfg.backend_name}')


def _load_cases_for_task(task: LayoutTask, dataset: DatasetConfig):
    records = load_mutation_records(task.mutation_jsonl_path)
    return build_cases(
        records,
        include_relations=dataset.include_relations,
        max_cases=dataset.max_cases,
        shard_index=dataset.shard_index,
        num_shards=dataset.num_shards,
    )


def run_benchmark(config: BenchmarkConfig) -> List[Path]:
    """Run every layout task in the config on ONE vLLM engine.

    Sequence:
      1. Start the backend once (one engine per cache_mode).
      2. Warm it up using cases from the FIRST layout task.
      3. Reset the prefix cache after warmup.
      4. For each layout task, reset the prefix cache and run its cases.
         The per-case reset inside InferenceRunner.run_case still fires for
         every (base, followup) pair within a task; the explicit reset here
         is only an extra cleanup at the layout boundary.
      5. Stop the backend.

    Each task gets its own output JSONL named via output.run_name_template.
    Returns the list of written paths in the order the tasks ran.
    """
    if not config.dataset.layout_tasks:
        raise ValueError("At least one LayoutTask is required.")
    if "{layout}" not in config.output.run_name_template:
        raise ValueError(
            "OutputConfig.run_name_template must contain '{layout}' so each "
            "layout task gets its own output JSONL."
        )

    # Pre-load every task's cases so we fail fast on missing input files,
    # BEFORE we pay the cost of spinning up the vLLM engine.
    task_cases = [(task, _load_cases_for_task(task, config.dataset))
                  for task in config.dataset.layout_tasks]

    backend = make_backend(config.backend)
    backend.start()
    runner = InferenceRunner(
        backend,
        warmup_iters=config.backend.warmup_iters,
        reset_cache_between_cases=config.backend.reset_cache_between_cases,
    )

    # ---- One-time engine warmup, reusing the first task's prompts. -----
    first_task, first_cases = task_cases[0]
    print(f"[benchmark] warmup using layout='{first_task.layout_name}' "
          f"({len(first_cases)} cases available; running "
          f"{runner.warmup_iters} warmup iters)")
    runner.warmup(first_cases)
    if config.backend.reset_cache_between_cases:
        backend.reset_prefix_cache()

    out_dir = Path(config.output.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for task, cases in task_cases:
        run_name = config.output.run_name_template.format(layout=task.layout_name)
        out_path = out_dir / f"{run_name}.jsonl"
        summary_path = out_dir / f"{run_name}.summary.json"

        print(f"[benchmark] running layout='{task.layout_name}' "
              f"({len(cases)} cases) -> {out_path.name}")
        # Belt-and-braces: per-case reset already runs at the start of every
        # run_case, but resetting once here guarantees the very first case of
        # this layout starts with no residual blocks from the prior layout.
        if config.backend.reset_cache_between_cases:
            backend.reset_prefix_cache()

        results = runner.run_cases(cases, do_warmup=False)

        with out_path.open('w', encoding='utf-8') as f:
            for res in results:
                f.write(json.dumps(res.to_dict(), ensure_ascii=False) + '\n')

        summary = {
            'num_cases': len(cases),
            'layout_name': task.layout_name,
            'mutation_jsonl_path': task.mutation_jsonl_path,
            'backend': asdict(config.backend),
            'output_path': str(out_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        written.append(out_path)

    backend.stop()
    return written


def parse_args() -> BenchmarkConfig:
    p = argparse.ArgumentParser(
        description="Run the prefix-cache benchmark across multiple layouts "
                    "using a SINGLE vLLM engine instance (per cache mode).",
    )
    p.add_argument('--backend-name', choices=['vllm', 'sglang'], default='vllm')
    p.add_argument('--model-name', required=True)
    p.add_argument('--enable-prefix-caching', action='store_true')
    p.add_argument('--disable-prefix-caching', action='store_true')
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    p.add_argument('--max-model-len', type=int, default=None,
                   help='Cap the engine context window. Required for models declaring '
                        'a very large window (Llama-3.1 declares 131072), which would '
                        'otherwise size the KV cache for the full window and fail '
                        'allocation. Set to MAX_PROMPT_TOKENS + max_new_tokens rounded '
                        'up to a multiple of 16. Omit to use the model default.')
    p.add_argument('--apply-chat-template', action='store_true', default=True,
                   help='Wrap each prompt as one user turn and append the assistant '
                        'turn marker (default). Required for chat models: without it '
                        'a prompt ending on a complete sentence reads as a finished '
                        'document and the model emits EOS instead of answering.')
    p.add_argument('--no-chat-template', dest='apply_chat_template', action='store_false',
                   help='Send prompts as raw completion text. Use only to reproduce '
                        'pre-fix runs or to benchmark a base (non-chat) model.')
    p.add_argument('--tokenizer-path', default=None,
                   help='Tokenizer supplying the chat template. Defaults to --model-name.')
    p.add_argument('--trust-remote-code', action='store_true')
    p.add_argument('--use-async-ttft', action='store_true', default=True,
                   help='Use AsyncLLMEngine to measure TTFT (default).')
    p.add_argument('--no-async-ttft', dest='use_async_ttft', action='store_false',
                   help='Fall back to offline LLM.generate; TTFT will be None.')
    p.add_argument('--warmup-iters', type=int, default=2,
                   help='Warmup requests before the measured loop. Counted ONCE per engine.')
    p.add_argument('--reset-cache-between-cases', action='store_true', default=True,
                   help='Reset the prefix KV cache before every (base, followup) pair '
                        '(default). Prevents cross-case contamination of unrelated_control.')
    p.add_argument('--no-reset-cache-between-cases', dest='reset_cache_between_cases',
                   action='store_false',
                   help='Disable per-case cache reset. Use only to reproduce the '
                        'pre-fix behavior where cases share cache state.')

    # Multi-layout sweep. Two parallel arrays: --layouts L1 L2 ...  and
    # --mutation-jsonls P1 P2 ... must have the same length. Each (Li, Pi)
    # pair becomes one LayoutTask, all running on the same engine.
    p.add_argument('--layouts', nargs='+', required=True,
                   help='Layout names, e.g. "original stable_first".')
    p.add_argument('--mutation-jsonls', nargs='+', required=True,
                   help='Mutation JSONL paths, one per layout, same order as --layouts.')

    p.add_argument('--max-cases', type=int, default=None)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--include-relations', nargs='*',
                   default=['exact_reuse', 'partial_reuse', 'unrelated_control'])

    p.add_argument(
        '--output-dir',
        default=str(Path(__file__).resolve().parents[1] / 'outputs' / 'benchmark_results'),
    )
    p.add_argument('--run-name-template', required=True,
                   help="Output filename template containing '{layout}', e.g. "
                        "'rag_chunk_reorder_{layout}_cache_on'.")

    args = p.parse_args()
    if len(args.layouts) != len(args.mutation_jsonls):
        p.error(
            f"--layouts and --mutation-jsonls must have the same length "
            f"(got {len(args.layouts)} vs {len(args.mutation_jsonls)})."
        )

    enable_prefix_caching = True
    if args.disable_prefix_caching:
        enable_prefix_caching = False
    elif args.enable_prefix_caching:
        enable_prefix_caching = True

    layout_tasks = [
        LayoutTask(layout_name=l, mutation_jsonl_path=j)
        for l, j in zip(args.layouts, args.mutation_jsonls)
    ]

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
            max_model_len=args.max_model_len,
            apply_chat_template=args.apply_chat_template,
            tokenizer_path=args.tokenizer_path,
            warmup_iters=args.warmup_iters,
            reset_cache_between_cases=args.reset_cache_between_cases,
        ),
        dataset=DatasetConfig(
            layout_tasks=layout_tasks,
            max_cases=args.max_cases,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            include_relations=args.include_relations,
        ),
        output=OutputConfig(
            output_dir=args.output_dir,
            run_name_template=args.run_name_template,
        ),
    )


if __name__ == '__main__':
    cfg = parse_args()
    out_paths = run_benchmark(cfg)
    for p in out_paths:
        print(f'Saved benchmark results to: {p}')
