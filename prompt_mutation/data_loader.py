
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

from datasets import load_dataset, Dataset

from prompt_generator import RAGExample, ScientificExample


@dataclass
class DataLoadConfig:
    workload: str
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    split: str = "train"
    max_samples: Optional[int] = None
    shard_index: int = 0
    num_shards: int = 1
    cache_dir: Optional[str] = None
    save_processed_dir: Optional[str] = None
    load_processed_dir: Optional[str] = None


def _apply_sample_and_shard(ds: Dataset, max_samples: Optional[int], shard_index: int, num_shards: int) -> Dataset:
    if num_shards > 1:
        ds = ds.shard(num_shards=num_shards, index=shard_index, contiguous=True)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def _save_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _convert_row_to_rag_example(row: Dict[str, Any]) -> Optional[RAGExample]:
    question = None
    for key in ["question", "query", "input", "prompt"]:
        if key in row and row[key]:
            question = str(row[key]).strip()
            break
    if not question:
        return None

    chunks: List[str] = []

    for key in ["context", "document", "documents", "passage", "paragraph"]:
        if key in row and row[key]:
            value = row[key]
            if isinstance(value, str):
                chunks.append(value.strip())
            elif isinstance(value, list):
                chunks.extend([str(x).strip() for x in value if isinstance(x, str) and str(x).strip()])

    if "ctxs" in row and row["ctxs"]:
        for ctx in row["ctxs"][:4]:
            if isinstance(ctx, dict):
                txt = ctx.get("text") or ctx.get("passage") or ctx.get("content")
                if txt:
                    chunks.append(str(txt).strip())

    if "retrieved_passages" in row and row["retrieved_passages"]:
        for p in row["retrieved_passages"][:4]:
            if isinstance(p, str) and p.strip():
                chunks.append(p.strip())
            elif isinstance(p, dict):
                txt = p.get("text") or p.get("passage") or p.get("content")
                if txt:
                    chunks.append(str(txt).strip())

    chunks = [c for c in chunks if c][:3]
    if not chunks:
        return None

    return RAGExample(
        system_instruction="You are a helpful QA assistant. Use the retrieved context to answer the question.",
        user_query=question,
        retrieved_chunks=chunks,
        output_instruction="Answer briefly and ground the answer in the retrieved documents.",
    )


def _convert_row_to_scientific_example(row: Dict[str, Any]) -> Optional[ScientificExample]:
    text = None
    for key in ["question", "problem", "prompt", "input", "text"]:
        if key in row and row[key]:
            text = str(row[key]).strip()
            break
    if not text:
        return None

    return ScientificExample(
        problem_description=text,
        equation_name="Heat equation",
        parameters={"alpha": 0.1, "source_strength": 1.0},
        grid={"nx": 64, "nt": 100},
        constraints={"boundary": "u(0,t)=0, u(L,t)=0", "initial": "sin(pi x)"},
        output_schema='Return JSON with keys: "time", "grid", and "state".',
    )


def _make_synthetic_scientific_examples(num_samples: int = 100) -> List[ScientificExample]:
    templates = [
        "Simulate 1D heat diffusion in a rod of length 1.0 meters.",
        "Model a damped harmonic oscillator over time.",
        "Solve a simple advection problem on a 1D grid.",
        "Compute the temperature evolution with fixed boundary conditions.",
    ]
    out = []
    for i in range(num_samples):
        desc = templates[i % len(templates)]
        out.append(
            ScientificExample(
                problem_description=desc,
                equation_name="Heat equation" if i % 2 == 0 else "Damped oscillator",
                parameters={"alpha": round(0.1 + 0.02 * i, 4), "beta": round(0.5 + 0.1 * i, 4)},
                grid={"nx": 64 + 8 * i, "nt": 100 + 10 * i},
                constraints={"boundary": "Dirichlet", "initial": "sin(pi x)" if i % 2 == 0 else "x(0)=1, v(0)=0"},
                output_schema='Return JSON with keys: "time", "state_tensor".',
            )
        )
    return out


def _save_processed_examples(examples: List[Any], path: Path) -> None:
    rows = [asdict(ex) for ex in examples]
    _save_jsonl(rows, path)


def _load_processed_rag(path: Path) -> List[RAGExample]:
    return [RAGExample(**row) for row in _load_jsonl(path)]


def _load_processed_scientific(path: Path) -> List[ScientificExample]:
    return [ScientificExample(**row) for row in _load_jsonl(path)]


def load_examples(config: DataLoadConfig) -> List[Any]:
    if config.load_processed_dir:
        path = Path(config.load_processed_dir)
        if config.workload == "rag":
            return _load_processed_rag(path)
        if config.workload == "scientific":
            return _load_processed_scientific(path)
        raise ValueError(f"Unsupported workload: {config.workload}")

    if config.workload == "scientific" and not config.dataset_name:
        examples = _make_synthetic_scientific_examples(config.max_samples or 100)
        examples = examples[: config.max_samples] if config.max_samples else examples
        if config.num_shards > 1:
            shard_size = max(1, len(examples) // config.num_shards)
            start = config.shard_index * shard_size
            end = len(examples) if config.shard_index == config.num_shards - 1 else min(len(examples), start + shard_size)
            examples = examples[start:end]
        if config.save_processed_dir:
            _save_processed_examples(examples, Path(config.save_processed_dir))
        return examples

    ds = load_dataset(
        path=config.dataset_name,
        name=config.dataset_config_name,
        split=config.split,
        cache_dir=config.cache_dir,
    )
    ds = _apply_sample_and_shard(ds, config.max_samples, config.shard_index, config.num_shards)

    examples = []
    for row in ds:
        ex = _convert_row_to_rag_example(row) if config.workload == "rag" else _convert_row_to_scientific_example(row)
        if ex is not None:
            examples.append(ex)

    if config.save_processed_dir:
        _save_processed_examples(examples, Path(config.save_processed_dir))
    return examples
