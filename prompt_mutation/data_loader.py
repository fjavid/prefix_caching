
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import random

from datasets import load_dataset, Dataset

from .prompt_generator import RAGExample, ScientificExample


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
    # File path (NOT a directory). The processed examples are stored as one JSONL file.
    save_processed_path: Optional[str] = None
    load_processed_path: Optional[str] = None
    # Minimum number of retrieved chunks per RAG example.
    # If a row provides fewer chunks, we pad with distractors sampled from other rows.
    # Set to 1 to disable padding.
    min_chunks: int = 3
    # Hard cap on the number of retrieved chunks (after padding).
    max_chunks: int = 4
    # RNG seed for distractor sampling.
    distractor_seed: int = 0
    # Truncate every retrieved chunk to at most this many whitespace-separated words.
    # Prevents one long Wikipedia paragraph from blowing the model's context window.
    # Set to 0 to disable truncation.
    max_chunk_words: int = 200


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


def _truncate_to_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _extract_chunks_from_row(row: Dict[str, Any], max_chunk_words: int = 0) -> List[str]:
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
    chunks = [c for c in chunks if c]
    if max_chunk_words > 0:
        chunks = [_truncate_to_words(c, max_chunk_words) for c in chunks]
    return chunks


def _convert_row_to_rag_example(row: Dict[str, Any], max_chunks: int,
                                max_chunk_words: int = 0) -> Optional[RAGExample]:
    question = None
    for key in ["question", "query", "input", "prompt"]:
        if key in row and row[key]:
            question = str(row[key]).strip()
            break
    if not question:
        return None

    chunks = _extract_chunks_from_row(row, max_chunk_words=max_chunk_words)[:max_chunks]
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


def _pad_rag_examples_with_distractors(
    examples: List[RAGExample],
    min_chunks: int,
    max_chunks: int,
    seed: int,
) -> List[RAGExample]:
    """Ensure every RAG example has between min_chunks and max_chunks retrieved chunks.

    Distractors are sampled from chunks belonging to OTHER examples in the slice,
    so chunk_reorder / document_replacement / document_insertion become non-trivial.
    """
    if min_chunks <= 1:
        return examples

    rng = random.Random(seed)

    # Per-example chunk pool sourced from other examples (avoid leaking own chunks as distractors).
    all_chunks: List[str] = []
    per_example_chunks: List[List[str]] = []
    for ex in examples:
        per_example_chunks.append(list(ex.retrieved_chunks))
        all_chunks.extend(ex.retrieved_chunks)

    padded: List[RAGExample] = []
    for i, ex in enumerate(examples):
        own_set = set(per_example_chunks[i])
        pool = [c for c in all_chunks if c not in own_set]
        if not pool:
            padded.append(ex)
            continue
        need = max(0, min_chunks - len(ex.retrieved_chunks))
        if need == 0:
            chunks = ex.retrieved_chunks[:max_chunks]
        else:
            sampled = rng.sample(pool, min(need, len(pool)))
            chunks = (ex.retrieved_chunks + sampled)[:max_chunks]
        padded.append(
            RAGExample(
                system_instruction=ex.system_instruction,
                user_query=ex.user_query,
                retrieved_chunks=chunks,
                output_instruction=ex.output_instruction,
            )
        )
    return padded


def load_examples(config: DataLoadConfig) -> List[Any]:
    if config.load_processed_path:
        path = Path(config.load_processed_path)
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
        if config.save_processed_path:
            _save_processed_examples(examples, Path(config.save_processed_path))
        return examples

    ds = load_dataset(
        path=config.dataset_name,
        name=config.dataset_config_name,
        split=config.split,
        cache_dir=config.cache_dir,
    )
    ds = _apply_sample_and_shard(ds, config.max_samples, config.shard_index, config.num_shards)

    examples: List[Any] = []
    for row in ds:
        if config.workload == "rag":
            ex = _convert_row_to_rag_example(
                row,
                max_chunks=config.max_chunks,
                max_chunk_words=config.max_chunk_words,
            )
        else:
            ex = _convert_row_to_scientific_example(row)
        if ex is not None:
            examples.append(ex)

    if config.workload == "rag" and config.min_chunks > 1:
        examples = _pad_rag_examples_with_distractors(
            examples,
            min_chunks=config.min_chunks,
            max_chunks=config.max_chunks,
            seed=config.distractor_seed,
        )

    if config.save_processed_path:
        _save_processed_examples(examples, Path(config.save_processed_path))
    return examples
