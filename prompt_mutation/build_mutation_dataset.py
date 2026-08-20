
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import argparse
import json

from .data_loader import DataLoadConfig, load_examples
from .llm_fn import LLMConfig, make_llm_fn
from .overlap_analyzer import OverlapAnalyzer
from .mutation_validation import ValidationConfig, validate_record
from .severity_calibration import SeverityConfig, SeverityCalibrator
from .prompt_generator import (
    save_prompt_records,
    AlgorithmicMeaningPreservingMutator,
    AlgorithmicMeaningChangingMutator,
    LLMMeaningPreservingMutator,
    LLMMeaningChangingMutator,
)


@dataclass
class BuildConfig:
    workload: str
    dataset_name: Optional[str]
    dataset_config_name: Optional[str]
    split: str
    max_samples: Optional[int]
    shard_index: int
    num_shards: int
    semantic_class: str
    generation_class: str
    mutation_type: str
    mutation_severity: float
    overlap_semantic_model_name: Optional[str]
    validation_backend: str
    validation_sentence_model_name: str
    validation_nli_model_name: str
    validation_bert_score_model_type: str
    severity_backend: str
    severity_sentence_model_name: str
    severity_nli_model_name: str
    severity_bert_score_model_type: str
    save_processed_path: Optional[str]
    load_processed_path: Optional[str]
    cache_dir: Optional[str]
    output_root: str
    llm_backend: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    llm_top_p: float
    llm_seed: Optional[int]
    min_chunks: int = 3
    max_chunks: int = 4
    max_chunk_words: int = 200

def build_dataset(config: BuildConfig) -> Path:
    load_cfg = DataLoadConfig(
        workload=config.workload,
        dataset_name=config.dataset_name,
        dataset_config_name=config.dataset_config_name,
        split=config.split,
        max_samples=config.max_samples,
        shard_index=config.shard_index,
        num_shards=config.num_shards,
        cache_dir=config.cache_dir,
        save_processed_path=config.save_processed_path,
        load_processed_path=config.load_processed_path,
        min_chunks=config.min_chunks,
        max_chunks=config.max_chunks,
        max_chunk_words=config.max_chunk_words,
        distractor_seed=config.llm_seed or 0,
    )
    examples = load_examples(load_cfg)

    if config.semantic_class == "meaning_preserving" and config.generation_class == "algorithmic":
        mutator = AlgorithmicMeaningPreservingMutator(seed=config.llm_seed or 0)
        llm_fn = None
    elif config.semantic_class == "meaning_changing" and config.generation_class == "algorithmic":
        mutator = AlgorithmicMeaningChangingMutator(seed=config.llm_seed or 0)
        llm_fn = None
    elif config.semantic_class == "meaning_preserving" and config.generation_class == "llm_generated":
        mutator = LLMMeaningPreservingMutator(seed=config.llm_seed or 0)
        llm_fn = make_llm_fn(LLMConfig(
            backend=config.llm_backend, model=config.llm_model, temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens, top_p=config.llm_top_p, seed=config.llm_seed,
        ))
    elif config.semantic_class == "meaning_changing" and config.generation_class == "llm_generated":
        mutator = LLMMeaningChangingMutator(seed=config.llm_seed or 0)
        llm_fn = make_llm_fn(LLMConfig(
            backend=config.llm_backend, model=config.llm_model, temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens, top_p=config.llm_top_p, seed=config.llm_seed,
        ))
    else:
        raise ValueError("Unsupported mutator combination.")

    analyzer = OverlapAnalyzer(semantic_model_name=config.overlap_semantic_model_name)
    validation_cfg = ValidationConfig(
        semantic_backend=config.validation_backend,
        sentence_model_name=config.validation_sentence_model_name,
        nli_model_name=config.validation_nli_model_name,
        bert_score_model_type=config.validation_bert_score_model_type,
    )
    severity_cfg = SeverityConfig(
        semantic_backend=config.severity_backend,
        sentence_model_name=config.severity_sentence_model_name,
        nli_model_name=config.severity_nli_model_name,
        bert_score_model_type=config.severity_bert_score_model_type,
    )
    severity_calibrator = SeverityCalibrator(severity_cfg)

    candidate_chunks = None
    if config.workload == "rag":
        candidate_chunks = []
        for ex in examples:
            candidate_chunks.extend(getattr(ex, "retrieved_chunks", []))

    records = []
    skipped = []
    for i, ex in enumerate(examples):
        # A single example may legitimately fail to mutate (e.g. a query
        # with no SYNONYMS hit, or a typo input where every word has only
        # duplicate adjacent chars). Skip and keep going so one bad input
        # doesn't waste an entire SLURM job.
        try:
            if config.workload == "rag":
                record = mutator.mutate_rag(
                    ex,
                    mutation_type=config.mutation_type,
                    mutation_severity=config.mutation_severity,
                    candidate_chunks=candidate_chunks,
                    llm_fn=llm_fn,
                )
            else:
                record = mutator.mutate_scientific(
                    ex,
                    mutation_type=config.mutation_type,
                    mutation_severity=config.mutation_severity,
                    llm_fn=llm_fn,
                )
        except ValueError as e:
            skipped.append({"example_index": i, "reason": str(e)})
            continue

        overlap_metrics = analyzer.analyze(record.base_prompt, record.mutated_prompt)
        validation_result = validate_record(record, validation_cfg)
        severity_result = severity_calibrator.measure(record)

        record.metadata["overlap_metrics"] = overlap_metrics.to_dict()
        record.metadata["validation"] = validation_result.to_dict()
        record.metadata["severity"] = severity_result.to_dict()
        # Reference-only. Carried so generated output can be scored downstream;
        # never rendered into base_prompt or mutated_prompt.
        reference_answers = getattr(ex, "reference_answers", None)
        if reference_answers:
            record.metadata["reference_answers"] = list(reference_answers)
        records.append(record)

    if not records:
        raise RuntimeError(
            f"Every input example failed to mutate for mutation_type="
            f"{config.mutation_type}. Sample reasons: "
            f"{[s['reason'] for s in skipped[:3]]}"
        )
    if skipped:
        print(
            f"WARNING: skipped {len(skipped)}/{len(examples)} examples that "
            f"failed to mutate (mutation_type={config.mutation_type}). "
            f"First 3 reasons: {[s['reason'] for s in skipped[:3]]}"
        )

    out_path = save_prompt_records(records, root_dir=config.output_root)
    summary = {
        "num_records": len(records),
        "num_skipped": len(skipped),
        "skipped": skipped,
        "config": asdict(config),
        "output_path": str(out_path),
    }
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path

def parse_args() -> BuildConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--workload", choices=["rag", "scientific"], required=True)
    p.add_argument("--dataset-name", default=None)
    p.add_argument("--dataset-config-name", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--max-samples", type=int, default=10)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--semantic-class", choices=["meaning_preserving", "meaning_changing"], required=True)
    p.add_argument("--generation-class", choices=["algorithmic", "llm_generated"], required=True)
    p.add_argument("--mutation-type", required=True)
    p.add_argument("--mutation-severity", type=float, default=1.0)
    p.add_argument("--overlap-semantic-model-name", default=None)
    p.add_argument("--validation-backend", default="sentence_transformer")
    p.add_argument("--validation-sentence-model-name", default="all-mpnet-base-v2")
    p.add_argument("--validation-nli-model-name", default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    p.add_argument("--validation-bert-score-model-type", default="microsoft/deberta-xlarge-mnli")
    p.add_argument("--severity-backend", default="sentence_transformer")
    p.add_argument("--severity-sentence-model-name", default="all-mpnet-base-v2")
    p.add_argument("--severity-nli-model-name", default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    p.add_argument("--severity-bert-score-model-type", default="microsoft/deberta-xlarge-mnli")
    p.add_argument("--save-processed-path", default=None,
                   help="JSONL file path to save preprocessed examples to.")
    p.add_argument("--load-processed-path", default=None,
                   help="JSONL file path to load preprocessed examples from (offline / no HF download).")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--output-root", default=str(Path(__file__).resolve().parents[1] / "outputs" / "mutation"))
    p.add_argument("--min-chunks", type=int, default=3,
                   help="Minimum retrieved_chunks per RAG example; pads with distractors if fewer.")
    p.add_argument("--max-chunks", type=int, default=4)
    p.add_argument("--max-chunk-words", type=int, default=200,
                   help="Per-chunk word cap (0 disables). Only used when loading "
                        "examples directly from HF, not when --load-processed-path is set.")
    p.add_argument("--llm-backend", default="mock")
    p.add_argument("--llm-model", default="mock-model")
    p.add_argument("--llm-temperature", type=float, default=0.0)
    p.add_argument("--llm-max-tokens", type=int, default=128)
    p.add_argument("--llm-top-p", type=float, default=1.0)
    p.add_argument("--llm-seed", type=int, default=0)
    args = p.parse_args()
    return BuildConfig(**vars(args))

if __name__ == "__main__":
    cfg = parse_args()
    path = build_dataset(cfg)
    print(f"Saved mutated dataset to: {path}")
