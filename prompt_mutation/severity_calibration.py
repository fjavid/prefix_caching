
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from difflib import SequenceMatcher

from .prompt_generator import PromptRecord
from .mutation_validation import ValidationConfig, SemanticScorer


@dataclass
class SeverityConfig:
    semantic_backend: str = "sentence_transformer"
    sentence_model_name: str = "all-mpnet-base-v2"
    nli_model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    bert_score_model_type: str = "microsoft/deberta-xlarge-mnli"


@dataclass
class SeverityResult:
    surface_severity: Dict[str, float]
    semantic_severity: Dict[str, float]
    task_severity: Dict[str, float]
    combined_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def tokenize(text: str) -> List[str]:
    return text.split()


def first_divergence_ratio(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    n = min(len(ta), len(tb))
    i = 0
    while i < n and ta[i] == tb[i]:
        i += 1
    denom = max(len(ta), len(tb), 1)
    return 1.0 - (i / denom)


def changed_token_ratio(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    sa, sb = set(ta), set(tb)
    if not sa and not sb:
        return 0.0
    return 1.0 - (len(sa & sb) / len(sa | sb))


class SeverityCalibrator:
    def __init__(self, config: Optional[SeverityConfig] = None) -> None:
        self.config = config or SeverityConfig()
        self.scorer = SemanticScorer(
            ValidationConfig(
                semantic_backend=self.config.semantic_backend,
                sentence_model_name=self.config.sentence_model_name,
                nli_model_name=self.config.nli_model_name,
                bert_score_model_type=self.config.bert_score_model_type,
            )
        )

    def measure(self, record: PromptRecord) -> SeverityResult:
        base, mutated = record.base_prompt, record.mutated_prompt
        mutation_type = record.mutation_type
        surface = {
            "sequence_diff": 1.0 - SequenceMatcher(None, base, mutated).ratio(),
            "changed_token_ratio": changed_token_ratio(base, mutated),
            "first_divergence_ratio": first_divergence_ratio(base, mutated),
        }
        semantic_scores = self.scorer.score(base, mutated, backend=self.config.semantic_backend)
        semantic = {}
        if "sentence_transformer_cosine" in semantic_scores:
            semantic["embedding_distance"] = 1.0 - semantic_scores["sentence_transformer_cosine"]
        if "bert_score_f1" in semantic_scores:
            semantic["bert_f1_distance"] = 1.0 - semantic_scores["bert_score_f1"]
        if "nli_ab_entailment" in semantic_scores:
            semantic["nli_non_entailment_ab"] = 1.0 - semantic_scores["nli_ab_entailment"]
        if "nli_ba_entailment" in semantic_scores:
            semantic["nli_non_entailment_ba"] = 1.0 - semantic_scores["nli_ba_entailment"]
        if not semantic:
            semantic["semantic_distance_fallback"] = 0.0

        task = {}
        if mutation_type == "typo":
            task["typo_magnitude"] = min(1.0, 2.5 * surface["changed_token_ratio"])
        elif mutation_type in {"template_rewrite", "synonym_substitution", "query_rephrase", "chunk_paraphrase", "problem_rephrase"}:
            task["rewrite_magnitude"] = surface["changed_token_ratio"]
        elif mutation_type in {"formatting", "chunk_reorder", "field_reorder"}:
            task["layout_magnitude"] = 0.7 * surface["changed_token_ratio"] + 0.3 * surface["first_divergence_ratio"]
        elif mutation_type == "parameter_change":
            task["parameter_delta_proxy"] = 0.4
        elif mutation_type == "unit_change":
            task["unit_shift"] = 0.6
        elif mutation_type == "constraint_change":
            task["constraint_shift"] = 0.65
        elif mutation_type == "grid_change":
            task["grid_shift"] = 0.5
        elif mutation_type == "document_replacement":
            task["evidence_shift"] = 0.75
        elif mutation_type == "document_insertion":
            task["evidence_shift"] = 0.55
        elif mutation_type == "query_target_change":
            task["target_shift"] = 0.65
        elif mutation_type == "time_reference_change":
            task["time_shift"] = 0.5
        elif mutation_type in {"polarity_flip", "negation_flip", "agreement_flip"}:
            task["logical_polarity_shift"] = 0.8
        else:
            task["generic_shift"] = surface["changed_token_ratio"]

        combined = 0.35 * self._mean(surface) + 0.35 * self._mean(semantic) + 0.30 * self._mean(task)
        return SeverityResult(surface_severity=surface, semantic_severity=semantic, task_severity=task, combined_score=float(combined))

    def _mean(self, d: Dict[str, float]) -> float:
        if not d:
            return 0.0
        return sum(d.values()) / len(d)
