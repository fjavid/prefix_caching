
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import re
from difflib import SequenceMatcher

from prompt_generator import PromptRecord

# NLI = Natural Language Inference.
# It tests whether one text entails, contradicts, or is neutral with respect to another.
# For meaning-preserving mutations, the strongest check is bidirectional entailment:
# original -> mutated and mutated -> original.


@dataclass
class ValidationConfig:
    semantic_backend: str = "sentence_transformer"   # sentence_transformer | bert_score | nli | hybrid
    sentence_model_name: str = "all-mpnet-base-v2"
    nli_model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    bert_score_model_type: str = "microsoft/deberta-xlarge-mnli"
    use_bidirectional_nli: bool = True
    min_semantic_similarity: float = 0.88
    min_bert_f1: float = 0.88
    min_nli_entail_prob: float = 0.55
    max_similarity_for_changed: float = 0.97
    min_changed_token_ratio: float = 0.01


@dataclass
class ValidationResult:
    is_valid: bool
    semantic_class: str
    backend_used: str
    score_summary: Dict[str, float]
    rule_summary: Dict[str, Any]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_for_formatting(text: str) -> str:
    text = text.lower()
    text = re.sub(r"#+", " ", text)
    text = re.sub(r"[\-\*\_`]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def whitespace_tokens(text: str) -> List[str]:
    return text.split()


def changed_token_ratio(a: str, b: str) -> float:
    ta, tb = whitespace_tokens(a), whitespace_tokens(b)
    sa, sb = set(ta), set(tb)
    if not sa and not sb:
        return 0.0
    return 1.0 - (len(sa & sb) / len(sa | sb))


def sequence_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def parse_rag_chunks(prompt: str) -> List[str]:
    matches = re.findall(r"\[Document\s+\d+\]\n(.*?)(?=\n\n\[Document\s+\d+\]\n|\n\n[A-Z][^:\n]*:\n|\Z)", prompt, flags=re.S)
    return [m.strip() for m in matches if m.strip()]


class SemanticScorer:
    def __init__(self, config: ValidationConfig) -> None:
        self.config = config
        self.embedding_model = None
        self.nli_pipeline = None

    def _ensure_embedding_model(self) -> None:
        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.config.sentence_model_name)

    def _ensure_nli_pipeline(self) -> None:
        if self.nli_pipeline is None:
            from transformers import pipeline
            self.nli_pipeline = pipeline(
                "text-classification",
                model=self.config.nli_model_name,
                return_all_scores=True,
            )

    def sentence_transformer_similarity(self, a: str, b: str) -> float:
        self._ensure_embedding_model()
        embs = self.embedding_model.encode([a, b], normalize_embeddings=True)
        return float((embs[0] * embs[1]).sum())

    def bert_score_f1(self, a: str, b: str) -> float:
        from bert_score import score
        _, _, F1 = score(
            [b],
            [a],
            lang="en",
            model_type=self.config.bert_score_model_type,
            verbose=False,
        )
        return float(F1[0].item())

    def nli_scores(self, premise: str, hypothesis: str) -> Dict[str, float]:
        self._ensure_nli_pipeline()
        outputs = self.nli_pipeline({"text": premise, "text_pair": hypothesis})[0]
        scores = {item["label"].lower(): float(item["score"]) for item in outputs}
        entail = scores.get("entailment", scores.get("label_2", 0.0))
        contra = scores.get("contradiction", scores.get("label_0", 0.0))
        neutral = scores.get("neutral", scores.get("label_1", 0.0))
        return {"entailment": entail, "contradiction": contra, "neutral": neutral}

    def score(self, a: str, b: str, backend: Optional[str] = None) -> Dict[str, float]:
        backend = backend or self.config.semantic_backend
        out: Dict[str, float] = {}
        if backend in {"sentence_transformer", "hybrid"}:
            out["sentence_transformer_cosine"] = self.sentence_transformer_similarity(a, b)
        if backend in {"bert_score", "hybrid"}:
            out["bert_score_f1"] = self.bert_score_f1(a, b)
        if backend in {"nli", "hybrid"}:
            ab = self.nli_scores(a, b)
            out["nli_ab_entailment"] = ab["entailment"]
            out["nli_ab_contradiction"] = ab["contradiction"]
            out["nli_ab_neutral"] = ab["neutral"]
            if self.config.use_bidirectional_nli:
                ba = self.nli_scores(b, a)
                out["nli_ba_entailment"] = ba["entailment"]
                out["nli_ba_contradiction"] = ba["contradiction"]
                out["nli_ba_neutral"] = ba["neutral"]
        return out


class MeaningPreservingValidator:
    def __init__(self, config: ValidationConfig) -> None:
        self.config = config
        self.scorer = SemanticScorer(config)

    def validate(self, record: PromptRecord) -> ValidationResult:
        base, mutated = record.base_prompt, record.mutated_prompt
        mutation_type = record.mutation_type
        notes: List[str] = []
        rules: Dict[str, Any] = {}

        if mutation_type == "formatting":
            rules["normalized_sequence_ratio"] = sequence_ratio(normalize_for_formatting(base), normalize_for_formatting(mutated))
        elif mutation_type in {"chunk_reorder", "field_reorder"}:
            if record.workload == "rag":
                rules["chunk_multiset_preserved"] = sorted(parse_rag_chunks(base)) == sorted(parse_rag_chunks(mutated))
            else:
                rules["same_core_words_ratio"] = sequence_ratio(" ".join(sorted(set(whitespace_tokens(base.lower())))),
                                                               " ".join(sorted(set(whitespace_tokens(mutated.lower())))))
        elif mutation_type == "typo":
            rules["changed_token_ratio"] = changed_token_ratio(base, mutated)

        scores = self.scorer.score(base, mutated)
        valid = True
        if "sentence_transformer_cosine" in scores:
            valid &= scores["sentence_transformer_cosine"] >= self.config.min_semantic_similarity
        if "bert_score_f1" in scores:
            valid &= scores["bert_score_f1"] >= self.config.min_bert_f1
        if "nli_ab_entailment" in scores:
            valid &= scores["nli_ab_entailment"] >= self.config.min_nli_entail_prob
        if "nli_ba_entailment" in scores:
            valid &= scores["nli_ba_entailment"] >= self.config.min_nli_entail_prob
        if mutation_type == "formatting":
            valid &= rules.get("normalized_sequence_ratio", 1.0) >= 0.97
        if mutation_type == "typo":
            valid &= rules.get("changed_token_ratio", 0.0) <= 0.15
        if mutation_type == "chunk_reorder" and record.workload == "rag":
            valid &= bool(rules.get("chunk_multiset_preserved", False))

        return ValidationResult(
            is_valid=bool(valid),
            semantic_class="meaning_preserving",
            backend_used=self.config.semantic_backend,
            score_summary=scores,
            rule_summary=rules,
            notes=notes,
        )


class MeaningChangingValidator:
    def __init__(self, config: ValidationConfig) -> None:
        self.config = config
        self.scorer = SemanticScorer(config)

    def validate(self, record: PromptRecord) -> ValidationResult:
        base, mutated = record.base_prompt, record.mutated_prompt
        mutation_type = record.mutation_type
        rules: Dict[str, Any] = {}
        notes: List[str] = []
        scores = self.scorer.score(base, mutated)

        if mutation_type in {"document_replacement", "document_insertion"}:
            base_chunks = parse_rag_chunks(base)
            mutated_chunks = parse_rag_chunks(mutated)
            rules["same_chunk_multiset"] = sorted(base_chunks) == sorted(mutated_chunks)
            rules["num_base_chunks"] = len(base_chunks)
            rules["num_mutated_chunks"] = len(mutated_chunks)
        elif mutation_type == "time_reference_change":
            rules["contains_today_tomorrow_flip"] = (("today" in base.lower() and "tomorrow" in mutated.lower()) or
                                                     ("today" not in base.lower() and "tomorrow" in mutated.lower()))
        elif mutation_type == "parameter_change":
            rules["parameter_field_changed"] = str(record.metadata.get("changed_field", "")).startswith("parameters.")
        elif mutation_type == "unit_change":
            rules["contains_unit_marker"] = any(term in mutated.lower() for term in ["centimeter", "centimeters", "fahrenheit", "milliseconds"])
        elif mutation_type == "constraint_change":
            rules["constraint_field_changed"] = str(record.metadata.get("changed_field", "")).startswith("constraints.")
        elif mutation_type == "grid_change":
            rules["grid_field_changed"] = str(record.metadata.get("changed_field", "")).startswith("grid.")
        elif mutation_type in {"polarity_flip", "negation_flip", "agreement_flip"}:
            rules["contains_polarity_cue"] = self._has_polarity_shift(base, mutated)

        rules["changed_token_ratio"] = changed_token_ratio(base, mutated)

        valid = True
        if "sentence_transformer_cosine" in scores:
            valid &= scores["sentence_transformer_cosine"] <= self.config.max_similarity_for_changed or rules["changed_token_ratio"] >= self.config.min_changed_token_ratio
        if mutation_type == "document_replacement":
            valid &= not bool(rules.get("same_chunk_multiset", True))
        elif mutation_type == "document_insertion":
            valid &= rules.get("num_mutated_chunks", 0) > rules.get("num_base_chunks", 0)
        elif mutation_type == "time_reference_change":
            valid &= bool(rules.get("contains_today_tomorrow_flip", False))
        elif mutation_type == "parameter_change":
            valid &= bool(rules.get("parameter_field_changed", False))
        elif mutation_type == "unit_change":
            valid &= bool(rules.get("contains_unit_marker", False))
        elif mutation_type == "constraint_change":
            valid &= bool(rules.get("constraint_field_changed", False))
        elif mutation_type == "grid_change":
            valid &= bool(rules.get("grid_field_changed", False))
        elif mutation_type in {"polarity_flip", "negation_flip", "agreement_flip"}:
            valid &= bool(rules.get("contains_polarity_cue", False))

        return ValidationResult(
            is_valid=bool(valid),
            semantic_class="meaning_changing",
            backend_used=self.config.semantic_backend,
            score_summary=scores,
            rule_summary=rules,
            notes=notes,
        )

    def _has_polarity_shift(self, a: str, b: str) -> bool:
        pairs = [
            ("must", "must not"), ("should", "should not"), ("include", "exclude"),
            ("allow", "disallow"), ("agree", "disagree"), ("positive", "negative"),
            ("increase", "decrease"), ("maximize", "minimize"),
        ]
        al, bl = a.lower(), b.lower()
        for x, y in pairs:
            if (x in al and y in bl) or (y in al and x in bl):
                return True
        if (" not " in bl and " not " not in al) or (" not " in al and " not " not in bl):
            return True
        return False


def validate_record(record: PromptRecord, config: Optional[ValidationConfig] = None) -> ValidationResult:
    config = config or ValidationConfig()
    if record.semantic_class == "meaning_preserving":
        return MeaningPreservingValidator(config).validate(record)
    if record.semantic_class == "meaning_changing":
        return MeaningChangingValidator(config).validate(record)
    raise ValueError(f"Unsupported semantic class: {record.semantic_class}")
