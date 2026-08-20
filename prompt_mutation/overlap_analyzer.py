
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from difflib import SequenceMatcher


@dataclass
class OverlapMetrics:
    """Surface-overlap metrics between a base prompt and its mutation.

    IMPORTANT — what "token" means here.

    Every `*_token*` field below counts WHITESPACE-SEPARATED WORDS
    (`whitespace_tokenize`, i.e. `str.split()`), measured on the raw prompt text
    before any chat template is applied. They are NOT model tokens, and they are
    NOT the positions vLLM matches when reusing cached KV blocks.

    The two differ substantially and by a record-dependent amount. Measured on
    `rag_typo_stable_first` with the TinyLlama-1.1B tokenizer, the ratio
    model_tokens / whitespace_words at the divergence point ranged 1.42 to 2.62
    over 60 records (mean 1.74, stdev 0.26). One record diverged at word 514 but
    at model token 932. The chat template adds a further constant offset
    (+6 model tokens for TinyLlama), which is small next to that gap.

    Consequence: these fields are valid as a measure of *surface* similarity and
    for ranking mutations by how early they perturb the prompt. They should not
    be described as a count of cached tokens or reusable KV blocks. Any plot
    using them must label the axis as whitespace words.

    Field names are kept as-is because they are already serialized in mutation
    and layout JSONLs; the definition above is authoritative over the name.
    """
    char_shared_prefix: int
    char_shared_prefix_ratio: float
    # Whitespace-word counts. See the class docstring: not model tokens.
    token_shared_prefix: int
    token_shared_prefix_ratio: float
    first_divergence_token: int
    token_jaccard: float
    sequence_match_ratio: float
    semantic_cosine: Optional[float]
    base_num_chars: int
    mutated_num_chars: int
    base_num_tokens: int
    mutated_num_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def whitespace_tokenize(text: str) -> List[str]:
    return text.split()


def first_divergence(seq_a: List[str], seq_b: List[str]) -> int:
    n = min(len(seq_a), len(seq_b))
    for i in range(n):
        if seq_a[i] != seq_b[i]:
            return i
    return n


def shared_prefix_len_str(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def jaccard_tokens(tokens_a: List[str], tokens_b: List[str]) -> float:
    sa, sb = set(tokens_a), set(tokens_b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


class OverlapAnalyzer:
    def __init__(self, semantic_model_name: Optional[str] = None) -> None:
        self.semantic_model_name = semantic_model_name
        self.semantic_model = None
        if semantic_model_name:
            from sentence_transformers import SentenceTransformer
            self.semantic_model = SentenceTransformer(semantic_model_name)

    def semantic_similarity(self, a: str, b: str) -> Optional[float]:
        if self.semantic_model is None:
            return None
        embeddings = self.semantic_model.encode([a, b], normalize_embeddings=True)
        return float((embeddings[0] * embeddings[1]).sum())

    def analyze(self, base_prompt: str, mutated_prompt: str) -> OverlapMetrics:
        char_prefix = shared_prefix_len_str(base_prompt, mutated_prompt)
        max_chars = max(len(base_prompt), len(mutated_prompt), 1)
        base_tokens = whitespace_tokenize(base_prompt)
        mutated_tokens = whitespace_tokenize(mutated_prompt)
        tok_prefix = first_divergence(base_tokens, mutated_tokens)
        max_tokens = max(len(base_tokens), len(mutated_tokens), 1)

        return OverlapMetrics(
            char_shared_prefix=char_prefix,
            char_shared_prefix_ratio=char_prefix / max_chars,
            token_shared_prefix=tok_prefix,
            token_shared_prefix_ratio=tok_prefix / max_tokens,
            first_divergence_token=tok_prefix,
            token_jaccard=jaccard_tokens(base_tokens, mutated_tokens),
            sequence_match_ratio=SequenceMatcher(None, base_prompt, mutated_prompt).ratio(),
            semantic_cosine=self.semantic_similarity(base_prompt, mutated_prompt),
            base_num_chars=len(base_prompt),
            mutated_num_chars=len(mutated_prompt),
            base_num_tokens=len(base_tokens),
            mutated_num_tokens=len(mutated_tokens),
        )
