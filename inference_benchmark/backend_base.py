from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class GenerationResult:
    text: str
    # Time from first prompt token sent to first output token received.
    # This is the metric most directly affected by prefix caching (it's
    # prefill-dominated).
    ttft_seconds: Optional[float]
    # End-to-end wall-clock time from request submission to last output
    # token. Includes both prefill and decode. Renamed from "latency" so
    # nobody confuses it with the more specific "latency = TTFT" used in
    # serving literature.
    wall_clock_seconds: float
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BackendBase:
    # vLLM matches the prefix cache in fixed-size token blocks; a block is
    # reusable only if every token in it matches.
    BLOCK_SIZE = 16

    def __init__(
        self,
        model_name: str,
        enable_prefix_caching: bool,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = False,
        use_async_ttft: bool = True,
        max_model_len: Optional[int] = None,
        apply_chat_template: bool = True,
        tokenizer_path: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.enable_prefix_caching = enable_prefix_caching
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.use_async_ttft = use_async_ttft
        self.max_model_len = max_model_len
        self.apply_chat_template = apply_chat_template
        self.tokenizer_path = tokenizer_path or model_name
        self._tokenizer = None
        self._started = False

    def _load_chat_tokenizer(self) -> None:
        """Load the tokenizer used to wrap prompts in the model's chat format.

        Called from start(). No-op when apply_chat_template is False, or when
        the tokenizer has no chat template (base, non-chat models), in which
        case prompts are passed through unchanged.
        """
        if not self.apply_chat_template:
            return
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            self.tokenizer_path, trust_remote_code=self.trust_remote_code
        )
        if getattr(tok, "chat_template", None) is None:
            print(
                f"[{self.backend_name()}] WARNING: tokenizer at {self.tokenizer_path} "
                f"has no chat_template; prompts will be sent as raw completion text. "
                f"Expect the model to emit EOS immediately when a prompt ends on a "
                f"complete sentence."
            )
            return
        self._tokenizer = tok

    def format_prompt(self, prompt: str) -> str:
        """Wrap a rendered prompt in the model's chat format.

        The whole prompt — every section, in whatever order the layout strategy
        produced — becomes a single user turn, and `add_generation_prompt=True`
        appends the assistant turn marker.

        This does not disturb the prefix-caching measurement. The leading
        markers are identical for every prompt, so they extend the shared prefix
        equally for a base prompt and its mutated counterpart, and the trailing
        assistant marker sits after the divergence point.

        First-divergence position shifts by a constant: measured at +6 model
        tokens for TinyLlama-1.1B-Chat across 6970 records in all eight layout
        files, with zero variance. The shift is the length of the template
        header, so the relative divergence position is preserved even though the
        absolute index moves.

        Note that the overlap metrics stored by the mutation and layout stages
        count WHITESPACE WORDS on the un-templated text, not model tokens. They
        are therefore not the token positions vLLM matches against. See
        prompt_mutation/overlap_analyzer.py.

        Without this, a prompt ending on a complete sentence (e.g. the RAG
        output instruction) reads as a finished document and the model emits
        EOS on the first token.
        """
        if self._tokenizer is None:
            return prompt
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def token_overlap(self, base_prompt: str, followup_prompt: str) -> Dict[str, Any]:
        """Model-token overlap between the two prompts a case actually sends.

        This is the quantity vLLM's prefix cache operates on, measured after the
        chat template is applied and with the model's own tokenizer. It is the
        engine-visible counterpart to the whitespace-word metrics recorded by
        prompt_mutation/overlap_analyzer.py, which undercount by a
        record-dependent factor and are therefore not usable as cached-token
        counts.

        `reusable_blocks` is the number of 16-token blocks fully shared before
        the first divergence. A block is reused only if all 16 tokens match, so
        a divergence partway into a block discards that whole block.

        Returns an empty dict when no tokenizer is loaded, so callers must treat
        the fields as optional.
        """
        if self._tokenizer is None:
            return {}
        base_ids = self._tokenizer(self.format_prompt(base_prompt)).input_ids
        followup_ids = self._tokenizer(self.format_prompt(followup_prompt)).input_ids
        divergence = min(len(base_ids), len(followup_ids))
        for i, (a, b) in enumerate(zip(base_ids, followup_ids)):
            if a != b:
                divergence = i
                break
        longest = max(len(base_ids), len(followup_ids))
        return {
            'base_prompt_tokens': len(base_ids),
            'followup_prompt_tokens': len(followup_ids),
            'first_divergence_model_token': divergence,
            'shared_prefix_model_token_ratio': (
                divergence / longest if longest else None
            ),
            'reusable_blocks': divergence // self.BLOCK_SIZE,
            'block_size': self.BLOCK_SIZE,
        }

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def restart(self) -> None:
        self.stop()
        self.start()

    def generate(self, prompt: str, request_id: Optional[str] = None) -> GenerationResult:
        raise NotImplementedError

    def reset_prefix_cache(self) -> bool:
        """Flush the engine's prefix KV cache between cases.

        Subclasses should override. Returns True if the reset actually
        happened, False if the backend has no working reset path (in which
        case the caller may want to log a warning since experimental
        isolation between cases is then NOT guaranteed).
        """
        return False

    def backend_name(self) -> str:
        raise NotImplementedError
