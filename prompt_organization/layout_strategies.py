
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import re


@dataclass
class OrganizedPrompt:
    prompt_text: str
    strategy_name: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Mutation → volatile-section mapping.
#
# vLLM's prefix cache reuses tokens from position 0 up to the first divergence
# between two requests. So the only way a layout can lengthen the shared prefix
# is to PUSH THE VOLATILE SECTION (the one that the mutation changes) TOWARD
# THE END of the prompt. Everything before it then forms one long shared
# prefix that the cache can reuse.
#
# RAG section ids: "system", "query", "chunks", "output"
# Scientific section ids: "problem", "equation", "parameters", "grid",
#                          "constraints", "output_schema"
#
# A value of None means "no single section is volatile" (e.g. field_reorder
# permutes the whole prompt). In that case stable_first falls back to a
# generic ordering rather than trying to push something specific last.
# ---------------------------------------------------------------------------

RAG_VOLATILE_SECTION: Dict[str, Optional[str]] = {
    # query-targeted mutations
    "typo": "query",
    "synonym_substitution": "query",
    "query_target_change": "query",
    "time_reference_change": "query",
    "negation_flip": "query",
    "agreement_flip": "query",
    "polarity_flip": "query",
    "query_rephrase": "query",
    "query_change": "query",
    # chunk-targeted mutations
    "chunk_reorder": "chunks",
    "document_replacement": "chunks",
    "document_insertion": "chunks",
    "chunk_paraphrase": "chunks",
    # output-instruction-targeted mutations
    "formatting": "output",
    "template_rewrite": "output",
}

SCIENTIFIC_VOLATILE_SECTION: Dict[str, Optional[str]] = {
    # problem-description-targeted mutations
    "typo": "problem",
    "synonym_substitution": "problem",
    "template_rewrite": "problem",
    "problem_rephrase": "problem",
    "scenario_change": "problem",
    "unit_change": "problem",
    "negation_flip": "problem",
    "agreement_flip": "problem",
    "polarity_flip": "problem",
    # field-specific mutations
    "parameter_change": "parameters",
    "constraint_change": "constraints",
    "grid_change": "grid",
    "formatting": "output_schema",
    # whole-prompt permutation; no single volatile section
    "field_reorder": None,
}


# ---------------------------------------------------------------------------
# RAG section parser/renderer.
# ---------------------------------------------------------------------------

RAG_DEFAULT_ORDER: List[str] = ["system", "query", "chunks", "output"]


def _extract_rag_sections(prompt: str) -> Dict[str, Any]:
    sections: Dict[str, Any] = {
        "system": "",
        "query": "",
        "chunks": [],
        "output": "",
    }

    rest = prompt
    if "\n\nUser Question:\n" in rest:
        head, rest = rest.split("\n\nUser Question:\n", 1)
        sections["system"] = head.strip()

    if "\n\nRetrieved Context:\n" in rest:
        q, rest = rest.split("\n\nRetrieved Context:\n", 1)
        sections["query"] = q.strip()
    else:
        sections["query"] = rest.strip()
        rest = ""

    if rest:
        idx = rest.rfind("\n\n")
        if idx != -1:
            chunk_block = rest[:idx].strip()
            sections["output"] = rest[idx + 2:].strip()
        else:
            chunk_block = rest.strip()
        chunk_matches = re.findall(
            r"\[Document\s+\d+\]\n(.*?)(?=\n\n\[Document\s+\d+\]\n|\Z)",
            chunk_block,
            flags=re.S,
        )
        sections["chunks"] = [c.strip() for c in chunk_matches if c.strip()]
    return sections


def _render_rag_section(name: str, sections: Dict[str, Any]) -> str:
    if name == "system":
        return sections["system"]
    if name == "query":
        return f"User Question:\n{sections['query']}"
    if name == "chunks":
        chunks_text = "\n\n".join(
            [f"[Document {i+1}]\n{c}" for i, c in enumerate(sections["chunks"])]
        )
        return f"Retrieved Context:\n{chunks_text}"
    if name == "output":
        return sections["output"]
    raise ValueError(f"Unknown RAG section: {name}")


def _render_rag(sections: Dict[str, Any], order: List[str]) -> str:
    parts: List[str] = []
    for name in order:
        block = _render_rag_section(name, sections)
        if block:
            parts.append(block)
    return "\n\n".join(parts).strip()


def _rag_order_for_volatile(volatile: Optional[str]) -> List[str]:
    """Default order with `volatile` moved to the end. If volatile is None or
    not a known section, returns the default order unchanged."""
    if not volatile or volatile not in RAG_DEFAULT_ORDER:
        return list(RAG_DEFAULT_ORDER)
    return [s for s in RAG_DEFAULT_ORDER if s != volatile] + [volatile]


# ---------------------------------------------------------------------------
# Scientific section parser/renderer.
# ---------------------------------------------------------------------------

SCIENTIFIC_DEFAULT_ORDER: List[str] = [
    "problem", "equation", "parameters", "grid", "constraints", "output_schema",
]

SCIENTIFIC_LABELS: Dict[str, str] = {
    "problem": "Problem",
    "equation": "Equation",
    "parameters": "Parameters",
    "grid": "Grid",
    "constraints": "Constraints",
    "output_schema": "Output Format",
}


def _extract_scientific_sections(prompt: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    body = prompt
    prefix = "Solve the following scientific problem."
    if body.startswith(prefix):
        body = body[len(prefix):].strip()
    pattern = re.compile(
        r"(Problem|Equation|Parameters|Grid|Constraints|Output Format):\n(.*?)"
        r"(?=\n\n(?:Problem|Equation|Parameters|Grid|Constraints|Output Format):\n|\Z)",
        flags=re.S,
    )
    label_to_id = {v: k for k, v in SCIENTIFIC_LABELS.items()}
    for label, content in pattern.findall(body):
        out[label_to_id[label]] = content.strip()
    return out


def _render_scientific(sections: Dict[str, str], order: List[str]) -> str:
    blocks = []
    for name in order:
        if name in sections:
            label = SCIENTIFIC_LABELS[name]
            blocks.append(f"{label}:\n{sections[name].strip()}")
    return "Solve the following scientific problem.\n\n" + "\n\n".join(blocks)


def _scientific_order_for_volatile(volatile: Optional[str]) -> List[str]:
    if not volatile or volatile not in SCIENTIFIC_DEFAULT_ORDER:
        return list(SCIENTIFIC_DEFAULT_ORDER)
    return [s for s in SCIENTIFIC_DEFAULT_ORDER if s != volatile] + [volatile]


# ---------------------------------------------------------------------------
# Strategies.
# ---------------------------------------------------------------------------


class BaseLayoutStrategy:
    name = "base"

    def organize_rag(self, prompt: str, mutation_type: Optional[str] = None) -> OrganizedPrompt:
        raise NotImplementedError

    def organize_scientific(self, prompt: str, mutation_type: Optional[str] = None) -> OrganizedPrompt:
        raise NotImplementedError


class OriginalLayoutStrategy(BaseLayoutStrategy):
    name = "original"

    def organize_rag(self, prompt: str, mutation_type: Optional[str] = None) -> OrganizedPrompt:
        return OrganizedPrompt(prompt_text=prompt, strategy_name=self.name, metadata={"changed": False})

    def organize_scientific(self, prompt: str, mutation_type: Optional[str] = None) -> OrganizedPrompt:
        return OrganizedPrompt(prompt_text=prompt, strategy_name=self.name, metadata={"changed": False})


class StableFirstLayoutStrategy(BaseLayoutStrategy):
    """Mutation-aware "stable-first" layout.

    For each mutation type, identifies the section whose content the mutation
    actually changes (the volatile section), and renders the prompt with that
    section moved to the END. All other sections retain their canonical order
    at the front, forming a long shared prefix between the base and mutated
    prompts that the prefix cache can reuse.

    When `mutation_type` is unknown or maps to None (e.g. field_reorder, where
    the WHOLE prompt is volatile), the default canonical order is used.
    """

    name = "stable_first"

    def organize_rag(self, prompt: str, mutation_type: Optional[str] = None) -> OrganizedPrompt:
        sections = _extract_rag_sections(prompt)
        volatile = RAG_VOLATILE_SECTION.get(mutation_type) if mutation_type else None
        order = _rag_order_for_volatile(volatile)
        text = _render_rag(sections, order)
        return OrganizedPrompt(
            prompt_text=text,
            strategy_name=self.name,
            metadata={
                "changed": order != list(RAG_DEFAULT_ORDER),
                "mutation_type": mutation_type,
                "volatile_section": volatile,
                "order": order,
                "layout_rule": "volatile section last; stable sections first in canonical order",
            },
        )

    def organize_scientific(self, prompt: str, mutation_type: Optional[str] = None) -> OrganizedPrompt:
        sections = _extract_scientific_sections(prompt)
        volatile = SCIENTIFIC_VOLATILE_SECTION.get(mutation_type) if mutation_type else None
        order = _scientific_order_for_volatile(volatile)
        text = _render_scientific(sections, order)
        return OrganizedPrompt(
            prompt_text=text,
            strategy_name=self.name,
            metadata={
                "changed": order != list(SCIENTIFIC_DEFAULT_ORDER),
                "mutation_type": mutation_type,
                "volatile_section": volatile,
                "order": order,
                "layout_rule": "volatile section last; stable sections first in canonical order",
            },
        )


class VolatileLastLayoutStrategy(BaseLayoutStrategy):
    """Alias for stable_first kept for backward compatibility with old configs."""

    name = "volatile_last"

    def organize_rag(self, prompt: str, mutation_type: Optional[str] = None) -> OrganizedPrompt:
        org = StableFirstLayoutStrategy().organize_rag(prompt, mutation_type=mutation_type)
        return OrganizedPrompt(prompt_text=org.prompt_text, strategy_name=self.name, metadata=org.metadata)

    def organize_scientific(self, prompt: str, mutation_type: Optional[str] = None) -> OrganizedPrompt:
        org = StableFirstLayoutStrategy().organize_scientific(prompt, mutation_type=mutation_type)
        return OrganizedPrompt(prompt_text=org.prompt_text, strategy_name=self.name, metadata=org.metadata)


def get_layout_strategy(name: str) -> BaseLayoutStrategy:
    strategies = {
        "original": OriginalLayoutStrategy,
        "stable_first": StableFirstLayoutStrategy,
        "volatile_last": VolatileLastLayoutStrategy,
    }
    if name not in strategies:
        raise ValueError(f"Unsupported layout strategy: {name}")
    return strategies[name]()
