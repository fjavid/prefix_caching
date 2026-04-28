
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


def _extract_rag_sections(prompt: str) -> Dict[str, Any]:
    sections: Dict[str, Any] = {
        "system_instruction": "",
        "user_query": "",
        "retrieved_chunks": [],
        "output_instruction": "",
    }

    if "\n\nUser Question:\n" in prompt:
        parts = prompt.split("\n\nUser Question:\n", 1)
        sections["system_instruction"] = parts[0].strip()
        rest = parts[1]
    else:
        rest = prompt

    if "\n\nRetrieved Context:\n" in rest:
        q, rest2 = rest.split("\n\nRetrieved Context:\n", 1)
        sections["user_query"] = q.strip()
    else:
        sections["user_query"] = rest.strip()
        rest2 = ""

    if "\n\n" in rest2:
        idx = rest2.rfind("\n\n")
        chunk_block = rest2[:idx].strip()
        sections["output_instruction"] = rest2[idx + 2 :].strip()
    else:
        chunk_block = rest2.strip()

    chunk_matches = re.findall(r"\[Document\s+\d+\]\n(.*?)(?=\n\n\[Document\s+\d+\]\n|\Z)", chunk_block, flags=re.S)
    sections["retrieved_chunks"] = [c.strip() for c in chunk_matches if c.strip()]
    return sections


def _extract_scientific_sections(prompt: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    body = prompt
    prefix = "Solve the following scientific problem."
    if body.startswith(prefix):
        body = body[len(prefix):].strip()
    pattern = re.compile(
        r"(Problem|Equation|Parameters|Grid|Constraints|Output Format):\n(.*?)(?=\n\n(?:Problem|Equation|Parameters|Grid|Constraints|Output Format):\n|\Z)",
        flags=re.S,
    )
    for label, content in pattern.findall(body):
        out[label] = content.strip()
    return out


def _render_scientific(sections: Dict[str, str], order: List[str]) -> str:
    blocks = [f"{label}:\n{sections[label].strip()}" for label in order if label in sections]
    return "Solve the following scientific problem.\n\n" + "\n\n".join(blocks)


def normalize_whitespace_and_headers(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class BaseLayoutStrategy:
    name = "base"

    def organize_rag(self, prompt: str) -> OrganizedPrompt:
        raise NotImplementedError

    def organize_scientific(self, prompt: str) -> OrganizedPrompt:
        raise NotImplementedError


class OriginalLayoutStrategy(BaseLayoutStrategy):
    name = "original"

    def organize_rag(self, prompt: str) -> OrganizedPrompt:
        return OrganizedPrompt(prompt_text=prompt, strategy_name=self.name, metadata={"changed": False})

    def organize_scientific(self, prompt: str) -> OrganizedPrompt:
        return OrganizedPrompt(prompt_text=prompt, strategy_name=self.name, metadata={"changed": False})


class StableFirstLayoutStrategy(BaseLayoutStrategy):
    name = "stable_first"

    def organize_rag(self, prompt: str) -> OrganizedPrompt:
        s = _extract_rag_sections(prompt)
        organized = "\n\n".join([
            s["system_instruction"].strip(),
            s["output_instruction"].strip(),
            f"Retrieved Context:\n" + "\n\n".join([f"[Document {i+1}]\n{c}" for i, c in enumerate(s["retrieved_chunks"])]),
            f"User Question:\n{s['user_query'].strip()}",
        ]).strip()
        return OrganizedPrompt(
            prompt_text=organized,
            strategy_name=self.name,
            metadata={"changed": True, "layout_rule": "stable scaffold first; volatile query last"},
        )

    def organize_scientific(self, prompt: str) -> OrganizedPrompt:
        s = _extract_scientific_sections(prompt)
        order = ["Equation", "Output Format", "Problem", "Parameters", "Grid", "Constraints"]
        organized = _render_scientific(s, order)
        return OrganizedPrompt(
            prompt_text=organized,
            strategy_name=self.name,
            metadata={"changed": True, "layout_rule": "equation and output schema first; volatile fields late"},
        )


class StableFirstNormalizedLayoutStrategy(BaseLayoutStrategy):
    name = "stable_first_normalized"

    def organize_rag(self, prompt: str) -> OrganizedPrompt:
        base = StableFirstLayoutStrategy().organize_rag(prompt)
        return OrganizedPrompt(
            prompt_text=normalize_whitespace_and_headers(base.prompt_text),
            strategy_name=self.name,
            metadata={"changed": True, "layout_rule": "stable-first + normalized formatting"},
        )

    def organize_scientific(self, prompt: str) -> OrganizedPrompt:
        base = StableFirstLayoutStrategy().organize_scientific(prompt)
        return OrganizedPrompt(
            prompt_text=normalize_whitespace_and_headers(base.prompt_text),
            strategy_name=self.name,
            metadata={"changed": True, "layout_rule": "stable-first + normalized formatting"},
        )


class VolatileLastLayoutStrategy(BaseLayoutStrategy):
    name = "volatile_last"

    def organize_rag(self, prompt: str) -> OrganizedPrompt:
        s = _extract_rag_sections(prompt)
        organized = "\n\n".join([
            s["system_instruction"].strip(),
            s["output_instruction"].strip(),
            "Retrieved Context:\n" + "\n\n".join([f"[Document {i+1}]\n{c}" for i, c in enumerate(s["retrieved_chunks"])]),
            f"User Question:\n{s['user_query'].strip()}",
        ]).strip()
        return OrganizedPrompt(
            prompt_text=organized,
            strategy_name=self.name,
            metadata={"changed": True, "layout_rule": "volatile query moved to final section"},
        )

    def organize_scientific(self, prompt: str) -> OrganizedPrompt:
        s = _extract_scientific_sections(prompt)
        order = ["Equation", "Problem", "Output Format", "Grid", "Parameters", "Constraints"]
        organized = _render_scientific(s, order)
        return OrganizedPrompt(
            prompt_text=organized,
            strategy_name=self.name,
            metadata={"changed": True, "layout_rule": "volatile parameter/constraint sections later"},
        )


def get_layout_strategy(name: str) -> BaseLayoutStrategy:
    strategies = {
        "original": OriginalLayoutStrategy,
        "stable_first": StableFirstLayoutStrategy,
        "stable_first_normalized": StableFirstNormalizedLayoutStrategy,
        "volatile_last": VolatileLastLayoutStrategy,
    }
    if name not in strategies:
        raise ValueError(f"Unsupported layout strategy: {name}")
    return strategies[name]()
