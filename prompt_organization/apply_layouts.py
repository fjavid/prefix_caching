
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from prompt_mutation.prompt_generator import PromptRecord
from .layout_strategies import get_layout_strategy


def load_records(path: str) -> List[PromptRecord]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(PromptRecord(**json.loads(line)))
    return rows


def apply_layout(records: List[PromptRecord], strategy_name: str) -> List[PromptRecord]:
    strategy = get_layout_strategy(strategy_name)
    out = []
    for rec in records:
        if rec.workload == "rag":
            base_org = strategy.organize_rag(rec.base_prompt, mutation_type=rec.mutation_type)
            mutated_org = strategy.organize_rag(rec.mutated_prompt, mutation_type=rec.mutation_type)
        elif rec.workload == "scientific":
            base_org = strategy.organize_scientific(rec.base_prompt, mutation_type=rec.mutation_type)
            mutated_org = strategy.organize_scientific(rec.mutated_prompt, mutation_type=rec.mutation_type)
        else:
            raise ValueError(f"Unsupported workload: {rec.workload}")

        metadata = dict(rec.metadata)
        metadata["prompt_organization"] = {
            "strategy_name": strategy_name,
            "base_layout_metadata": base_org.metadata,
            "mutated_layout_metadata": mutated_org.metadata,
        }

        out.append(
            PromptRecord(
                prompt_id=rec.prompt_id,
                workload=rec.workload,
                semantic_class=rec.semantic_class,
                generation_class=rec.generation_class,
                mutation_type=rec.mutation_type,
                mutation_severity=rec.mutation_severity,
                base_prompt=base_org.prompt_text,
                mutated_prompt=mutated_org.prompt_text,
                metadata=metadata,
            )
        )
    return out


def save_records(records: List[PromptRecord], output_path: str) -> Path:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-path", required=True)
    p.add_argument("--strategy-name", required=True, choices=["original", "stable_first", "stable_first_normalized", "volatile_last"])
    p.add_argument("--output-path", required=True)
    args = p.parse_args()

    records = load_records(args.input_path)
    out_records = apply_layout(records, args.strategy_name)
    out_path = save_records(out_records, args.output_path)
    print(f"Saved organized prompts to: {out_path}")


if __name__ == "__main__":
    main()
