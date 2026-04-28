from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import json
import random

from prompt_mutation.prompt_generator import PromptRecord


@dataclass
class BenchmarkCase:
    case_id: str
    workload: str
    semantic_class: str
    mutation_type: str
    relation: str
    base_prompt: str
    followup_prompt: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_mutation_records(path: str) -> List[PromptRecord]:
    records: List[PromptRecord] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            records.append(PromptRecord(**row))
    return records


def shard_list(items: List[Any], shard_index: int, num_shards: int) -> List[Any]:
    if num_shards <= 1:
        return items
    n = len(items)
    chunk = max(1, n // num_shards)
    start = shard_index * chunk
    end = n if shard_index == num_shards - 1 else min(n, start + chunk)
    return items[start:end]


def build_cases(
    mutation_records: List[PromptRecord],
    include_relations: Optional[List[str]] = None,
    max_cases: Optional[int] = None,
    shard_index: int = 0,
    num_shards: int = 1,
    seed: int = 0,
) -> List[BenchmarkCase]:
    rng = random.Random(seed)
    include_relations = include_relations or ['exact_reuse', 'partial_reuse', 'unrelated_control']

    partial_cases: List[BenchmarkCase] = []
    for rec in mutation_records:
        partial_cases.append(BenchmarkCase(
            case_id=rec.prompt_id + '::partial',
            workload=rec.workload,
            semantic_class=rec.semantic_class,
            mutation_type=rec.mutation_type,
            relation='partial_reuse',
            base_prompt=rec.base_prompt,
            followup_prompt=rec.mutated_prompt,
            metadata=rec.metadata,
        ))

    exact_cases: List[BenchmarkCase] = []
    if 'exact_reuse' in include_relations:
        for rec in mutation_records:
            exact_cases.append(BenchmarkCase(
                case_id=rec.prompt_id + '::exact',
                workload=rec.workload,
                semantic_class=rec.semantic_class,
                mutation_type=rec.mutation_type,
                relation='exact_reuse',
                base_prompt=rec.base_prompt,
                followup_prompt=rec.base_prompt,
                metadata=rec.metadata,
            ))

    unrelated_cases: List[BenchmarkCase] = []
    if 'unrelated_control' in include_relations and len(mutation_records) > 1:
        shuffled = mutation_records[:]
        rng.shuffle(shuffled)
        for rec, other in zip(mutation_records, shuffled):
            if rec.prompt_id == other.prompt_id:
                continue
            unrelated_cases.append(BenchmarkCase(
                case_id=rec.prompt_id + '::control',
                workload=rec.workload,
                semantic_class=rec.semantic_class,
                mutation_type=rec.mutation_type,
                relation='unrelated_control',
                base_prompt=rec.base_prompt,
                followup_prompt=other.base_prompt,
                metadata=rec.metadata,
            ))

    all_cases: List[BenchmarkCase] = []
    if 'exact_reuse' in include_relations:
        all_cases.extend(exact_cases)
    if 'partial_reuse' in include_relations:
        all_cases.extend(partial_cases)
    if 'unrelated_control' in include_relations:
        all_cases.extend(unrelated_cases)

    all_cases = shard_list(all_cases, shard_index=shard_index, num_shards=num_shards)
    if max_cases is not None:
        all_cases = all_cases[:max_cases]
    return all_cases
