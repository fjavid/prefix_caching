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


def _reference_answers(rec: PromptRecord) -> List[str]:
    """Ground-truth answer(s) for a record's BASE prompt."""
    return list(rec.metadata.get('reference_answers') or [])


def _case_metadata(
    rec: PromptRecord,
    followup_answers: Optional[List[str]],
    followup_note: Optional[str] = None,
) -> Dict[str, Any]:
    """Copy a record's metadata and attach per-request reference answers.

    The mutation stage stores a single `reference_answers` field, which is the
    ground truth for the BASE prompt only. That is not sufficient here: a case
    issues two requests, and for two of the three relations the followup prompt
    is not the base prompt.

    - exact_reuse       followup IS the base prompt, so the same answer applies.
    - partial_reuse     followup is the mutated prompt. For a meaning-preserving
                        mutation the answer is unchanged by construction. For a
                        meaning-changing mutation the base answer is wrong, and
                        no ground truth exists for the mutated prompt, so the
                        followup answer is None with a stated reason.
    - unrelated_control followup is a DIFFERENT record's base prompt, so the
                        answer must come from that record.

    The ambiguous single-valued `reference_answers` key is removed so there is
    one source of truth per request.
    """
    metadata = dict(rec.metadata)
    metadata.pop('reference_answers', None)
    metadata['base_reference_answers'] = _reference_answers(rec)
    metadata['followup_reference_answers'] = (
        list(followup_answers) if followup_answers is not None else None
    )
    if followup_note:
        metadata['followup_reference_note'] = followup_note
    return metadata


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
    degenerate: List[str] = []
    for rec in mutation_records:
        if rec.base_prompt == rec.mutated_prompt:
            degenerate.append(rec.prompt_id)
            continue
        # A meaning-preserving mutation leaves the correct answer unchanged by
        # definition. A meaning-changing mutation invalidates it, and the
        # pipeline has no ground truth for the mutated prompt.
        if rec.semantic_class == 'meaning_preserving':
            followup_answers = _reference_answers(rec)
            followup_note = None
        else:
            followup_answers = None
            followup_note = (
                'meaning_changing mutation: the base answer does not apply to '
                'the mutated prompt and no ground truth exists for it'
            )
        partial_cases.append(BenchmarkCase(
            case_id=rec.prompt_id + '::partial',
            workload=rec.workload,
            semantic_class=rec.semantic_class,
            mutation_type=rec.mutation_type,
            relation='partial_reuse',
            base_prompt=rec.base_prompt,
            followup_prompt=rec.mutated_prompt,
            metadata=_case_metadata(rec, followup_answers, followup_note),
        ))
    if degenerate:
        raise ValueError(
            f"{len(degenerate)} mutation record(s) have base_prompt == mutated_prompt "
            f"(first: {degenerate[0]}). Fix the mutation step before benchmarking."
        )

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
                # Followup is the base prompt verbatim.
                metadata=_case_metadata(rec, _reference_answers(rec)),
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
                # The followup prompt is `other`'s base prompt, so its answer
                # must come from `other`, not from `rec`.
                metadata=_case_metadata(rec, _reference_answers(other)),
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
