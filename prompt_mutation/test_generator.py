from __future__ import annotations

from typing import List, Optional
from datasets import load_dataset

from prompt_generator import (
    PromptGenerator,
    RAGExample,
    ScientificExample,
)


def test_prompt_generator_from_dataset(
    workload: str = "rag",
    dataset_name: str = "natural_questions",
    split: str = "train",
    num_samples: int = 3,
    mutation_type: str = "document_replacement",
    seed: int = 42,
) -> None:
    """
    Sample examples from a dataset, convert them into prompt objects,
    mutate them with PromptGenerator, and print the results.

    Supported workload values:
      - "rag"
      - "scientific"

    Example:
        test_prompt_generator_from_dataset(
            workload="rag",
            dataset_name="natural_questions",
            split="train",
            num_samples=2,
            mutation_type="document_replacement",
        )
    """
    gen = PromptGenerator(seed=seed)

    if workload == "rag":
        examples = _load_rag_examples(
            dataset_name=dataset_name,
            split=split,
            num_samples=num_samples,
        )

        candidate_chunks = _collect_candidate_chunks(examples)

        for i, ex in enumerate(examples):
            record = gen.generate_rag_pair(
                base=ex,
                mutation_type=mutation_type,
                mutation_severity=1.0,
                candidate_chunks=candidate_chunks,
            )

            print(f"\n{'=' * 80}")
            print(f"RAG SAMPLE {i}")
            print(f"Mutation: {mutation_type}")
            print(f"{'-' * 80}")
            print("BASE PROMPT:\n")
            print(record.base_prompt)
            print(f"\n{'-' * 80}")
            print("MUTATED PROMPT:\n")
            print(record.mutated_prompt)
            print(f"\n{'-' * 80}")
            print("METADATA:")
            print(record.metadata)

    elif workload == "scientific":
        examples = _load_scientific_examples(
            dataset_name=dataset_name,
            split=split,
            num_samples=num_samples,
        )

        for i, ex in enumerate(examples):
            record = gen.generate_scientific_pair(
                base=ex,
                mutation_type=mutation_type,
                mutation_severity=1.0,
            )

            print(f"\n{'=' * 80}")
            print(f"SCIENTIFIC SAMPLE {i}")
            print(f"Mutation: {mutation_type}")
            print(f"{'-' * 80}")
            print("BASE PROMPT:\n")
            print(record.base_prompt)
            print(f"\n{'-' * 80}")
            print("MUTATED PROMPT:\n")
            print(record.mutated_prompt)
            print(f"\n{'-' * 80}")
            print("METADATA:")
            print(record.metadata)

    else:
        raise ValueError(f"Unsupported workload: {workload}")


def _load_rag_examples(
    dataset_name: str,
    split: str,
    num_samples: int,
) -> List[RAGExample]:
    """
    Load a small number of RAG-style examples.

    This function handles a few common dataset shapes. You will likely need
    to adapt field names for the exact dataset you choose.
    """
    ds = load_dataset(dataset_name, split=split)

    examples: List[RAGExample] = []
    max_take = min(num_samples * 5, len(ds))

    for row in ds.select(range(max_take)):
        ex = _convert_row_to_rag_example(row)
        if ex is not None:
            examples.append(ex)
        if len(examples) >= num_samples:
            break

    if not examples:
        raise RuntimeError(
            f"Could not build any RAGExample objects from dataset '{dataset_name}'. "
            "You may need to adapt _convert_row_to_rag_example()."
        )

    return examples


def _convert_row_to_rag_example(row) -> Optional[RAGExample]:
    """
    Convert one dataset row into a RAGExample.

    This supports several common field patterns:
      - question / query
      - context / document / passages
      - answer fields are ignored here because we are building prompts,
        not labels yet
    """
    question = None
    for key in ["question", "query", "input", "prompt"]:
        if key in row and row[key]:
            question = str(row[key]).strip()
            break

    if question is None:
        return None

    chunks: List[str] = []

    # Common single-context fields
    for key in ["context", "document", "documents", "passage", "paragraph"]:
        if key in row and row[key]:
            value = row[key]
            if isinstance(value, str):
                chunks.append(value.strip())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        chunks.append(item.strip())

    # Common nested / retrieved fields
    if "ctxs" in row and row["ctxs"]:
        for ctx in row["ctxs"][:4]:
            if isinstance(ctx, dict):
                text = ctx.get("text") or ctx.get("passage") or ctx.get("content")
                if text:
                    chunks.append(str(text).strip())

    if "retrieved_passages" in row and row["retrieved_passages"]:
        for p in row["retrieved_passages"][:4]:
            if isinstance(p, str) and p.strip():
                chunks.append(p.strip())
            elif isinstance(p, dict):
                text = p.get("text") or p.get("passage") or p.get("content")
                if text:
                    chunks.append(str(text).strip())

    # Fallback: if we found no chunks, skip this row
    chunks = [c for c in chunks if c]
    if not chunks:
        return None

    # Limit the number of retrieved chunks for cleaner prompts
    chunks = chunks[:3]

    return RAGExample(
        system_instruction="You are a helpful QA assistant. Use the retrieved context to answer the question.",
        user_query=question,
        retrieved_chunks=chunks,
        output_instruction="Answer briefly and ground the answer in the retrieved documents.",
    )


def _collect_candidate_chunks(examples: List[RAGExample]) -> List[str]:
    chunks: List[str] = []
    for ex in examples:
        chunks.extend(ex.retrieved_chunks)
    return chunks


def _load_scientific_examples(
    dataset_name: str,
    split: str,
    num_samples: int,
) -> List[ScientificExample]:
    """
    Load or synthesize structured scientific prompts.

    If your scientific dataset is already structured, adapt this function.
    For now, it supports a light fallback:
      - if the dataset has text fields, use them as problem descriptions
      - otherwise, synthesize examples from templates
    """
    try:
        ds = load_dataset(dataset_name, split=split)
        examples: List[ScientificExample] = []
        max_take = min(num_samples * 5, len(ds))

        for row in ds.select(range(max_take)):
            ex = _convert_row_to_scientific_example(row)
            if ex is not None:
                examples.append(ex)
            if len(examples) >= num_samples:
                break

        if examples:
            return examples

    except Exception:
        pass

    # Fallback synthetic examples
    return _make_synthetic_scientific_examples(num_samples)


def _convert_row_to_scientific_example(row) -> Optional[ScientificExample]:
    text = None
    for key in ["question", "problem", "prompt", "input", "text"]:
        if key in row and row[key]:
            text = str(row[key]).strip()
            break

    if text is None:
        return None

    return ScientificExample(
        problem_description=text,
        equation_name="Heat equation",
        parameters={"alpha": 0.1, "source_strength": 1.0},
        grid={"nx": 64, "nt": 100},
        constraints={
            "boundary": "u(0,t)=0, u(L,t)=0",
            "initial": "sin(pi x)",
        },
        output_schema='Return JSON with keys: "time", "grid", and "state".',
    )


def _make_synthetic_scientific_examples(num_samples: int) -> List[ScientificExample]:
    examples: List[ScientificExample] = []
    templates = [
        "Simulate 1D heat diffusion in a rod of length 1.0.",
        "Model a damped harmonic oscillator over time.",
        "Solve a simple advection problem on a 1D grid.",
        "Compute the temperature evolution with fixed boundary conditions.",
    ]

    for i in range(num_samples):
        desc = templates[i % len(templates)]
        examples.append(
            ScientificExample(
                problem_description=desc,
                equation_name="Heat equation" if i % 2 == 0 else "Damped oscillator",
                parameters={
                    "alpha": round(0.1 + 0.02 * i, 4),
                    "beta": round(0.5 + 0.1 * i, 4),
                },
                grid={
                    "nx": 64 + 8 * i,
                    "nt": 100 + 10 * i,
                },
                constraints={
                    "boundary": "Dirichlet",
                    "initial": "sin(pi x)" if i % 2 == 0 else "x(0)=1, v(0)=0",
                },
                output_schema='Return JSON with keys: "time", "state_tensor".',
            )
        )

    return examples