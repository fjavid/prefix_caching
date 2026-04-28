# Prompt Mutation Module

This directory contains the code and artifacts for the prompt-mutation stage of the prefix-caching benchmark.

## Purpose
The goal of this module is to generate controlled prompt mutations, measure how much overlap they preserve, validate whether they are meaning-preserving or meaning-changing, and assign severity scores before the prompts are passed to the serving benchmark.

## Directory contents
- `prompt_generator.py`: defines prompt structures and mutation operators for RAG and structured scientific prompts.
- `data_loader.py`: loads raw datasets or synthetic examples, supports partial loading, sharding, and local processed caches.
- `llm_fn.py`: provides a unified interface for mock, OpenAI, or local Hugging Face LLM backends.
- `overlap_analyzer.py`: computes textual, token-level, and optional embedding-based overlap metrics.
- `mutation_validation.py`: validates mutations using rule-based checks plus semantic validation backends.
- `severity_calibration.py`: computes surface, semantic, and task-specific severity scores.
- `build_mutation_dataset_v2.py`: end-to-end dataset builder that generates mutations and attaches overlap, validation, and severity metadata.
- `test_validation_severity.py`: post-processing script to validate an existing mutation dataset and add severity scores.
- `test_mutations_v2.ipynb`: interactive notebook for testing the workflow.

## Typical workflow
1. Load a dataset shard or synthetic examples with `data_loader.py`.
2. Generate mutations with `prompt_generator.py`.
3. Measure overlap with `overlap_analyzer.py`.
4. Validate semantic class with `mutation_validation.py`.
5. Measure mutation severity with `severity_calibration.py`.
6. Save the final mutation dataset with `build_mutation_dataset_v2.py`.

## Example commands
Generate a small scientific mutation dataset:

```bash
python build_mutation_dataset_v2.py \
  --workload scientific \
  --semantic-class meaning_changing \
  --generation-class algorithmic \
  --mutation-type parameter_change \
  --max-samples 20
```

Generate a small RAG mutation dataset from a Hugging Face dataset:

```bash
python build_mutation_dataset_v2.py \
  --workload rag \
  --dataset-name natural_questions \
  --split train[:200] \
  --semantic-class meaning_preserving \
  --generation-class algorithmic \
  --mutation-type formatting \
  --max-samples 20
```

Validate an existing mutation dataset and add validation/severity fields:

```bash
python test_validation_severity.py \
  --input-path ../data/mutation/scientific/meaning_changing/algorithmic/parameter_change/<file>.jsonl
```

## Outputs
Generated datasets are stored under `../data/mutation/<workload>/<semantic_class>/<generation_class>/<mutation_type>/`.
Each record contains:
- the base prompt
- the mutated prompt
- mutation metadata
- overlap metrics
- validation results
- severity scores

## Notes
- For the core benchmark, algorithmic mutations should be treated as the primary ground truth.
- LLM-generated mutations are supported, but should be analyzed in separate buckets.
- PyTorch is intentionally not pinned in `requirements.txt`; install the version that matches your CUDA setup.
