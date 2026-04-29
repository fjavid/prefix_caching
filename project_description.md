# Prefix Caching Under Realistic Prompt Mutations

## Goal
Build a workload-centric benchmark for LLM serving that measures how prefix caching degrades under realistic prompt mutations, identifies which mutation patterns are most harmful, and quantifies how much TTFT can be recovered through better prompt organization alone.

Prefix caching reuses precomputed KV caches for repeated prompt prefixes, reducing latency, cost, and especially time to first token (TTFT). Modern inference platforms such as vLLM, TensorRT-LLM, and SGLang emphasize prefix and cache reuse because repeated prompt structure is common in practice. However, reuse becomes much less straightforward when prompts are only partially identical. This project studies prefix caching under partial prompt overlap in workloads such as RAG pipelines, multi-turn chat, agent-style workflows, and structured scientific reasoning. The benchmark aims to measure how overlap patterns affect TTFT, end-to-end latency, throughput, GPU memory usage, and output correctness, and to identify when partial reuse becomes ineffective or harmful. It also aims to design simple prompt-layout strategies that improve TTFT and throughput for partially overlapping prompts.

## Problem framing
Standard prefix caching works well for exact shared prefixes, but real prompts often differ in small yet important ways. The key question is not only whether prompts overlap, but whether the overlap is semantically irrelevant, semantically meaningful, or positioned too late in the prompt to be useful. This project therefore studies where cacheability breaks, why it breaks, and which prompt-layout choices preserve reusable prefill work without requiring more advanced cache-composition methods.

## Workloads
The first version of the benchmark focuses on two workload families:
- **RAG-style prompts**, where the prompt contains a system instruction, user query, and retrieved context chunks.
- **Structured scientific prompts**, where the prompt contains a problem description, equation or task description, parameters, grid settings, constraints, and output schema.

These two families are intentionally different. RAG workloads stress document order, retrieval changes, and query variation, while scientific workloads stress parameter variation, units, constraints, and structured prompt fields.

## Mutation taxonomy
Prompt mutations are divided into two semantic classes.

### 1. Meaning-preserving mutations
These mutations change the form of the prompt while preserving the intended answer.
Examples include:
- typo or noise injection
- formatting changes
- template rewrites
- rewording or paraphrasing
- synonym substitution
- chunk reorder in RAG
- field reorder in scientific prompts
- boilerplate or instruction-style variation

These mutations test how fragile prefix caching is to harmless prompt variation. The expected behavior is high cache reuse with equivalent or near-equivalent outputs.

### 2. Meaning-changing mutations
These mutations legitimately change what the prompt is asking for.
Examples include:
- parameter change
- unit change
- date or time change
- retrieved-chunk replacement or insertion
- query target change
- constraint change
- grid or resolution change
- logical polarity changes such as negation flips, agreement flips, or positive/negative direction changes

These mutations test how prefix caching behaves when prompts remain structurally similar but the correct output should change.

## Step 1: Define the workload and mutation taxonomy
For each workload family, define controlled mutation types and analyze them separately. The benchmark should study one mutation family at a time first, then later include mixed mutations. This separation is important because meaning-preserving and meaning-changing mutations have different evaluation goals and should not be mixed in analysis.

## Step 2: Build the prompt generator and overlap analyzer
Implement a prompt generator that produces prompt pairs or prompt families with controlled reuse. For each pair, compute overlap statistics such as:
- shared prefix length
- first divergence position
- shared-prefix ratio
- token overlap or token Jaccard
- edit-style metrics
- optional semantic similarity scores

For RAG prompts, public QA and retrieval datasets can seed the prompt families. Useful sources include Natural Questions, HotpotQA, MuSiQue, BEIR, and FlashRAG-style processed datasets. For scientific prompts, public datasets can provide realistic problem statements, but controlled prompt families should mostly come from templates, parameter sweeps, and fixed output schemas.

Algorithmic and template-based mutations should be treated as the main ground truth because they are reproducible and exactly labeled. LLM-generated rephrasing can be added later as a separate bucket.

## Mutation validation
Mutation generation alone is not enough. Each mutation should also be validated.

For **meaning-preserving mutations**, validation can combine:
- sentence-transformer cosine similarity
- BERTScore
- optional NLI-based bidirectional entailment checks
- rule-based checks for formatting, reorder, and typo families

For **meaning-changing mutations**, validation should primarily rely on the known changed field and mutation-specific checks, such as:
- parameter-field change detection
- unit marker detection
- retrieved-chunk set differences
- today/tomorrow flips
- logical polarity cues such as `must` → `must not`, `agree` → `disagree`, or `increase` → `decrease`

## Mutation severity calibration
Each mutation should also receive severity scores from three perspectives:
- **surface severity**: sequence difference, changed-token ratio, first-divergence ratio
- **semantic severity**: embedding distance, BERTScore distance, NLI non-entailment
- **task severity**: mutation-family-specific signals such as parameter magnitude, unit shift, evidence shift, or polarity shift

This makes it possible to compare not only mutation type, but also mutation strength.

## Step 3: Build the benchmark harness
Run inference with and without prefix caching on the same prompt sets. Record TTFT, end-to-end latency, throughput, output length, cache-related reuse information if available, and GPU memory statistics. TTFT should be the primary metric because prefix caching mainly reduces repeated prefill work.

## Step 4: Add prompt-organization baselines
Design alternative prompt layouts that keep stable content early and volatile content late. Examples include:
- normalizing formatting
- preserving consistent chunk order
- isolating changing fields near the end of the prompt
- canonicalizing structured sections

Re-run the same workloads to measure how much performance can be recovered through organization alone, before using more advanced cache composition methods.

## Step 5: Analyze breakpoints and extract guidance
The final analysis should identify:
- which mutation families destroy reuse fastest
- which changes preserve reuse well
- how first-divergence position affects TTFT gains
- how much performance can be recovered through better prompt organization
- whether semantic similarity and prefix similarity move together or diverge

The benchmark should produce both system-level findings and practical design rules for building prompts that remain cache-friendly under realistic variation.

## Planned repository organization
The repository should be organized by project stage. Everything related to prompt mutation lives under the `prompt_mutation/` subdirectory. Future stages, such as inference benchmarking or prompt-organization baselines, can live in separate subdirectories with their own code and documentation.
