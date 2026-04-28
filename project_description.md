# Prefix caching under realistic chat workloads
Build a workload-centric benchmark for LLM serving that measures how prefix caching degrades under realistic prompt mutations, identifies which mutation patterns are most harmful, and quantifies how much TTFT can be recovered through better prompt organization alone.

Prefix caching reuses precomputed KV caches for repeated prompt prefixes, reducing latency, cost, and especially time to first token (TTFT). Modern inference platforms such as vLLM, TensorRT-LLM, and SGLang emphasize prefix and cache reuse because repeated prompt structure is common in practice. However, reuse becomes much less straightforward when prompts are only partially identical. In this project, we study prefix caching under partial prompt overlap in workloads such as agent pipelines, RAG systems, multi-turn chat, and scientific reasoning. We aim to measure how different overlap patterns affect TTFT, end-to-end latency, throughput, GPU memory usage, and output correctness, and to identify when partial reuse becomes ineffective or leads to degraded generations. We also aim to design simple prompt-layout strategies that improve TTFT and throughput for partially overlapping prompts.


## Problem framing:
Standard prefix caching works well for exact shared prefixes, but real prompts in chat, RAG, agents, and structured workflows often differ in small but important ways. This project studies where cacheability breaks, why it breaks, and which prompt-layout choices preserve reusable prefill work without requiring more advanced cache-composition methods.

## Steps
1. Define the workload and mutation taxonomy
We focus on RAG-style prompts, and structured scientific prompts. For each case, we define controlled mutation types and study the effect of each mutation type and value on the prefix caching. In addition we classify the prompt mutation into two categories, 
1) Class A: meaning-preserving mutations: 
These are prompts that are different at the token level but are intended to produce the same answer or essentially the same answer. This includes misspelling, paraphrasing, rewording, formatting changes, and chunk reorder when the information content is unchanged.

2) Class B: meaning-changing mutations: 
These are prompts that legitimately ask for a different answer. This includes changing units, changing parameters, changing retrieved evidence, changing dates like “today” vs “tomorrow,” or changing boundary conditions in scientific prompts.

Class A (meaning-preserving) category tells us how fragile prefix caching is to harmless irrelevant and semantically identical changes while Class B (meaning-changing) category can tell us how prefix caching behaves when prompts share the same structure but are semantically different. The general consensus for meaning-preserving mutations is to have high cache resuse with identical or equivalent outputs. Any output degaradation should be treated as a negative point. For meaning-changing mutations, the idea is to reuse cache if possible but the output is supposed to be different, hence the correctness of the results should be evaluated based on the mutated prompt. 

For meaning-preserving mutations, use techniques that change form but not intent:

typo/noise injection: small misspellings, dropped punctuation, spacing changes
formatting changes: bullets vs paragraph, section headers, JSON-style labels, extra separators
template rewrites: “Answer briefly” → “Provide a concise response”
query rewording: same question with different wording
sentence-level paraphrase: rewrite one instruction or one context chunk without changing content
chunk reorder: reorder retrieved chunks when order should not matter
field reorder in scientific prompts: parameters before equation, or constraints before grid
synonym substitution: replace local words with close synonyms
boilerplate variation: change the assistant role text or output instruction wording while preserving task

For meaning-changing mutations, use techniques that should legitimately change the answer:

parameter change: coefficients, initial values, thresholds
unit change: meters to centimeters, Celsius to Fahrenheit
date/time change: today vs tomorrow, 2024 vs 2025
constraint change: different boundary condition, conservation rule, allowed assumptions
retrieved-chunk replacement: swap one supporting document with a different one
retrieved-chunk insertion/deletion: add or remove evidence
query target change: one entity, location, or variable changes
schema/output change with semantic effect: ask for max instead of average, summary instead of exact value
resolution/grid change in scientific prompts when the requested output tensor shape changes
scenario change: same setup, different operating condition or regime

2. Build the prompt generator and overlap analyzer
Implement a generator that produces prompt pairs or families with controlled reuse. Add utilities that measure shared prefix length, first divergence point, total token overlap, and mutation type.
For RAG-style prompts, the easiest path is to build on public QA and retrieval datasets. FlashRAG provides many preprocessed RAG benchmarks. There are several common sources such as Natural Questions, HotpotQA, MuSiQue, and BEIR which can be used as retrieval corpora with diverse domains. 
For structured scientific prompts, however, the public benchmarks are not enough. Datasets like Turing-Open-Reasoning and WildSci can give us realistic science problem statements but they do not provide controlled prompt structure. We hence need to generate our own prompt families from templates, parameter sweeps, and fixed schemas, so we know exactly what changed and where. 
We can also use LLMs to generate variants of prompts however, algorithmic and template-based mutations will be the ground truth here because they are reproducible and exact. LLM-paraphrasing can be added separately to the datasets in different buckets.
As metrics for measuring the mutations, we can focus on token-prefix, edit-style and semantic metrics. Token-prefix metrics can be parameters such as shared prefix length, first divergence position, shared-prifix ratio and many more. Edit-style metrics can be formulate as token edit distance, simple diff stattistics, similarly to describe how much the prompt changes structurally. Semantic similarity metrics measure the embeding cosine similarity with tools such as SentenceTransformers, which are specifically designed for semantic textual similarity and embedding-based similarity scoring.

3. Build the benchmark harness
Run inference with and without prefix caching on the same prompt sets. Record TTFT, end-to-end latency, throughput, and memory-related statistics, with TTFT as the primary metric.

4. Add prompt-organization baselines
Create alternative prompt layouts that keep stable content early and volatile content late, normalize formatting, and preserve chunk order. Re-run the same workloads to measure how much performance is recovered through organization alone.

5. Analyze breakpoints and extract guidance
Plot TTFT benefit versus first divergence position and workload type. Summarize which mutations destroy reuse fastest, which preserve it, and which prompt-design rules consistently recover substantial benefit.