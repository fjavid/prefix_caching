# Review

## Scope

Reviewed the eight requested source files against `research.md` §§1.1–1.2 and `SPECS/model-scale-8b.md`. Read `prompt_organization/apply_layouts.py`, `prompt_organization/layout_strategies.py`, `inference_benchmark/case_builder.py`, `inference_benchmark/inference_runner.py`, and the analysis metric consumers only to trace the explicitly requested layout, case, warmup, metadata, and overlap behavior.

Shell scripts, `AGENT_CONFIG.md` changes, `RUNBOOK.md`, task batches 2–10, and unrelated worktree changes were not reviewed.

## Blocking issues

1. **The overlap fields used by analysis do not measure the token prefix seen by vLLM.** `prompt_mutation/build_mutation_dataset.py:148` computes overlap on raw prompt text, and the layout stage recomputes it with whitespace tokenization on the raw layout-rendered text. `inference_benchmark/vllm_backend.py:144` then adds the chat template before vLLM tokenizes the request. The resulting `first_divergence_token` is nevertheless plotted as “number of cached tokens.” With the real TinyLlama tokenizer, all 6,970 records in the eight current layout files shifted by exactly six model tokens after templating. Because vLLM reuses 16-token blocks, that shift added one reusable block for 34.6%–74.6% of records depending on the file. Cache-on/off TTFT remains a valid measurement, but correlations and plots against the stored raw `first_divergence_token` or `token_shared_prefix_ratio` do not describe the cache input and can misattribute block-boundary effects. `inference_benchmark/backend_base.py:86-89` also says the first-divergence position is unchanged, contradicting the observed constant shift.

2. **`reference_answers` is not associated with the request whose output it would score.** `prompt_mutation/build_mutation_dataset.py:155-159` attaches the source example’s answer once to the mutation record. That answer is valid for both prompts only for meaning-preserving mutations. It is invalid for a meaning-changing followup. In `inference_benchmark/case_builder.py:99-107`, an `unrelated_control` followup comes from another record while metadata, including `reference_answers`, comes from the base record; a scorer would therefore compare the followup output to the wrong answer. The four model-scale mutations are meaning-preserving, so their exact/partial pairs carry the correct answer, but every unrelated-control followup is still misassociated. This does not leak the answer to the model, but it would invalidate output-quality measurements that consume the field as currently named.

## Non-blocking issues

None within the requested consequential-issue threshold.

## Missing tests

- A tokenizer-specific invariant test for both supported models: template prefix/suffix are constant, the model-token first divergence shifts by a constant, and the generation marker follows the complete prompt for every layout.
- A pipeline test with distinct sentinel answers for base, mutated, and unrelated prompts, asserting per-request reference association in serialized benchmark records and absence from every engine prompt.
- The local cache contains TinyLlama but not Llama-3.1-8B-Instruct, so the latter’s actual tokenizer template was not locally exercised. The pilot must verify it before the full run.
- No real vLLM engine or GPU path was run, as prohibited. The async/offline constructor checks used stubs matching the relevant APIs.

## Commands and results

- `PYTHONPYCACHEPREFIX=tmp/review_pycache .venv/bin/python -m py_compile <all eight scoped files>` — exit 0.
- `git diff --check -- <all eight scoped files>` — no errors.
- Real TinyLlama tokenizer over all eight layout JSONLs (6,970 records) — templated first divergence was raw model-token divergence +6 for every mutation pair; the final assistant marker followed the complete rendered prompt; no embedded chat-control markers were found in the data.
- Sentinel-answer exercise over all 15 RAG mutation operators and all three layout strategies — the sentinel never appeared in a rendered prompt; it survived layout and exact/partial case construction into the benchmark result metadata.
- Legacy load of `outputs/processed/rag_examples.jsonl` — all 1,000 pre-field records deserialized; `reference_answers` defaulted to `[]`.
- Backend constructor/CLI checks — SGLang still constructs through `BackendBase`; `max_model_len=2048` reached stubbed `AsyncEngineArgs` and `LLM`; omitting it omitted the offline kwarg and retained `None` in async args; chat-template defaults and opt-out parsed correctly.
- Warmup/result exercise with a fake backend — warmup, base, and followup requests all used the same formatting path; reference metadata was serialized but never passed to `generate()`.

## Suggested fixes

1. Preserve the existing raw surface-overlap fields if they are useful, but add engine-visible metrics computed with the selected model tokenizer after applying the exact chat template: model-token prompt lengths, first divergence, shared-prefix ratio, and reusable 16-token block count. Use these fields for TTFT correlation and cache-mechanism plots. Correct the `format_prompt()` docstring to state the measured constant shift.
2. Store references per generated request. At minimum distinguish `base_reference_answers` from `followup_reference_answers`; for meaning-preserving mutations they may be equal, for meaning-changing followups the value must be explicitly unavailable unless new ground truth exists, and unrelated controls must take the followup reference from the other record.

Narrow execute prompt:

> Fix only the two blockers in `TASKS/handoffs/2026-08-19-codex-review-prereq-fixes.md`. Add engine-visible, chat-templated model-token overlap/block metrics without changing prompt text, request order, or TTFT timing. Make benchmark references explicit per base/followup request, using the other record’s answer for unrelated controls and no false answer for meaning-changing followups. Add CPU-only regression checks for both behaviors and preserve legacy processed-JSONL loading. Do not modify shell scripts or generated outputs.

## Final git status

The pre-existing worktree contains modified `AGENT_CONFIG.md`, shell scripts, the seven changed scoped Python files, and untracked documentation/task files. `inference_benchmark/sglang_backend.py` is unchanged. This review added only this handoff file; it did not stage, commit, push, or modify source/generated outputs.

## Approval recommendation

Not approved. Chat wrapping, answer non-leakage, backward deserialization, warmup formatting, SGLang construction, and both `max_model_len` paths are correct in the reviewed implementation, but the stored overlap metrics do not describe engine-visible reuse and reference metadata can score followup outputs against the wrong answer.
