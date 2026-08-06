---
name: review
description: Review a scoped change — a diff, a file set, or a task batch — for blocking issues, non-blocking issues, missing tests, and readiness to commit.
allowed-tools: Read, Grep, Glob, Bash
---

# Skill: review

Use this skill after implementation and before committing. The goal is to find correctness problems, hidden risks, compatibility breaks, missing tests, and misleading docs or outputs in a **scoped** change.

For broad repository or subsystem evaluation, use `inspect`.

## When to use

Reviewing the current diff, a specific file set, or a task batch; checking whether an implementation matches its PRD; deciding whether a change is safe to commit.

## Inputs

Read the context files listed in `AGENTS.md`, the relevant `SPECS/` and `TASKS/` files, the current diff and `git status`, and the relevant source, tests, and outputs.

## Rules

- Read-only. Do not edit unless explicitly asked.
- Stay within the requested scope, and state what was not reviewed.
- Do not approve on passing tests alone. Verify behavior against requirements.
- Check docs and claims against the implementation.
- Check generated outputs for staleness or misleading content.
- If you find blockers, do not recommend commit.

## Process

1. Identify scope and boundaries.
2. Inspect `git status` and the relevant diffs.
3. Read the source spec or task, if present.
4. Check implementation logic, tests, compatibility, and failure modes.
5. Check docs and generated artifacts.
6. Run the requested tests if appropriate.
7. Summarize findings by severity and give a recommendation.

## What to look for

Incorrect logic; broken edge cases; silent failure or fail-open behavior; malformed-input handling; stale or misleading generated outputs; missing or brittle tests; API/CLI compatibility breaks; nondeterminism and reproducibility issues; performance regressions; broad unrelated refactors; accidental generated-file changes; documentation overclaims; plus the domain-specific risks listed in `AGENT_CONFIG.md`.

## Severity

**Blocking** — the change should not be committed: wrong results; misleading plots or reports; broken documented behavior; an unhandled realistic edge case; unsafe fail-open parsing; missing validation for a core claim; tests that pass without covering the important behavior; unintended file changes.

**Non-blocking** — minor clarity, optional hardening, style, nice-to-have tests, small doc improvements.

## Output format

```md
# Review

## Scope

What was reviewed, and what was not.

## Blocking issues

Each with file/behavior reference and explanation.

## Non-blocking issues

## Missing tests

Specific missing tests or validation.

## Commands and results

## Suggested fixes

Concrete repairs. If blockers exist, include a narrow `execute` fix prompt.

## Final git status

## Approval recommendation

Approved or not approved, with reason.

## Commit instructions

Only if approved and `AGENT_CONFIG.md` permits staging: the exact scoped `git add -- ...` and a commit message. Otherwise omit this section entirely.
```

## Approval standard

Approve only when: no blockers remain; validation supports the claims; scope is controlled; docs match behavior; `git status` is understood; and no generated or unrelated files are accidentally included.
