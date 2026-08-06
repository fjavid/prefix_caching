---
name: execute
description: Implement one scoped vertical batch of code changes with tests, validation, and a concise handoff. Use for coding, debugging, refactoring, and requested file changes.
---

# Skill: execute

Use this skill when the user wants code or file changes. The goal is to implement one clearly scoped vertical slice, validate it, and return a precise report.

## When to use

Implementing a task batch; fixing a specific bug; adding tests; refactoring a scoped component; updating documentation tied to a code change; creating or modifying scripts.

Do not use `execute` for broad planning. Use `grill` or `plan` first if requirements are unclear.

## Inputs

Read the context files listed in `AGENTS.md`, the relevant `SPECS/` and `TASKS/` files, current `git status`, and the relevant source, tests, and outputs.

## Rules

- Stay within the requested scope; one vertical batch at a time.
- Do not edit unrelated files or generated-output directories.
- Do not begin later batches unless explicitly requested.
- Add or update tests when practical.
- Preserve backward compatibility unless the task explicitly changes the contract.
- Investigate failures rather than rerunning commands and hoping.
- Do not hide failures, and do not claim success validation does not support.
- Follow the environment, test, and git rules in `AGENT_CONFIG.md`.

## Process

1. Restate the goal and scope internally.
2. Inspect existing behavior and relevant tests.
3. Identify the minimal coherent change.
4. Implement it.
5. Add or update focused tests.
6. Run targeted validation, then broader validation if appropriate and not expensive.
7. Inspect `git diff` and `git status`.
8. Return the report below.

## Implementation principles

- Prefer small, reviewable changes over broad rewrites.
- Fail closed on malformed or inconsistent input when correctness matters.
- Do not silently ignore contradictions in structured data.
- Keep CLI behavior and output contracts stable unless intentionally changed.
- Update docs when user-facing behavior changes.
- Avoid hidden dependencies on local-only artifacts.

## Output format

```md
# Implementation report

## Goal

What was implemented.

## Root cause or design rationale

Why this change was needed.

## Files changed

Each changed file and its purpose.

## Behavior changes

What now behaves differently, including compatibility impact.

## Tests and validation

Commands run and their results.

## Risks and caveats

Known limitations and remaining uncertainty.

## Final git status

Staged, modified, and untracked files.

## Reviewer prompt

A ready-to-paste prompt for independent review, naming: files in scope; files out of scope; behavior claims to verify; tests to run; outputs to inspect; and an instruction not to edit, stage, commit, or push.

## Next step

Ready for review, or blocked. Do not recommend commit — that is `review`'s decision.
```
