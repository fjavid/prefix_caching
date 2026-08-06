---
name: plan
description: Turn an approved idea or PRD into an implementation task plan divided into small vertical executable batches.
---

# Skill: plan

Use this skill once the idea is clear enough to implement. The goal is to convert a PRD into a task file with vertically executable batches.

## When to use

Turning a PRD into an implementation plan; splitting work into vertical batches; defining validation; identifying dependencies and ordering; preparing handoffs for execution.

Do not implement code during `plan`.

## Inputs

Read the context files listed in `AGENTS.md`, the target `SPECS/*.md`, existing `TASKS/*.md`, and the relevant source, tests, and prior reports.

## Rules

- Do not edit code unless explicitly asked.
- Keep the plan grounded in the actual repo, not in what the PRD assumes exists.
- Prefer vertical slices over horizontal refactors.
- Each batch must be independently reviewable and include validation.
- Call out risky dependencies and ordering constraints.
- List open questions explicitly; do not bury them.

## Process

1. Read the PRD and repo context.
2. Confirm the intended outcome and non-goals.
3. Identify affected components and their data, API, CLI, doc, and test contracts.
4. Split the work into vertical batches.
5. Define acceptance criteria and validation per batch.
6. Identify rollback and compatibility concerns.
7. Write the task file in `TASKS/<slug>.md`, matching the PRD's slug.

## Vertical batch definition

A good batch implements meaningful end-to-end behavior, touches only necessary layers, is testable independently, is small enough to review, avoids speculative refactors, and leaves the repo working.

Avoid batches like "refactor all utilities", "update all tests", "implement the whole feature", "cleanup later".

The plan should address, where applicable: existing data and file formats; configuration schemas; serialized artifacts; public APIs and CLIs; migration or explicit incompatibility handling; targeted and regression validation.

## Batch status vocabulary

Use exactly one of: `pending`, `in_progress`, `blocked`, `done`, `abandoned`. This is what lets a later session resume the task file without the original conversation.

## Output format

```md
# Implementation plan

## Source spec

Path to the PRD.

## Goal

Brief implementation goal.

## Affected areas

Files, modules, tests, docs, commands likely involved.

## Assumptions

Assumptions taken from the spec and repo.

## Open questions

Questions that still affect implementation.

## Batch plan

### Batch 1: <name>

Goal:
Scope:
Out of scope:
Implementation notes:
Tests/validation:
Acceptance criteria:
Review focus:

### Batch 2: <name>

...

## Task file path

`TASKS/<slug>.md`

## Risks

Key correctness, compatibility, testing, or rollout risks.

## Recommended next prompt

A ready-to-paste `execute` prompt for Batch 1.
```

## Task file format

```md
# Task: <title>

## Source spec

## Overall goal

## Global constraints

## Batches

### Batch 1: <name>

Status: pending | in_progress | blocked | done | abandoned
Scope:
Implementation steps:
Validation:
Review notes:

### Batch 2: <name>

...

## Review history

## Final acceptance criteria
```
