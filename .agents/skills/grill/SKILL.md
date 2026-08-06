---
name: grill
description: Read an initial idea or plan, challenge assumptions, ask clarifying questions, and gather enough detail to write a strong PRD/spec.
---

# Skill: grill

Use this skill when the user has an idea but the requirements, risks, or success criteria are not yet clear. The goal is to understand the idea deeply before planning or implementation.

## When to use

Early feature ideas; vague implementation requests; research directions; new workflows; unclear bug reports; architectural changes; experiment plans; PRD preparation.

Do not implement code during `grill`.

## Inputs

The idea may come from the user's message, a note in `SPECS/`, a previous task file, or existing code and outputs. Read the context files listed in `AGENTS.md` as needed.

## Rules

- Do not edit code.
- Do not write the final PRD until the important unknowns are resolved or explicitly accepted as assumptions.
- Never ask what inspecting the repository would answer.
- Prefer a few high-value questions over many vague ones.
- Challenge the idea respectfully; surface hidden risks, edge cases, and success criteria.
- Separate must-have requirements from nice-to-have ideas.
- If the user wants one-question-at-a-time grilling, do that.
- Do not produce a separate archaeology or investigation report unless asked.

## Process

1. Read the idea and relevant context.
2. Summarize the apparent goal in plain language.
3. Identify major unknowns, risks, and failure modes.
4. Ask clarifying questions.
5. Record assumptions where the user declines to answer.
6. Continue until there is enough detail to write a PRD.
7. Produce the PRD when asked.

## Question style

Ask about: intended consumer; exact success criteria; input/output contracts; expected workflow; data assumptions; constraints and non-goals; compatibility; performance; testing expectations; failure handling; rollout; what must not change.

Avoid: "What should the feature do?", "Any other details?", "What is the goal?"

Prefer: "Should this preserve the old CLI output format, or can it break compatibility?", "Should malformed inputs fail closed, warn and skip, or be repaired?", "What is the smallest slice that would prove this works?"

## Output format during grilling

```md
# Grill

## My understanding

Brief summary of the idea.

## Key risks or ambiguities

Important issues to resolve.

## Questions

Numbered questions, grouped if helpful.

## Suggested default assumptions

Reasonable defaults if the user wants to move quickly.

## PRD readiness

Whether enough detail exists to write the PRD.
```

## Output format when writing the PRD

Write or update `SPECS/<slug>.md` with:

```md
# PRD: <title>

## Goal

## Motivation

## User-facing behavior

## Non-goals

## Requirements

## Inputs and outputs

## Edge cases

## Compatibility constraints

## Testing and validation

## Rollout plan

## Open questions
```
