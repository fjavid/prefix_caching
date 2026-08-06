# Workflow: idea to code

**Default workflow.** Use it unless the user selects another.

Turns an idea into reviewed, working code through small vertical batches, with the user at every stage gate.

## When to use

The starting point is an idea, a feature request, a research direction, or a bug described in conversation — anything where the requirements are not already written down and approved.

## Preconditions

`AGENT_CONFIG.md` exists and has no placeholders. If not, run `bootstrap` first.

## Sequence

```text
idea ──▶ grill ──▶ plan ──▶ execute ──▶ review ──┬──▶ done
                              ▲                  │
                              └──── blockers ────┘
```

1. **`grill`** — clarify the idea, challenge assumptions, resolve unknowns. Produces a PRD in `SPECS/<slug>.md`.
2. **`plan`** — convert the approved PRD into vertical batches. Produces `TASKS/<slug>.md`.
3. **`execute`** — implement one batch, with validation. One batch at a time.
4. **`review`** — gate that batch. If blockers, return to `execute` with a narrow fix prompt, then review again.
5. Repeat 3–4 for the next batch until every batch is `done`.

## Human in the loop

The user approves at each gate. Do not cross a gate on your own judgment:

| Gate | Requires |
|---|---|
| PRD written | user approves it before `plan` |
| Plan written | user approves it before `execute` |
| Batch implemented | `review` approves before the next batch |
| All batches done | user decides on commit |

Do not start a later batch unless asked. Do not skip `grill` because the idea seems clear — that judgment is the user's.

## Entering mid-workflow

Skip stages whose output already exists. Enter at `plan` when an approved PRD exists; at `execute` when an approved batch exists; at `review` when implementation is complete. State which stage you are entering and why.

## Stopping

Stop and ask the user when: the PRD's open questions block planning; a batch turns out to need a design decision the PRD does not cover; `review` finds a blocker that is a specification problem rather than a code problem; or the work would require an operation listed as expensive in `AGENT_CONFIG.md`.

## Produces

A PRD in `SPECS/`, a task file in `TASKS/` with per-batch status, implemented code with validation, and a review verdict per batch.
