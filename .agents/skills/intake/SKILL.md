---
name: intake
description: Start work from an existing tracker issue (GitHub or Linear) instead of from a conversation. Reads the issue, judges how well specified it is, and routes it into the normal grill/plan/execute workflow.
allowed-tools: Read, Grep, Glob, Bash
---

# Skill: intake

Use this skill when the starting point is an issue that already exists in a tracker, rather than an idea in the conversation.

`intake` is an **entry point, not a stage**. It does not replace `grill`, `plan`, or `execute` — it decides which of them the issue is ready for, and hands off.

It serves two workflows:

- **Idea to code** — run standalone, it reads an issue and routes it into `grill` or `plan`, with the user approving each gate.
- **Issue resolution** — as the first step of the autonomous workflow, where it also decides whether the issue is specific enough to implement without a human. If it is not, escalate instead of routing onward.

If `AGENT_CONFIG.md` records no tracker, this skill does not apply. Say so and ask the user to paste the work item directly.

## When to use

- "Start on issue #42."
- "Pick up the next open issue in the backlog."
- "What's ready to work on?"

## Inputs

Read the context files listed in `AGENTS.md`, the tracker configuration in `AGENT_CONFIG.md`, the issue itself with its full comment thread, any linked issues or PRs, and any existing `SPECS/` or `TASKS/` file that already covers it.

Use whatever tracker access `AGENT_CONFIG.md` specifies — the `gh` CLI for GitHub, MCP tools for Linear.

## Rules

- Read-only against the tracker by default. Do not create, close, assign, or comment on issues unless the user explicitly asks.
- Never invent scope the issue does not state. An issue is a request, not a spec.
- Check `SPECS/` and `TASKS/` before starting — the issue may already have a PRD or plan, in which case resume rather than restart.
- Never skip straight to `execute` on a vague issue. Route it to `grill`.
- If several issues qualify, list them and let the user choose. Do not pick silently.

## Process

1. Fetch the issue: title, body, labels, comments, linked items, current state.
2. Check whether `SPECS/` or `TASKS/` already covers it.
3. Assess readiness against the routing table below.
4. Record the issue reference so later artifacts stay linked to it.
5. Hand off to the chosen stage with a ready-to-paste prompt.

## Routing

| Issue state | Route to |
|---|---|
| Vague, or missing success criteria, scope, or constraints | `grill` |
| Well specified, no PRD in `SPECS/` yet | `grill`, to formalize it quickly |
| A PRD exists and is approved, but no task file | `plan` |
| A task file exists with `pending` batches | `execute` on the next batch |
| Reports a defect with clear reproduction steps | `execute`, scoped as a fix |
| Asks a question or requests investigation, not a change | `inspect` |

When routing to `grill`, carry the issue's own wording into the PRD's Motivation section rather than paraphrasing it away.

Inside the issue-resolution workflow the first two rows do not apply — that workflow has no `grill` stage. An issue that would route to `grill` is instead an escalation: report that the issue is underspecified, say exactly what is missing, and stop.

## Traceability

Any `SPECS/` or `TASKS/` file created downstream should name the issue in its header, so the trail from tracker to spec to task to commit stays intact. Use the same slug for the spec and the task file.

## Output format

```md
# Intake

## Issue

Identifier, title, state, and link.

## What it asks for

The request in plain language, distinguishing what the issue states from what it implies.

## Existing artifacts

Any SPECS/ or TASKS/ file that already covers this, or "none".

## Readiness assessment

What is specified, and what is missing.

## Open questions

What must be answered before implementation, if anything.

## Recommended route

The next stage, and why.

## Handoff prompt

A ready-to-paste prompt for that stage.
```
