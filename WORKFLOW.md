# Workflows

A **skill** is one unit of work. A **workflow** is an ordered composition of skills with its own entry conditions, human-interaction rules, and stopping rules.

This repo supports more than one workflow. This file is the registry and the shared rules; each workflow's detail lives in its own file.

## Registry

Available workflows are the files in `.agents/workflows/`. List that directory to discover them — do not rely on a hard-coded list.

| Workflow | File | Default? | Human in the loop |
|---|---|---|---|
| Idea to code | `.agents/workflows/idea-to-code.md` | **yes** | at every stage gate |
| Issue resolution | `.agents/workflows/issue-resolution.md` | no — opt-in | only on escalation |

**Idea to code is the default.** Use it unless the user selects another one.

Any non-default workflow must be **explicitly requested**. If the user's request could plausibly mean a non-default workflow — for example "work through the open issues" — ask which workflow they want before starting. Never enter an autonomous workflow by inference.

Individual skills can also be run standalone, outside any workflow. `inspect`, `present`, `summarize`, and `bootstrap` are usually run that way.

## Stages and skills

| Skill | Purpose |
|---|---|
| `bootstrap` | create or refresh `AGENT_CONFIG.md` |
| `summarize` | generate the repo overview document |
| `intake` | read a tracker issue and assess readiness |
| `grill` | clarify an idea into a PRD in `SPECS/` |
| `plan` | split an approved PRD into vertical batches in `TASKS/` |
| `execute` | implement one batch with validation |
| `review` | gate a scoped change; find blockers; approve or reject |
| `inspect` | broad read-only assessment; never gates a commit |
| `present` | communicate results to humans |

`review` vs `inspect`: `review` is scoped — one diff, one batch — and gates a commit. `inspect` is broad and gates nothing.

## Agents

`AGENT_CONFIG.md` declares which agents are enabled here and which stages each may run. That table wins over the defaults below.

| Agent | Default role | Default stages |
|---|---|---|
| Claude | orchestrator; owns continuity and final synthesis | all |
| Codex | implementation, tests, debugging, scoped review | `execute`, `review`, `plan` |
| Grok (optional) | adversarial critique and alternative reasoning | `grill`, `review`, `inspect` |
| Antigravity (optional) | as configured | as configured |

Full role definitions: `.agents/roles/<agent>.md`.

Only Claude may delegate. Other agents run standalone, must not assume another agent is present, and must not attempt to invoke one.

## User control

Explicit user instructions override every default here. The user may pick the workflow, the stage, or the agent; skip a stage; or repeat one.

When the user names an agent, use that agent. Do not silently substitute another or add redundant parallel agents. If the named agent cannot do the task, explain why before changing the assignment.

## Delegation

Claude delegates only when the sub-agent adds real value, and stays responsible for verifying the result. Do not trust a sub-agent report you have not checked against the diff, the tests, or `git status`.

Invocation commands for each enabled agent are recorded in `AGENT_CONFIG.md`.

Sub-agents write their report to `TASKS/handoffs/<date>-<agent>-<slug>.md` and return that path, so the work is inspectable rather than trapped in a transcript.

Handoff template:

```text
You are acting as a sub-agent. Do not delegate further.

Read:
- AGENTS.md
- AGENT_CONFIG.md
- .agents/roles/<agent>.md
- .agents/skills/<skill>/SKILL.md
- <relevant SPECS/, TASKS/, source, and output files>

Context:
- <the feature or problem, and what is already done>

Scope:
- in scope: <files/directories>
- out of scope: <files/directories>

Task:
- goal: <exact goal>
- editing allowed: <yes/no>
- forbidden: staging, committing, pushing, and any scope expansion

Validation:
- run: <commands>
- acceptance criteria: <criteria>

Return:
- follow the Output format in the skill file
- write it to TASKS/handoffs/<date>-<agent>-<slug>.md and return the path
```

Sub-agents stay inside the delegated scope and never expand it silently.

## Handoff checklist

A handoff is ready when it answers: which workflow; which skill; which agent; the exact goal; established context; in-scope and out-of-scope files; whether editing is allowed; what must stay backward compatible; what validation to run; what risks need attention.

## Completion checklist

Before calling a stage complete: the requested scope is done; the result matches the approved spec or task; validation passed and failures were investigated, not just rerun; unrelated behavior is unchanged; evidence has clear provenance; remaining risks are stated; the next step is explicit.

## Adding a workflow

Create `.agents/workflows/<name>.md` documenting: when to use it; whether it is opt-in; preconditions; the skill sequence; human-interaction rules; stopping and escalation rules; and what it produces. Then add a row to the registry table above.
