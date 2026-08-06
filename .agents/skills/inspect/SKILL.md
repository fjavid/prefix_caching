---
name: inspect
description: Broad read-only assessment of a repository, subsystem, workflow, or result set, producing a risk roadmap. Use for wide evaluation, not for reviewing a specific diff or gating a commit.
allowed-tools: Read, Grep, Glob, Bash
---

# Skill: inspect

Use this skill for comprehensive evaluation. The goal is to understand the larger system and produce a report identifying important risks, gaps, and inconsistencies.

Use `review` instead for: the current git diff, a narrow implementation batch, or any decision about whether a change is ready to commit. `inspect` never gates a commit.

## When to use

- repository-wide or subsystem evaluation;
- architecture evaluation;
- experiment or result-set evaluation;
- workflow, test-strategy, or documentation-consistency evaluation;
- building a roadmap of issues before planning repairs.

## Inputs

Read the context files listed in `AGENTS.md`, plus the source layout, tests, scripts, and any outputs or logs in scope. Do not assume any file exists; continue with what is available.

## Rules

- Read-only. Do not edit, and do not repair findings unless the user explicitly switches the task to execution.
- Do not turn this into a line-by-line code review.
- Prefer targeted inspection and cheap commands over expensive runs.
- Clearly separate confirmed findings from hypotheses.
- Treat generated artifacts as possibly stale until provenance is checked.
- Be explicit about what was not inspected.

## Process

1. Clarify the scope.
2. Read repo-level instructions and context.
3. Map the relevant architecture, workflows, and artifacts.
4. Identify correctness, design, validation, and process risks.
5. Check whether tests and outputs support the claims being made.
6. Look for stale, contradictory, or misleading docs and artifacts.
7. Group findings by severity and theme.
8. Recommend next steps without implementing them.

## What to look for

Architecture and boundaries; data and artifact contracts; experiment provenance and reproducibility; validation strategy; test coverage quality; documentation accuracy; command reliability; generated-output staleness; dependency and environment assumptions; hidden coupling; domain assumptions (see `AGENT_CONFIG.md`); workflow fragility; unclear ownership or next steps.

## Severity

**Blocking** findings are broad risks that may invalidate conclusions, make outputs misleading, hide correctness bugs across workflows, make future implementation unsafe, or show that architecture and documentation are substantially inconsistent.

**Non-blocking** findings are cleanup opportunities, documentation improvements, optional hardening, and useful-but-not-urgent workflow improvements.

## Output format

```md
# Inspection

## Scope

What was inspected and what was explicitly out of scope.

## Executive summary

Brief overall assessment.

## System map

Important components, workflows, artifacts, or data paths.

## Blocking risks

Broad issues requiring attention before further work.

## Non-blocking risks

Important but less urgent concerns.

## Validation and test gaps

Missing or weak evidence.

## Documentation and artifact gaps

Stale, contradictory, unclear, or missing docs and artifacts.

## Suggested next steps

Recommended repair, planning, review, or execution tasks.

## Commands run

Commands and results, if any.

## Files inspected

Important files and directories reviewed.

## Confidence and caveats

What is confirmed, what is inferred, what remains unknown.
```
