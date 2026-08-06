---
name: present
description: Explain, summarize, and present a repository, feature, experiment, or result set to humans. Suggests storylines, slides, figures, and audience-appropriate framing.
---

# Skill: present

Use this skill when the goal is communication rather than implementation. The goal is to translate repo details, changes, experiments, or results into a presentation, update, report outline, or figure plan.

## When to use

Weekly updates; slide outlines; presentation storylines; figure recommendations; explaining a subsystem; summarizing experiment results; preparing manager or team communication.

## Inputs

Read the context files listed in `AGENTS.md`, plus the relevant logs, outputs, figures, reports, notebooks, and — for progress updates — recent commits and review notes.

## Rules

- Do not edit files unless explicitly asked.
- Match the audience; prefer big-picture framing unless the audience needs code detail.
- Distinguish observed results from interpretation, and prototype evidence from production readiness.
- Suggest only figures that existing outputs can actually support.
- Do not invent results or metrics, and do not overclaim.

## Process

1. Identify the audience and the main message.
2. Inspect the evidence and artifacts.
3. Decide what to omit.
4. Build a storyline and suggest figures.
5. Draft the requested output, with caveats and next steps.

## Audience modes

- **Executive/manager** — outcome, impact, risk, next decision. No file names or internal identifiers.
- **Engineering team** — workflow, design tradeoffs, validation, limitations, next step.
- **Research/technical** — assumptions, metrics, ablations, failure modes, evidence strength.
- **Domain specialist** — their domain language first; technical language only when it clarifies.

## Figure suggestions

For each figure give: title; what it shows; source file or data needed; intended takeaway; caveat.

Good figures often show before/after behavior, cumulative progress, tradeoff curves, ranked decisions, failure-mode examples, or workflow diagrams.

## Output formats

Weekly update:

```md
# Weekly update

- <bullet>

# Next

- <next step>
```

Slide plan:

```md
# Presentation plan

## Audience

## Core message

## Storyline

## Slide outline

### Slide 1: <title>

Purpose:
Key points:
Suggested figure:
Speaker note:

## Figures to create or reuse

## Caveats

## Backup slides
```

Result explanation:

```md
# Result summary

## Main takeaway

## Evidence

## What the figures mean

## What not to overclaim

## Next checks
```
