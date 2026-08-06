---
name: summarize
description: Generate or refresh the repository overview document — purpose, goals, architecture, data flow, key modules, and domain background. Use when a repo has no overview doc, or the existing one has drifted from the code.
---

# Skill: summarize

Produce the document that explains what this repository *is* — the one an agent or a new contributor reads to orient themselves.

The output is usually `ARCHITECTURE.md`, but the name is whatever `AGENT_CONFIG.md` records under "Repo overview". Never assume `ARCHITECTURE.md` exists or is the right name; read the config.

## When to use

- `AGENT_CONFIG.md` records no overview document and the user agrees to create one;
- the existing overview has drifted from the code;
- a major restructuring changed the architecture;
- the user asks for a repo overview.

Do not create this document unilaterally. Confirm with the user first — it is a durable artifact they will have to maintain.

## Rules

- **Describe what exists, not what should exist.** This is documentation, not a proposal. Improvement ideas belong in `inspect` output.
- Ground every claim in a file you actually read. No inference presented as fact.
- Mark anything uncertain as uncertain, or leave it out.
- Do not duplicate `README.md`. README is *how to run it*; this is *what it is and how it is built*. Cross-reference instead of restating.
- Do not duplicate `AGENT_CONFIG.md` rules or `.agents/` content.
- Do not edit code.
- Prefer a shorter accurate document over a longer speculative one.
- On a refresh, preserve sections that are still correct; rewrite only what drifted, and say what changed.

## Process

1. Read `AGENT_CONFIG.md` for the target filename and project identity.
2. Read `README.md` and any per-module READMEs.
3. Map the directory structure and identify the real entry points.
4. Trace the primary data or control flow end to end.
5. Identify the core abstractions and where they are defined.
6. Identify configuration surfaces, external dependencies, and generated artifacts.
7. Note the domain or scientific background needed to read the code.
8. Draft the document, marking gaps explicitly.
9. Write it, and record its path in `AGENT_CONFIG.md` under "Repo overview".

## Document template

```md
# Architecture

## Purpose

What this repository does and why it exists.

## Goals and non-goals

What it is trying to achieve, and what it deliberately does not.

## Background

Domain, scientific, or technical context needed to read the code. Define the
terms the code uses without explaining.

## System overview

The main components and how they relate. A diagram if it helps.

## Directory map

Each significant directory and its responsibility.

## Data and control flow

The primary path end to end: what enters, what transforms it, what comes out.

## Key abstractions

The central types, interfaces, or concepts, and where they are defined.

## Configuration

Configuration surfaces, environment variables, and their effects.

## External dependencies

Services, models, datasets, and infrastructure this repo relies on.

## Generated artifacts

What the repo produces, where it lands, and how to judge whether it is current.

## Invariants and assumptions

What must stay true for the system to be correct.

## Known gaps

What is undocumented, unverified, or inconsistent — stated plainly.
```

Drop sections that do not apply. An empty section is worse than an absent one.

## Output format

```md
# Summary report

## Document written

Path, and whether created or refreshed.

## Sources read

The files this was grounded in.

## What changed

For a refresh: which sections were rewritten and why.

## Gaps

What could not be determined from the code, and what would resolve it.

## Config update

Confirmation that AGENT_CONFIG.md records the overview document path.
```
