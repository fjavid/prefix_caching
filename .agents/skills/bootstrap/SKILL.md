---
name: bootstrap
description: Set up or refresh AGENT_CONFIG.md, the single repo-specific configuration file for agents. Infers what it can from the repo, asks the user a short set of clear questions, and writes the result. Re-runnable at any time.
---

# Skill: bootstrap

Use this skill to create or refresh `AGENT_CONFIG.md` — the one file holding every repo-specific agent rule.

Run it when:

- `AGENT_CONFIG.md` is missing or still contains `<PLACEHOLDER>` markers;
- the system was just copied into a new repo;
- the environment, test command, tracker, or enabled agents have changed;
- the user asks to reconfigure.

## Rules

- **Infer first, ask second.** Read the repo before asking anything. Never ask what the repo already answers.
- **Ask at most 10 questions**, and fewer whenever inference covers a section.
- **Keep questions short and closed.** Prefer yes/no or a pick-from-list. Never open-ended.
- **Show your inferred answer inside the question** so the user can confirm with one word.
- **Batch the questions**, do not interrogate one at a time.
- Do not edit code. Do not stage, commit, or push.
- On a re-run, read the existing `AGENT_CONFIG.md` and ask only about placeholder, empty, or contradicted sections. Preserve everything the user already confirmed.
- Never write repo-specific detail into portable files. It all goes in `AGENT_CONFIG.md`.

## Inference pass

Before asking anything, determine from the repo:

- language, package manager, and dependency files;
- virtualenv or interpreter path;
- test framework and how tests are laid out;
- entry points, pipelines, and scripts;
- `.gitignore` and which directories hold generated output;
- whether a README already documents how to run the repo;
- whether any document describes the architecture or design — check `ARCHITECTURE.md`, `DESIGN.md`, `docs/`, and the README's own sections, since the name varies;
- default branch and whether a remote exists;
- any expensive operations (training, cluster submission, paid APIs).

Write down what you inferred. That becomes the draft.

## The question set

Ask only what inference could not settle, capped at 10. Draw from these, in priority order:

1. One-line description of what this repo is, to confirm or correct.
2. Confirm the interpreter/environment command.
3. Confirm the default test command.
4. Confirm the do-not-modify directories.
5. May agents stage changes? Commit? Push? (one question, three parts)
6. Which optional agents are enabled besides Claude and Codex — Grok, Antigravity, neither?
7. For each enabled optional agent: which stages may it run?
8. Is there an issue tracker to link (GitHub/Linear/Jira) — and if so, project name and ID? This is what enables `intake` and the issue-resolution workflow.
9. Which operations must never run unattended (training, cluster jobs, paid API calls)?
10. Confirm the repo overview document — the file describing architecture and domain background. Offer the candidate you found, or report that none exists.

If none exists, ask whether the user wants one generated. If yes, run `summarize` after `bootstrap` completes and record the resulting path. Do not generate it unasked, and do not assume the name is `ARCHITECTURE.md`.

If a slot remains under the cap, ask about domain-specific correctness risks reviewers should always flag. Otherwise propose them from the code and let the user correct.

If the user declines a question, record the safe default and mark it as an assumption.

## Safe defaults

Use these when the user does not answer:

- staging, committing, and pushing: **not allowed**;
- optional agents: **disabled**;
- non-default workflows: **not permitted** without an explicit request;
- repo overview document: **none**, and not generated;
- expensive operations: **never run unattended**;
- temp files: `./tmp/`;
- generated directories: whatever `.gitignore` lists.

## Process

1. Run the inference pass.
2. Read `AGENT_CONFIG.md` if it exists; note which sections need input.
3. Ask the batched questions (≤10).
4. Fill `.agents/templates/AGENT_CONFIG.template.md` with the answers.
5. Write `AGENT_CONFIG.md`. Ensure no `<PLACEHOLDER>` markers remain.
6. Ensure the temp directory is gitignored.
7. Return the summary below.

## Output format

```md
# Bootstrap

## Inferred

What was determined from the repo without asking.

## Confirmed with the user

Answers received.

## Assumptions

Defaults applied where the user did not answer.

## Written

Path to AGENT_CONFIG.md, plus any other file touched.

## Remaining gaps

Anything still unknown, and what would resolve it.

## Next step

The recommended next stage.
```
