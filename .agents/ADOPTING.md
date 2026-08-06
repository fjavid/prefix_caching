# Adopting this system in a new repo

## Install

```bash
bash .agents/install.sh
```

This creates the per-skill symlinks under `.claude/skills/`, `.codex/skills/`, and `.grok/skills/`, scaffolds `SPECS/`, `TASKS/`, and `tmp/`, gitignores `tmp/`, and drops a placeholder `AGENT_CONFIG.md`.

Run it again after adding a skill — it is idempotent.

## Configure

Start any agent and run the `bootstrap` skill. It reads the repo, asks up to 10 short questions, and writes `AGENT_CONFIG.md`.

Nothing else should be edited during adoption. If you find yourself editing a portable file to describe your repo, that detail belongs in `AGENT_CONFIG.md` instead.

## What is portable vs repo-specific

| Portable — copy as-is | Repo-specific — generated per repo |
|---|---|
| `AGENTS.md` | `AGENT_CONFIG.md` |
| `WORKFLOW.md` | `README.md` |
| `CLAUDE.md`, `CODEX.md`, `GROK.md` | `SPECS/`, `TASKS/` |
| `.agents/` | |

## Adding a skill

1. Create `.agents/skills/<name>/SKILL.md` with `name` and `description` frontmatter. Add `allowed-tools: Read, Grep, Glob, Bash` if the skill must be read-only.
2. Re-run `.agents/install.sh` to symlink it for every agent.
3. Add a row to the skill table in `WORKFLOW.md`.

## Adding a workflow

1. Create `.agents/workflows/<name>.md` documenting: when to use it; whether it is opt-in; preconditions; the skill sequence; human-interaction rules; stopping and escalation rules; and what it produces.
2. Add a row to the registry table in `WORKFLOW.md`.

No installer step — workflows are read by path, not symlinked.

Skills and workflows are both discovered by listing their directory. No file hard-codes either list.

## Adding an agent

1. Create `.agents/roles/<agent>.md`.
2. Add a thin loader at the repo root if the agent's harness auto-loads a specific filename.
3. Add `.<agent>/skills/` to the loop in `install.sh`.
4. Re-run `bootstrap` to enable it in `AGENT_CONFIG.md`.

## Caveat

Symlinks do not survive zip archives or copies onto Windows filesystems. Copy `.agents/` and run the installer rather than copying the agent directories.
