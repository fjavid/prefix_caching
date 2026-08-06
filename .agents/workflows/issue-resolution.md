# Workflow: issue resolution

**Opt-in. Not the default.** Autonomous: it runs Claude and Codex against each other without the user, and only reaches out when it gets stuck.

Takes a tracker issue (GitHub or Linear) and drives it to a commit: plan, implement, review, fix, commit. There is no `grill` stage — the issue *is* the specification. If the issue is not specific enough to be one, this workflow escalates rather than guessing.

## When to use

Only when the user explicitly asks for it — by name, or unmistakably ("resolve issue #42 autonomously", "run the issue workflow on the backlog").

If a request could mean this workflow or `idea-to-code`, **ask which one**. "Look at issue #42" is not explicit; it might mean "read it and tell me what you think." Never enter this workflow by inference.

## Preconditions

All of these must hold. If any fails, stop and say which:

- a tracker is configured in `AGENT_CONFIG.md`;
- committing is allowed in `AGENT_CONFIG.md`;
- Codex is enabled;
- the issue is assigned or the user has named it;
- the working tree is clean, so the resulting diff is attributable to this workflow.

## Sequence

```text
intake ──▶ plan ──▶ execute ──▶ review ──┬── approved ──▶ commit
                       ▲                 │
                       └─── blockers ────┘
                        (max 3 rounds)
```

1. **`intake`** (Claude) — read the issue and its thread. Assess whether it is specific enough to implement. If not, escalate.
2. **`plan`** (Claude) — write `TASKS/<issue-ref>-<slug>.md` with vertical batches. No PRD; the issue is the spec, quoted in the task file's header.
3. **`execute`** (Codex) — implement one batch. Codex does not stage, commit, or push.
4. **`review`** (Claude) — gate the batch against the issue and the task file. Claude reviews Codex's work; it does not review its own.
5. On blockers, hand a narrow fix prompt back to Codex and return to step 3. **Maximum 3 execute/review rounds per batch.**
6. On approval, move to the next batch. When all batches are `done`, commit.

## Round limit

Three execute/review rounds per batch. On the fourth, stop and escalate with the full history — what was tried, what each review found, and why it did not converge.

The counter is per batch and resets when a batch is approved. It does not reset when the same blocker reappears in a new form; a blocker that keeps returning is a signal to escalate, not to retry.

## Escalate immediately when

Do not burn rounds on these — stop at once:

- the issue is ambiguous in a way that needs a product or research decision;
- the fix requires scope beyond what the issue describes;
- the fix requires an operation listed as expensive or dangerous in `AGENT_CONFIG.md`;
- a review blocker is a specification problem, not a code problem;
- the change would break a documented contract;
- two consecutive reviews disagree about whether something is a blocker;
- anything requires deleting or overwriting data, or touching a do-not-modify directory.

Escalation means: stop, do not commit, and report the state clearly. It is a normal outcome, not a failure.

## Committing

Commit only after `review` approves every batch, and only within what `AGENT_CONFIG.md` permits.

- Use exact scoped `git add -- <files>`; never `git add .`.
- Reference the issue in the commit message.
- Push only if `AGENT_CONFIG.md` allows pushing.
- Do not close, comment on, or reassign the issue unless the user explicitly asked. Tracker writes are not part of this workflow.

## Autonomy boundaries

Running without a human does not widen what agents may do. Every universal safety rule in `AGENTS.md` and every restriction in `AGENT_CONFIG.md` still applies — unchanged. Specifically: no scope expansion, no expensive operations, no touching do-not-modify directories, and no claiming validation that did not run.

Codex never commits, even here. Only Claude commits, and only at the end.

## Produces

A task file in `TASKS/`, implemented code, a review verdict per batch, and either a commit or an escalation report naming the exact blocker.

## Final report

Whether it committed or escalated, report: the issue, batches attempted, rounds used per batch, what each review found, files changed, validation run and its results, and the commit SHA or the reason for stopping.
