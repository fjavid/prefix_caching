#!/usr/bin/env bash
# Install the multi-agent system into this repo. Idempotent — safe to re-run
# after adding a skill or an agent.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
AGENTS=(claude codex grok)

echo "Installing agent system v$(cat .agents/VERSION) into ${ROOT}"

# 1. Per-skill symlinks for each agent. Per-skill (not per-directory) so an
#    agent can keep private skills alongside the shared ones.
for agent in "${AGENTS[@]}"; do
  mkdir -p ".${agent}/skills"
  for skill_dir in .agents/skills/*/; do
    skill="$(basename "${skill_dir}")"
    link=".${agent}/skills/${skill}"
    [ -L "${link}" ] && rm -f "${link}"
    if [ -e "${link}" ]; then
      echo "  skip  ${link} (real directory, not replacing)"
      continue
    fi
    ln -s "../../.agents/skills/${skill}" "${link}"
  done
  # Drop symlinks whose target skill no longer exists.
  for link in ".${agent}"/skills/*; do
    [ -L "${link}" ] || continue
    [ -e "${link}" ] || { rm -f "${link}"; echo "  prune ${link}"; }
  done
  echo "  ok    .${agent}/skills/"
done

# 2. Working directories.
mkdir -p SPECS TASKS TASKS/handoffs tmp
touch tmp/.gitkeep

# 3. Keep temp files out of git.
if [ -f .gitignore ]; then
  grep -qxF 'tmp/' .gitignore || printf '\n# agent scratch space\ntmp/\n' >>.gitignore
else
  printf 'tmp/\n' >.gitignore
fi
echo "  ok    SPECS/ TASKS/ tmp/ (tmp gitignored)"

# 4. Placeholder config so the bootstrap gate in AGENTS.md fires.
if [ ! -f AGENT_CONFIG.md ]; then
  cp .agents/templates/AGENT_CONFIG.template.md AGENT_CONFIG.md
  echo "  ok    AGENT_CONFIG.md created from template"
else
  echo "  skip  AGENT_CONFIG.md (already exists)"
fi

echo
echo "Done. Next: start an agent and run the 'bootstrap' skill to fill AGENT_CONFIG.md."
