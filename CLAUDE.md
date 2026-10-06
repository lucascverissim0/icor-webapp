# ICOR Web App — Claude Code

@AGENTS.md

The directives above were written for Codex and apply unchanged to Claude Code:
"Codex" means whichever agent is working, and `docs/CODEX_HANDOFF.md` remains the
single durable handoff. Read it before working, as `AGENTS.md` requires.

## This machine

- Since 2026-09-23 the only active machine is Lucas's Mac; the clone is `~/ICOR`.
  Windows paths (`C:\Users\LucasCravoVERISSIMO\...`) and PowerShell instructions in
  older handoff entries describe the retired Windows machine.
- GitHub account for this repository: `lucascverissim0`. `gh` holds it alongside the
  unrelated work account `lverissimo-01`, which must stay the active `gh` account (another
  session uses it). A repo-local credential helper in `.git/config` makes git here use
  `lucascverissim0` regardless; for `gh` commands prefix
  `GH_TOKEN=$(gh auth token --user lucascverissim0)`. Never switch the active account.
- Commits use the repository-local identity `Lucas Cravo Verissimo
  <lucas.c.verissimo@gmail.com>`.
