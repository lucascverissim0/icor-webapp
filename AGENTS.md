# ICOR Web App — Codex Directives

These instructions apply to the entire repository.

## Mission and scope

- The long-term goal is to repair and improve the ICOR web app until it is reliable, secure, maintainable, and exceptionally user friendly.
- Work local-first. Run and demonstrate a local version as changes are developed.
- Do not deploy, publish, or intentionally modify the behavior or data of the existing multi-user web deployment unless Lucas explicitly authorizes that production action.
- Preserve unrelated user changes. Never discard work or rewrite Git history without explicit authorization.

## Mandatory session startup

Before investigating, planning, editing, or running the application:

1. Read this file completely.
2. Read `docs/CODEX_HANDOFF.md` completely.
3. Inspect `git status` and preserve all existing changes.
4. Continue from the recorded checkpoint instead of repeating completed investigation.

## Durable continuity

`docs/CODEX_HANDOFF.md` is the durable project memory. Keep it accurate and useful across cleared conversations.

Before every final response that contains material progress, update the handoff with all relevant new information, including:

- confirmed findings and their evidence;
- decisions and user preferences;
- files changed and why;
- verification commands and actual results;
- unresolved risks or uncertainties;
- the exact next recommended actions;
- whether a local server or other long-running process is active.

Do not claim that context is safe to clear until all material information from the session has been recorded in the repository. Do not fill the handoff with routine narration or speculation; preserve verified facts, consequential decisions, and actionable state.

## Required final-response footer

End every final response with exactly one of these status lines:

- `Context safety: SAFE TO CLEAR — durable handoff is current.`
- `Context safety: DO NOT CLEAR — <concise reason>.`

Use `SAFE TO CLEAR` only after the handoff has been updated and verified. “Clear” refers to clearing the Codex conversation/context. Clarify separately if closing a terminal would stop a running local server.

## Security and Git authentication

- Never place API keys, access tokens, passwords, credential material, or customer/private data in source files, documentation, commands whose output is shown, commits, or the handoff.
- Treat the historical OpenAI key exposure documented in the handoff as compromised until Lucas confirms revocation and rotation.
- Avoid interactive Git credential prompts and account-selection pop-ups. Use existing non-interactive credentials (for example, Git Credential Manager with `GCM_INTERACTIVE=never`) for network operations. If authentication is unavailable or expired, stop and tell Lucas instead of opening an interactive authenticator.
- Public read access may be used without credentials. Pushing, deploying, rotating secrets, rewriting history, and other consequential remote actions require explicit user authorization.

## Engineering workflow

- Diagnose before fixing, and address root causes rather than masking symptoms.
- For features and bug fixes, define expected behavior and tests before implementation.
- Keep production and local configuration clearly separated. Never use production customer data or production secrets for local demonstrations.
- Prefer small, reviewable changes with focused tests.
- Run verification appropriate to each change and record the command and outcome in the handoff before reporting completion.
- Never describe tests, builds, or fixes as successful without fresh command output proving the claim.

