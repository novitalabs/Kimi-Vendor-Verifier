# Kimi-Vendor-Verifier Fork

This is Novita's fork of `MoonshotAI/Kimi-Vendor-Verifier`.

- Read sibling `../AGENTS.md` for workspace ownership and task routing.
- Upstream-owned files stay in their original paths and are synced from Moonshot.
- Keep this repository limited to upstream source plus minimal fork metadata and
  deliberate upstream-sync commits.
- Do not add Novita tests, profiles, orchestration, acceptance reporting, Code
  Bench, or deepSWE adapters here. They are owned by sibling
  `../novita-self-test/`.
- Preserve `upstream/main` ancestry and the existing `origin`/`upstream` remotes.
- Run the official 611-case collection gate after upstream or fixture changes.

No dynamic endpoint, credential, or run evidence belongs in this repository.
