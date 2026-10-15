# Research Game P1 Vertical Slice Plan

## Goal

Turn the P0 research island into a minimal playable loop that can be completed in the browser without backend services.

## Scope

- Add a sequential task chain: observatory -> personal lab -> compute workshop -> paper workshop -> academic plaza.
- Track each zone as locked, ready, active, or complete.
- Add a code-native research artifact backpack.
- Add a simulated experiment interaction in the compute workshop.
- Add fixed AI tutor guidance per zone and progress stage.
- Persist progress with `localStorage`.
- Keep the existing route `/research-game` and selected bitmap assets.

## Non-goals

- No Supabase persistence in this pass.
- No real backend experiment execution.
- No new generated bitmap assets.
- No replacement of the existing learning dashboard.

## Phases

- [x] Phase 1: Model the task chain and persistent game state.
- [x] Phase 2: Replace the P0 task buttons with playable progression.
- [x] Phase 3: Add the experiment simulator and artifact backpack.
- [x] Phase 4: Verify lint, build, desktop/mobile UI, and localStorage behavior.

## Verification

- `npm run lint` passed.
- `npm run build` passed. Vite still reports the existing large chunk warning.
- Python Playwright desktop full loop passed: complete all five zones, reach `5/5`, show reset action, and preserve completion after reload.
- Python Playwright mobile smoke passed: reset localStorage, complete the first task, and advance to `个人实验室`.
- Browser console warnings/errors during the Playwright checks: 0 desktop, 0 mobile.
- Screenshots saved under `output/research-game-p1/`.
- Screenshot contact sheet sent to Feishu.

## Errors Encountered

- `python` was not installed for local screenshot contact-sheet generation. Retried with `python3`, which succeeded.
- The repo does not include a local Node `playwright` package, so the direct `require('playwright')` check failed. Retried with the already installed Python Playwright API, which passed.

## Status

P1 complete.
