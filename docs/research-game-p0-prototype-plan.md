# Research Game P0 Prototype Plan

## Goal

Add a first-pass interactive research island page that uses the selected bitmap assets to validate whether the game direction works inside the current SPA.

## Scope

- Add a dedicated route for the research game prototype.
- Keep the existing course dashboard and learning pages intact.
- Copy selected assets into `public/game/research-archipelago/` for Vite/GitHub Pages serving.
- Build an interactive map with building selection, task summary, researcher avatar, and tutor outfit switching.
- Do not implement backend game state, inventory, experiments, social systems, or asset generation in this pass.

## Phases

- [x] Phase 1: Inspect route/layout patterns and asset paths.
- [x] Phase 2: Copy selected assets into public runtime paths.
- [x] Phase 3: Build the prototype React page and route/nav entry.
- [x] Phase 4: Verify build, lint, and local visual behavior.

## Decisions

- Route: `/research-game`.
- Visual tone: refined low-poly research command surface, denser than a landing page.
- First screen: map interface with selectable buildings and context panel.
- Functional UI remains code-native; bitmap assets provide world/character/building visuals.

## Status

P0 prototype complete. Selected assets are copied under `public/game/research-archipelago/`; the running page is `/research-game`.

## Verification

- `npm run lint`
- `npm run build`
- Playwright desktop and mobile screenshots plus interaction checks.
- Screenshots saved under `output/research-game-p0/` and sent through Feishu.
