# Playable Research World P0

## Goal

Replace the static hotspot map with a persistent player-controlled research
world. Merge topic/data analysis and hypothesis work into one research lab, and
add a research academy for learners without systematic research training.

## World

The game has one social hub and five destination scenes:

1. `campus`: Research Commons lobby and five portals.
2. `academy`: Research Academy with the curriculum-grounded mentor agent.
3. `research`: Model and Systems Research Lab; topic, provenance, and hypothesis.
4. `compute`: fixed experiment runner and recovery.
5. `paper`: analysis, reproduction, review, and paper artifacts.
6. `plaza`: publication, reactions, and social presence.

The existing ResearchRun state machine remains authoritative. Game scenes and
NPC dialogue do not duplicate measured results or run transitions.

## P0 Deliverables

- Phaser canvas embedded in the existing React/Vite application.
- Full-screen game route with click-to-move, WASD, camera, bounds, portals,
  proximity interaction, and scene transitions.
- Six persistent research-role presets using the current low-poly character as
  the temporary P0 body asset.
- One mentor NPC per scene and interactive research objects.
- Research Academy lesson paths and local retrieval across all 503 lessons.
- Optional AI mentor chat grounded in retrieved course content through the
  existing tutor service.
- Embedded ResearchRun workspaces opened from objects in the research scenes.
- Local-first `game_profiles` persistence and optional Supabase sync.
- Supabase Presence for membership and Broadcast for movement/emote events.

## Boundaries

- P0 character presets share a temporary body asset and differ by research
  role, color, and identity badge. Production animation atlases remain an asset
  follow-up.
- Movement is client-authoritative and suitable only for a noncompetitive
  social world. Trading or competitive mechanics require an authoritative game
  server.
- NPCs propose actions; ResearchRun mutations still require explicit player
  commands and server validation.
- No production Supabase migration is applied in this pass.

## Verification

- TypeScript build and strict ESLint.
- Canvas pixel checks, movement, portal entry, NPC interaction, and overlay
  behavior with Playwright on desktop and mobile.
- Local fallback without Supabase, tutor, or Research API.
- Connected ResearchRun workspace when the API is configured.

## P0 Result

- The former observatory and personal lab are one `research` scene with two
  in-world workstations: project/data terminal and hypothesis whiteboard.
- The new `academy` scene contains a research mentor grounded in four curated
  learning paths and local retrieval over the full 503-lesson catalog.
- First entry creates one of six role presets. The same profile persists locally
  per user and can sync to `game_profiles` after migration 015 is applied.
- Supabase Presence tracks scene membership; Broadcast carries transient player
  positions and wave events. No global student count is exposed.
- Initial Phaser loading is scene-scoped. The lobby does not prefetch the five
  destination backgrounds.

## Deferred

- The six P0 roles share one temporary body image. Directional walk animation,
  modular clothing, and animation atlases are P1 assets.
- P0 collision prevents leaving the world bounds. Interior obstacle maps and
  pathfinding are deferred because click-to-move must route around them rather
  than stop against furniture.
- P0 movement is intentionally noncompetitive. Shared experiments, inventory,
  trading, and authoritative simulation require a dedicated game server.
