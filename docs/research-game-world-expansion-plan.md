# Research Game World Expansion Plan

## Goal

Turn the current collection of small 3D showrooms into a readable exploration
game: the player traverses a larger Commons, discovers research landmarks,
talks to NPC Agents without leaving the game dialogue surface, and enters the
existing research workspaces through world interactions.

## Design Brief

- Player promise: inhabit a calm research community where walking, observing,
  and talking naturally lead to verifiable research work.
- Target feeling: cozy, curious, purposeful, and socially alive.
- Primary verb: point-and-click exploration.
- Secondary verbs: talk, discover, inspect, travel, and operate a workstation.
- Short loop: choose a visible landmark -> walk there -> receive contextual
  feedback -> decide the next destination.
- Session loop: discover the Commons -> consult an NPC -> enter a specialist
  area -> operate an existing ResearchRun workspace -> return with progress.
- Pressure: orientation and incomplete research knowledge, not combat or a
  punitive timer.
- Reward: discovered field notes, clearer next actions, curriculum references,
  and access to verified research work.
- Failure/retry: no hard failure in this slice; unreachable clicks are rejected
  without changing state and the world map always restores orientation.
- Skill expression: efficient route selection and asking increasingly precise
  research questions.
- Non-goals: combat, economy, inventory, authored GLB replacement, voice/audio,
  or changes to ResearchRun, Tutor, presence, profile, and Supabase contracts.

## Core Loop Contract

The player clicks a destination to explore research landmarks while distance
and incomplete information create orientation cost; discoveries and NPC advice
reveal useful next actions, while an invalid route simply leaves the player in
place to choose another path.

## Level Plan

- Expand the Commons from a compact five-building oval into a larger island
  with a central forum, a readable ring road, six destination districts, side
  paths, water/landscape boundaries, and distant scale cues.
- Put the player at the south arrival terrace. The first decision is between
  the guide at the forum and a nearby field-note landmark.
- Use the central beacon, observatory dome, compute tower, paper hall, academy,
  and community pavilion as orientation anchors with different silhouettes.
- Add several optional field-note landmarks along secondary paths. Discovery
  persists locally and is shown as a compact objective tracker.
- Preserve portal travel into separate specialist routes; the four existing
  ResearchRun workspaces remain the progression-bearing destinations.

## Public Case Findings

- OpenAI's Voxel Velocity example succeeds through one tightly specified core
  mode, authored tracks, exact handling rules, opponents, items, a minimap, and
  repeated play-and-test iteration rather than a single visual pass.
- World of ClaudeCraft makes a large procedural world readable through a town
  hub, six surrounding activity hubs, roads, quests, gathering nodes, named
  landmarks, a world map, and a deterministic game core shared by browser,
  server, and automated agents.
- Maze Hunter turns inexpensive procedural geometry into play by combining a
  maze topology, pathfinding enemies, pickups, state reversals, waves, audio,
  and a minimap. Geometry density alone is not its game loop.
- A public Claude Code/SpacetimeDB case shows that multiplayer scope is most
  manageable when a working template owns synchronization and changes are made
  in small, tested increments.
- A community workflow with several published games recommends a durable GDD,
  architecture notes, small implementation steps, and many targeted iterations;
  its author reports that a strong hero asset required about thirty focused
  prompts rather than one broad request.

## Implementation Plan

1. Replace the two-stage NPC flow with one persistent game dialogue surface.
   Keep greeting, chat history, streaming response, curriculum recommendations,
   and structured travel/workspace choices inside that surface.
2. Add field-note interactions and local discovery progress without changing
   backend schemas.
3. Expand the Commons bounds, navigation grid, obstacle layout, portal
   positions, paths, buildings, vegetation, landmarks, and world-map framing.
4. Add a compact objective tracker and map legend that do not obscure the
   player or click targets.
5. Extend browser checks for persistent NPC dialogue, agent questions/actions,
   long-distance A* travel, landmark discovery persistence, map labels, desktop
   and mobile fit, all scene routes, and the 2D fallback.

## Technical Budget

- Desktop target: 1440x900, DPR capped at 1.5, 55+ FPS after warmup.
- Mobile target: 390x844, DPR capped at 1.25, 30+ FPS after warmup.
- Keep repeated vegetation and small props instanced or geometry/material
  shared; no new runtime image download is required for this pass.
- Keep one shadow-casting directional light and avoid new per-landmark dynamic
  lights.
- Target under 180 draw calls and under 350k visible triangles in the expanded
  Commons active view; measure rather than assume.

## Skill And Reference Ledger

- Director: loaded `threejs-game-director/SKILL.md`.
- Gameplay, graphics, UI, debug/profile, and QA phase skills: loaded from the
  local Codex skill set.
- Gameplay workflows, game/level design, UI patterns/checklists, graphics
  architecture/model/render references, debug/performance checklist, and
  QA/visual/playtest/release references: loaded.
- 3D/image/audio generator skills: not installed under the director's sibling
  paths. External generation is out of scope because this pass changes level
  topology, interaction consistency, and repeated support props rather than
  claiming final premium hero assets or audio.

## Acceptance

- An NPC interaction never switches to the old page-like `AgentPanel`; greeting,
  free-form chat, recommendations, and structured actions remain in the same
  dialogue box.
- The Commons traversal area is materially larger and contains a main route,
  optional branches, at least five distinctive destination silhouettes, and
  discoverable side activities.
- A first-time player can identify the current objective and discovery progress
  without reading instructions.
- Desktop/mobile browser checks show a nonblank scene, no overlap or overflow,
  working click navigation, dialogue, discovery, portals, and workspaces.

