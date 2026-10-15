# Research Game 3D Vertical Slice

## Goal

Ship a playable 3D graybox for the Research Commons and Research Academy while
preserving the current Phaser world as a fallback. The default play camera
follows the local player; a separate world-map mode shows the whole island.

## Player Promise

The player inhabits a calm low-poly research campus, explores it as a character,
and obtains concrete research guidance by interacting with people and places.

## First Playable Loop

1. Spawn in the Research Commons.
2. Move with camera-relative controls and identify nearby landmarks.
3. Talk to the campus guide or open the world map.
4. Reach the Research Academy entrance.
5. Enter the academy and talk to the curriculum-grounded mentor.
6. Select a research-training path and return to exploration.

Falling or leaving the island resets the player to the current scene spawn. The
slice has no combat, timer, economy, or punitive failure state.

## Camera Contract

- `follow`: perspective third-person camera, smoothly damped behind and above
  the player. Movement remains camera-relative and the player stays in the
  lower-middle part of the viewport.
- `map`: orthographic elevated overview of the full current scene. Player movement
  pauses. The map shows the player, destination labels, and the current target.
- `dialogue`: softly framed player/NPC view while the existing React dialogue
  panel is open.
- `workstation`: reserved for later object-focused ResearchRun interaction.

Desktop controls are WASD/arrows, pointer drag to orbit, `E` to interact, and
`M` to toggle the map. Mobile gets a fixed joystick, interaction button, and map
button with stable touch targets.

## Scene Plan

### Research Commons

- Player spawn on the south-central arrival path.
- Campus guide beside the spawn as the first optional interaction.
- Research Academy is the primary destination and strongest landmark.
- The research lab, compute center, paper studio, and plaza remain visible as
  authored landmarks; their portals route to the existing 2D scenes in this
  slice.
- Ocean, foliage, lights, and NPC idle motion provide restrained ambient life.

### Research Academy

- A distinct interior scene with a mentor, lesson terminals, and an exit portal.
- The existing 503-lesson mentor and AgentPanel remain the interaction source of
  truth; the 3D scene only supplies movement, proximity, and framing.

## Architecture

- New 3D code lives under `src/game3d/`; the existing `src/game/` Phaser runtime
  remains intact.
- `ResearchGamePage` selects 3D for `campus` and `academy` by default. Query
  parameter `renderer=2d` forces the current Phaser shell. Other scenes continue
  to render with Phaser until migrated.
- Three.js rendering uses React Three Fiber and Drei. Rapier provides collision;
  Ecctrl provides the character-controller foundation.
- The existing profile and Supabase Presence contracts remain two-dimensional.
  A deterministic adapter maps 3D `(x, z)` scene coordinates to legacy `(x, y)`
  coordinates, so migration `015` and production data are unchanged.
- Existing ResearchRun, Tutor, academy content, profile, and presence modules
  remain authoritative. The 3D layer does not duplicate their business state.

## Module Boundaries

- `ResearchGame3DShell.tsx`: profile, route, HUD, overlays, interaction state.
- `ResearchGame3DCanvas.tsx`: Canvas lifecycle, loading, diagnostics, camera mode.
- `world3d.ts`: scene definitions, spawns, portals, NPCs, and coordinate bounds.
- `components/ResearchIslandWorld.tsx`: authored procedural low-poly environment.
- `components/PlayerAvatar3D.tsx`: local character and walk/idle presentation.
- `components/RemoteAvatar3D.tsx`: presence-driven remote character.
- `components/CameraRig.tsx`: follow/map/dialogue camera state machine.
- `coordinates.ts`: reversible 3D-to-legacy profile/presence conversion.

## Performance Budget

- Desktop target: 60 FPS; mobile floor: 30 FPS on the test viewport.
- Pixel ratio cap: 1.5 desktop, 1.25 mobile.
- Draw calls: at most 150 desktop and 100 mobile in the Commons.
- Visible triangles: at most 200,000 desktop and 100,000 mobile.
- One 1024 px shadow map and a bounded set of important shadow casters.
- No external GLB or large texture is required for the first load; the slice uses
  authored procedural geometry and the existing role palettes.

## Acceptance Checks

- 3D Canvas is nonblank on 1440x900 and 390x844 viewports.
- Keyboard and mobile input visibly change the player's position and animation.
- The follow camera tracks movement without clipping through world geometry.
- `M`/map button shows the complete scene, labels, and current player position;
  movement pauses until map mode closes.
- `E`, click, and touch can interact with the guide and academy mentor.
- Crossing the academy portal changes the URL and loads a visibly different 3D
  scene; refresh preserves the scene and position.
- `?renderer=2d` loads the unchanged Phaser implementation.
- No horizontal overflow, UI overlap, broken requests, page errors, or console
  errors on the two target viewports.
- `window.__THREE_GAME_DIAGNOSTICS__` reports renderer, camera, player, draw-call,
  triangle, and interaction state for browser verification.

## Boundaries And Deferred Work

- No production Supabase migration or ResearchRun/Tutor backend change.
- No authoritative multiplayer, inventory, economy, trading, combat, or voice.
- No full 3D migration of research, compute, paper, or plaza in this slice.
- No claim of production-quality art. Image-to-3D assets, audio, VFX, and final
  character rigs are deferred until the graybox interaction and camera pass.

## Skill And Asset Ledger

- Planning/directing: `threejs-game-director`.
- Controller, collision, interaction: `threejs-gameplay-systems`.
- Procedural low-poly rendering: `threejs-aaa-graphics-builder`.
- HUD and mobile controls: `threejs-game-ui-designer`.
- Runtime diagnostics and performance: `threejs-debug-profiler`.
- Desktop/mobile browser gate: `threejs-qa-release`.
- External generated assets: none in this graybox; procedural geometry is
  authored in the repository. Audio and generated 3D assets are explicitly
  deferred.
