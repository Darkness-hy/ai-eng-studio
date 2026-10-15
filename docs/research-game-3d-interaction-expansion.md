# Research Game 3D Interaction Expansion

## Player Interaction Contract

- Primary desktop input: left click selects a destination.
- Camera input: right-button drag or middle-button drag orbits; wheel zooms.
- Clicking an NPC, workstation, or portal creates an approach command. The
  interaction activates only after the player reaches its interaction radius.
- A new click replaces the active path. Escape cancels overlays and the current
  destination. Mobile retains the joystick and also accepts tap-to-move.

## Navigation

- Use PathFinding.js A* over a per-scene 0.5 m grid.
- Scene definitions own rectangular navigation obstacles matching their Rapier
  colliders. Start and goal cells are projected to the nearest walkable cell.
- Path points are compressed to direction changes. Ecctrl continues to own
  collision, grounding, acceleration, and body rotation.

## NPC Interaction

- Stage 1 is a compact bottom dialogue box over the live 3D scene. It includes
  the NPC portrait, identity, greeting, one primary continue action, and exit.
- Stage 2 opens the existing detailed Agent or curriculum interface. This keeps
  current tutoring and structured-action behavior while making the encounter
  feel like an in-world conversation.

## Scene Set

- Commons: upgraded civic research campus with stone, concrete, glass, metal,
  planting, curbs, glazing, facade depth, and restrained colors.
- Academy: research methods studio and curriculum mentor.
- Research: project terminal, evidence wall, hypothesis table, methods mentor.
- Compute: server aisle, experiment console, status lighting, engineer Agent.
- Paper: manuscript tables, evidence shelves, review board, reviewer Agent.
- Plaza: poster pavilion, presentation boards, seating, community host Agent.

## Verification Targets

- Desktop 1440x900 and mobile 390x844 render nonblank without overflow.
- Click movement changes world position and camera position.
- Right-drag camera orbit does not move the player.
- All six scenes render distinct canvas pixels and expose the expected Agent and
  workspace interactions.
- The 2D query fallback remains functional.
