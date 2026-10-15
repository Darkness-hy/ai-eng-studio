# Notes: New User Progress Leak

## Findings
- `src/lib/progress.ts` stores all progress under one browser key: `aes:progress:v1`.
- `src/lib/placement.ts` stores placement under one browser key: `aes:placement:v1`.
- `src/lib/sync.ts` initial login merge pulls remote rows into that shared local state, then pushes the full local state to the signed-in account.
- Therefore a newly signed-in user can inherit anonymous or previous-user browser state and upload it as their own progress.
- Fix direction: keep anonymous local state, but switch signed-in users to per-user localStorage keys before cloud sync.
- `src/App.tsx` already gates route rendering while cloud auth is loading, so the main pages should not render anonymous state during initial session lookup.

## Verification
- `npm run build` passed.
- `npm run lint` passed.
- `git diff --check` passed for the touched files.

---

# Notes: Research Game Spatial Navigation

## Current Findings
- The selected world image already depicts a radial campus, but its buildings do not faithfully match all five selected zone assets.
- `ResearchGamePage.tsx` currently overlays each zone as a framed image card and keeps all work panels on the map route.
- The five selected zone assets are available under `public/game/research-archipelago/` as transparent PNG files.
- The current progression state is stored under `research-game:p1-state`; it can be shared across overview and zone routes without a backend migration.
- Desktop hover must be paired with keyboard focus and persistent touch labels for mobile accessibility.

## Implementation Direction
- Generate one 16:9 integrated map using the selected observatory, personal lab, compute workshop, paper workshop, and academic plaza as visual references.
- Keep region labels and status outside the bitmap so localization and accessibility remain possible.
- Move mission actions, tutor guidance, experiment selection, and artifact details to a dedicated zone page.

## Completed Implementation
- Promoted `public/game/research-archipelago/world/integrated-five-zone-campus.png` as the map background.
- Added percentage-based map hotspots with hover, keyboard-focus, status, and touch labels.
- Added `/research-game/:zoneId` for all five zone workspaces.
- Preserved `research-game:p1-state` and extended it with per-zone autosaved notes.
- Renamed tutor UI roles to `研究方法导师`, `科研岛向导`, and `学术交流导师`.
- Saved verification screenshots under `output/research-game-map-v2/`.

## Lossless Image Inventory
- Scope: every PNG under `public/game/research-archipelago/`.
- Count: 11 files.
- Original total: 16,039,265 bytes.
- Delivery format: lossless WebP siblings, retaining PNG sources for rollback.
- Acceptance rule: decoded RGBA pixels and dimensions must match exactly.
- Encoded total: 10,672,690 bytes.
- Reduction: 5,366,575 bytes, or 33.5%.
- Validation: all 11 decoded WebP files are pixel-identical to their PNG source in RGBA space.
- Browser verification: all 10 runtime assets returned `200 image/webp`; no PNG requests or console messages.

## Confirmed ResearchRun Choices
- Use a fixed small CPU experiment template first; do not execute arbitrary user code in P0.
- Build a minimal runner and keep ARIS/AutoResearchClaw behind adapters.
- Keep artifacts private until the user explicitly publishes them.

## ResearchRun Contract Findings
- Existing data access is Supabase-first with `auth.uid()` RLS and `service_role` background agents.
- The browser should never write runner-owned status, metrics, events, verification verdicts, hashes, or public visibility directly.
- A separate Research API must validate the Supabase access token and derive `user_id` from the JWT subject; request bodies never accept `user_id`.
- Five P0 tables are sufficient: runs, hypotheses, experiment jobs, artifacts, and append-only events.
- Public sharing should expose only verified `verification_bundle` or `research_card` artifacts; raw metrics, logs, code, and private notes remain owner/admin-only.
- API POST operations need idempotency keys; run rows also need an integer version for optimistic concurrency.
- The first template should use bundled scikit-learn data with no network dependency and deterministic baseline-versus-ablation analysis.

## ResearchRun Contract Deliverables
- `docs/research-run-design.md` defines the domain model, five-zone mapping,
  fixed CPU protocol, lifecycle, security boundaries, engine adapter, artifact
  verification, and implementation gates.
- `docs/research-run-openapi.json` is the machine-readable OpenAPI 3.1 contract
  for 14 paths and 16 operations, including authenticated SSE resume behavior.
- `supabase/013-research-runs.sql` contains five tables, constraints, indexes,
  read-only client RLS/grants, a private Storage bucket, and atomic
  publication/withdrawal commands restricted to server roles.
- The canonical research question is resolved by template; clients cannot pair
  arbitrary question text with the fixed Wine experiment.
- `templateVersion`, exact estimator parameters, fixed seeds, resource limits,
  command fingerprints, and artifact SHA-256 values preserve reproducibility.
- `analysis_ready` separates completed experiments from active execution.
- Publication does not change run status, but it increments the command version
  and appends an auditable SSE event in the same transaction.

## ResearchRun Contract Verification
- OpenAPI JSON parsing and all local `$ref` resolutions passed.
- Structural checks passed for path parameters, unique operation IDs,
  authentication, idempotency headers, `expectedVersion`, and forbidden owner
  fields in request bodies.
- AJV accepted six valid request/resource fixtures and rejected `userId`, a
  research-question override, and an arbitrary command in experiment config.
- SQL static checks passed for table count, RLS count, no client write policy or
  grant, service-role-only publication RPCs, fixed template/config invariants,
  and SQL/OpenAPI enum parity.
- During the contract-only stage no SQL was executed. The later backend stage
  promoted and exercised the migration in disposable PostgreSQL 16.

## ResearchRun P0 Backend Implementation
- Added `server/research-api/` with FastAPI, Supabase JWT verification,
  RFC 8785 request fingerprints, optimistic concurrency, idempotent commands,
  SSE events, memory/PostgreSQL stores, and all canonical API operations.
- Added a fixed sklearn runner in local-subprocess and production-Docker modes.
  Docker mode enforces no network, read-only root, one CPU, 1 GiB memory, PID
  cap, and a 120-second API timeout.
- Deterministic analysis uses `0.01` practical effect and `4/5` direction
  agreement. Verification requires prediction-hash equality and `1e-9` metric
  tolerance; fit time is ignored.
- The actual Wine result is `meanMacroF1Delta=0.0421466702377266` with `3/5`
  seed agreement, so the honest verdict is `partially_supported`.
- `013` was executed repeatedly in disposable PostgreSQL 16. The full suite
  passed with PostgreSQL/RLS and queue recovery enabled: 14 tests.
- The Docker smoke completed through create, hypothesis, two jobs, analysis,
  verification, publication, and withdrawal with final run version `11` and
  eight artifacts.
- Production Supabase was not modified. The frontend localStorage simulation
  remains unchanged.

## ResearchRun P1 Frontend Integration
- Added a typed browser adapter for all ResearchRun commands and authenticated
  fetch-based SSE. Development bearer tokens are compiled only in Vite DEV
  mode; production connected mode uses the current Supabase access token.
- Added active-run restoration and event-driven snapshot refresh with explicit
  loading, mutation, reconnect, version-conflict, and structured-error states.
- Connected all five regions to real work: topic, accepted hypothesis, fixed
  baseline/ablation queue, deterministic analysis, reproduction verification,
  and research-card publication/withdrawal.
- Added a Vite `/research-api` proxy so the browser and SSE use the same exposed
  frontend origin. API configuration remains opt-in; no API URL means the
  existing `research-game:p1-state` demo continues unchanged.
- SSE now emits an immediate comment frame so proxies flush the stream and the
  frontend reports a live connection without waiting for the 15-second heartbeat.
- Real browser validation used the pinned Docker runner image
  `sha256:60edaaa0788448c52db3dec0c436d8f1e056cb1e6dce87848ed7580915332009`.
  The full flow reached public version 10 with two successful jobs, honest
  `partially_supported` analysis, and a passed reproduction bundle.
- Desktop and 390 px mobile checks found no overflow, broken runtime images,
  page exceptions, or console errors. Screenshots are in
  `output/research-game-p1/`.

## Codebase RAG Capstone ResearchRun
- Replaced the active classification template with `codebase_rag_retrieval_v1`
  from Capstone 02.
- The bundled benchmark contains 24 public-code chunks and 22 golden queries;
  its canonical SHA-256 is
  `88b59cac3d7e94d4eb4f4868962ef9fb01ef482de4f960efe6e988a44a65f544`.
- The fixed variants are `bm25_body`, `bm25_ast`, `hybrid`, and
  `hybrid_rerank` with identical query and Bootstrap inputs.
- Current deterministic results show `hybrid` as the best variant. The final
  reranker slightly reduces MRR@10 relative to hybrid, so the UI and artifact
  preserve it as a negative ablation.
- The full pipeline still improves over body-only BM25 by about `0.3636`
  MRR@10 with `5/5` paired Bootstrap direction agreement.
- Production Supabase remains unchanged.
- Final verification built runner image `0.2.0` at digest
  `sha256:3c066ed5712f0f6ac5c1e3902b610520fe5ea1c4bc20a38c9eea20b271db5f58`.
- Fresh/repeat PostgreSQL migration and RLS suite passed all 14 tests. The real
  Docker HTTP flow completed with verdict `supported`, best variant `hybrid`,
  mean MRR@10 delta `+0.363636`, direction agreement `5/5`, and 8 artifacts.
- Connected desktop/mobile and explicit local-fallback browser checks passed;
  screenshots are in `output/research-game-code-rag/`.
- Persistent preview services now use the new API and runner image at frontend
  port `5181` and backend port `8790`.

## Claude Code / Codex Game Research

- OpenAI's official Codex example, Voxel Velocity, demonstrates that one deep
  mode with authored maps, exact movement rules, opponents, items, minimap/HUD,
  and repeated automated playtesting creates more value than many shallow modes.
- World of ClaudeCraft demonstrates a relevant world structure: a central town,
  six surrounding activity hubs, named roads and landmarks, quests/gathering,
  persistent progress, social presence, and a world map. Its browser, online,
  and automated-agent hosts share one deterministic simulation contract.
- Maze Hunter demonstrates that procedural assets remain viable when topology
  and state are strong: maze routes, pursuit AI, pickups, power reversal, timed
  waves, minimap, lighting, and generated audio all support the core loop.
- The SpacetimeDB multiplayer case recommends starting from proven networking
  infrastructure and iterating in small tested steps instead of asking the
  coding agent for a complete online game in one pass.
- The EnzeD workflow and showcase emphasize a reviewed GDD, architecture notes,
  small testable milestones, and targeted asset iterations. The useful lesson
  is the workflow; self-reported production and popularity claims are not used
  as quality evidence here.
- Applied decision: expand the Commons through routes, landmarks, discovery
  progress, and purposeful destinations. Do not treat raw area or prop count as
  the definition of playability.

## Persistent Dialogue And Commons Expansion Result

- The 3D shell no longer hands NPC conversations to the page-like `AgentPanel`.
  The game dialogue owns intro text, transcript history, streaming Tutor text,
  guide travel, workspace actions, and the 503-course mentor experience.
- Campus navigation now uses a `96 x 68` authored walkable topology. Twelve
  path segments connect the central forum, five destination plazas, and six
  optional research-method landmarks.
- Field notes are saved under `research-game:field-notes-v1:<user-or-guest>`;
  the HUD and world map show only the collected fraction, with no backend or
  Supabase migration.
- The active campus measured 162 draw calls and 32,752 triangles in software
  WebGL. The map overview measured 56 draw calls and 21,918 triangles after
  merged roads, simplified destination silhouettes, and map-only shadow gating.
- Production-preview browser QA passed for desktop and mobile. Evidence is in
  `output/research-game-world-expansion-final/`.

## Ground Flicker Reproduction

- Reproduced on the persistent `5181` preview at `1440 x 900` with the player
  fixed at world position `(0, 12)` and a settled follow camera.
- Canvas CSS and drawing-buffer sizes both measured `1440 x 900`; the page had
  no console or runtime errors.
- The flicker remained under `prefers-reduced-motion: reduce`, which disables
  water and tree movement. Consecutive-frame differences still concentrated on
  the central plaza and road overlap.
- Root cause: `CampusPaths` places the road surface and `PlazaSurface` circle on
  the same world Y plane. The depth buffer alternates between their different
  materials; path ribbons also use path-local UVs at shared intersections.
- Baseline evidence: `output/ground-flicker-debug/normal-*.png`,
  `reduced-*.png`, and the amplified `reduced-diff-0-1.png`.

## Ground Flicker Fix Result

- Ground layers now use increasing visual heights and deterministic render
  order: road edge, road surface, plaza border, then plaza surface. Negative
  polygon offsets provide an additional depth-buffer margin across GPUs.
- Road ribbon UVs now derive from world X/Z coordinates, eliminating texture
  disagreement where independently authored routes overlap.
- Controlled reduced-motion comparison improved from mean absolute difference
  `3.623` / changed pixels `6.27%` to `0.012` / `0.05%`. The independent full
  production QA measured `0.067` / `0.34%`, within the regression thresholds
  of `0.5` / `1%`.
- `output/ground-flicker-final/desktop-ground-stability-before.png` shows the
  corrected surface; `desktop-ground-stability-diff.png` is nearly black.
# Readable Research Game UI And DeepSeek NPCs - 2026-07-17

- Live tutor health before changes: ready, provider `deepseek`, model
  `deepseek-v4-pro`, max concurrency 3, inflight 0.
- Root issue: both Research Game dialogue implementations call `askTutor`, but
  the backend always applies the Cici system identity. NPC identity currently
  exists only in browser-provided context and is therefore neither authoritative
  nor reliably independent.
- The persistent `5181` Vite service has Research API variables but no tutor
  endpoint. Add a same-origin `/tutor-api` Vite proxy and service environment so
  local tunneled testing uses the host tutor service without CORS or mixed-origin
  problems.
- Current game labels and dialogue copy frequently use 8-12.5 px text. Raise
  ordinary copy to roughly 14-16 px, preserve 11-12 px only for true metadata,
  and increase fixed dimensions where needed so labels do not reflow controls.
- Dialogue portraits are currently 170 x 210 px desktop and 78-100 x 136-164 px
  mobile. Allocate a larger stable portrait column and allow the transparent
  image to extend above the dialogue surface while keeping text unobstructed.
- Implemented server-owned persona IDs for 林澈、苏老师、周岚、程野、顾言、许禾.
  NPC requests use DeepSeek with thinking disabled and persona caps of 180-280
  tokens; ordinary Cici tutor requests are unchanged.
- Real smoke result for `campus-guide`: 67 Chinese characters, identified as
  林澈/科研岛向导, no 茜茜 identity leakage, HTTP/SSE complete.
- Vite preview now proxies `/tutor-api` to `127.0.0.1:8787`; persistent service
  drop-in supplies both chat and health frontend variables.
- Final browser evidence confirms 14-17 px dialogue copy, 350+ px desktop
  portrait bounds, 220+ px mobile portrait bounds, no mobile overflow, and
  distinct `campus-guide` / `research-mentor` request payloads.
