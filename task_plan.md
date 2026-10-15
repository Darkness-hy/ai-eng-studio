# Task Plan: Research Island HUD Style Directions

## Goal
Generate four landscape 3D game HUD mockups and copy the final PNGs to the requested visual-style-options directory.

## Phases
- [x] Generate A Academic Editorial with one built-in image generation call.
- [x] Generate B Calm Atmospheric with one built-in image generation call.
- [x] Generate C Research Terminal with one built-in image generation call.
- [x] Generate D Architectural Exhibition with one built-in image generation call.
- [x] Copy, inspect, and list all four final PNG files.

## Verification
- All four files are PNG, 1672 x 941, 8-bit RGB, non-interlaced.
- Each scheme used exactly one built-in image generation call.

---

# Task Plan: New User Progress Leak

## Goal
Fix the bug where a brand-new user sees nonzero learning progress and placement results.

## Phases
- [x] Phase 1: Identify where 20/503 learning progress is computed.
- [x] Phase 2: Identify where 10/50 placement result is computed.
- [ ] Phase 3: Patch the source so signed-in new users do not inherit browser-local progress.
- [x] Phase 4: Verify with build/lint and targeted code checks.

## Key Questions
1. Are progress and placement read from Supabase, localStorage, or both?
2. Does the app merge anonymous/local progress into a signed-in user's view?
3. Should a newly signed-in user start from cloud-only data unless explicit import is requested?

## Decisions Made
- Signed-in users should use per-user localStorage keys. Anonymous progress remains under the existing anonymous key.
- Cloud sync should merge only the current user's scoped local cache with that user's remote rows, never the shared anonymous cache.

## Errors Encountered
- Build-output literal grep for `aes:progress:v1:user:` returned false because the bundler minified/split template strings. Source-level check confirms the user-scoped key generation is present.

## Status
Implementation and local verification complete.

---

# Task Plan: Research Game Spatial Navigation

## Goal
Replace the five floating zone cards with one integrated research-island map, make each map region interactive, and move zone work into dedicated pages.

## Scope
- Frontend and game-map asset only.
- Preserve the existing local P1 progression and simulated experiment behavior.
- Discuss the unified ResearchRun API and minimum vertical slice; do not implement backend APIs in this pass.

## Phases
- [x] P0: Inspect the selected zone assets, current map, routes, and persisted game state.
- [x] P0: Generate and promote a new integrated five-zone world map.
- [x] P0: Replace map cards with responsive hover/focus hotspots.
- [x] P0: Add dedicated zone routes and move each zone's work surface onto its own page.
- [x] P0: Rename awkward tutor outfit labels.
- [x] P0: Run lint, build, and targeted desktop/mobile Playwright checks.
- [x] P1: Present the ResearchRun API and minimum vertical-slice design for user review.

## Decisions
- Every region can be opened from the map; progression locks actions, not navigation.
- Hotspots use percentage geometry over a fixed-aspect map so desktop and mobile stay aligned.
- The generated bitmap contains no labels or UI. Names, status, focus, and hover treatment remain code-native.
- The backend design remains adapter-based so the frontend is not coupled to ARIS or AutoResearchClaw file names.

## Verification
- `npm run lint` passed.
- `npm run build` passed; the existing large-chunk advisory remains unchanged.
- `git diff --check` passed.
- Playwright desktop flow passed: five hotspots, hover highlight, zone routing, note persistence, completion persistence, and map progress.
- Playwright mobile checks passed at 390 x 844: five hotspots, no horizontal overflow, locked-zone entry, and disabled locked action.
- Browser console warnings/errors: 0.

---

# Task Plan: Research Game Lossless Image Delivery

## Goal
Reduce research-game image transfer size without changing decoded pixels.

## Scope
- Convert all 11 PNG files under `public/game/research-archipelago/` to lossless WebP siblings.
- Keep original PNG files for rollback in this pass.
- Update runtime references to WebP only after pixel-equality validation.

## Phases
- [x] P0: Inventory image count, byte size, runtime references, and available codecs.
- [x] P0: Encode lossless WebP siblings with alpha preservation.
- [x] P0: Validate dimensions, modes, and decoded RGBA pixel equality.
- [x] P0: Update frontend asset references.
- [x] P0: Verify lint, build, HTTP delivery, and browser rendering.

## Result
- Original PNG total: 16,039,265 bytes.
- Lossless WebP total: 10,672,690 bytes.
- Saved: 5,366,575 bytes (33.5%).
- Pixel equality: passed for all 11 assets after RGBA decoding.
- Runtime: 10 WebP assets requested, 0 PNG requests, 0 failed responses, 0 console warnings/errors.

## Confirmed ResearchRun Decisions
- P0 experiment: fixed `scikit-learn` CPU template.
- Orchestration: minimal in-house runner behind a `ResearchEngine` adapter.
- Research artifacts: private by default, published only by explicit user action.

---

# Task Plan: ResearchRun Contract Design

## Goal
Produce reviewable contracts for the first real research vertical slice without starting a backend service or applying database changes.

## Deliverables
- `docs/research-run-design.md`: domain model, state machine, security boundaries, fixed CPU experiment, and engine adapter.
- `docs/research-run-openapi.json`: OpenAPI 3.1 HTTP/SSE contract.
- `supabase/013-research-runs.sql`: Supabase tables, constraints, indexes, RLS,
  grants, and explicit publication RPCs.

## Phases
- [x] P0: Inspect existing Supabase, RLS, auth, SSE, and server conventions.
- [x] P0: Define ResearchRun states, invariants, artifacts, and failure semantics.
- [x] P0: Write and validate the OpenAPI contract.
- [x] P0: Write and statically audit the Supabase DDL/RLS draft.
- [x] P0: Cross-check API schemas against SQL columns and document unresolved implementation choices.

## Boundaries
- Do not create `server/research-api/` in this pass.
- Do not execute SQL against production Supabase during contract design.
- Do not run generated or user-supplied code.

## Result
- The P0 template is fixed to `sklearn_wine_logreg_v1` version `1.0.0`; the
  server supplies the canonical research question and exact experiment config.
- The normalized lifecycle includes `analysis_ready` so completed jobs are not
  represented as still running.
- All command endpoints use optimistic concurrency; command-style POST/DELETE
  operations also require idempotency keys.
- Five owner-scoped tables cover runs, hypotheses, job attempts, artifacts, and
  append-only events.
- Browser roles are read-only. Publication is an explicit, atomic,
  `service_role`-only command with private-by-default artifacts.

## Verification
- OpenAPI JSON parse passed.
- All 141 local OpenAPI references resolved.
- Structural checks passed for 14 paths and 16 unique operations.
- AJV accepted six valid fixtures and rejected injected `userId`, question
  override, and arbitrary experiment-command fields.
- SQL static audit passed for five tables, five SELECT-only RLS policies,
  client grants, publication RPC boundaries, fixed config, and API enum parity.
- SQL was not executed and no PostgreSQL parser is installed in this workspace.

---

# Task Plan: ResearchRun P0 Backend Vertical Slice

## Goal
Implement the approved topic-to-verified-artifact workflow as an independent,
locally runnable Research API without applying changes to production Supabase.

## Confirmed Evidence Rules
- `increase` is supported when mean paired macro-F1 delta is at least `0.01`
  and at least four of five seeds agree with the direction.
- Reproduction requires identical prediction hashes and macro F1/accuracy
  within `1e-9`; fit time is diagnostic only.

## Phases
- [x] P0: Promote the reviewed schema to an idempotent `013` migration.
- [x] P0: Implement fixed template, isolated runner, deterministic analysis,
  and verification.
- [x] P0: Implement auth, state machine, idempotency, events, artifacts, and the
  OpenAPI endpoints in a standalone FastAPI service.
- [x] P0: Implement an in-memory test store and PostgreSQL/Supabase transaction
  store behind one interface.
- [x] P0: Add unit, API-contract, state-transition, and end-to-end CPU tests.
- [x] P0: Start the service locally and run a real vertical-slice smoke test.
- [ ] P1: Replace the frontend simulation with the API adapter after backend
  behavior is verified.

## Boundaries
- Do not apply SQL to production Supabase in this pass.
- Do not reuse the Tutor service process or credentials.
- Do not accept source code, shell commands, package names, URLs, or arbitrary
  experiment configuration from clients.
- Do not remove the existing localStorage research-game flow yet.

## Result
- All 16 canonical HTTP/SSE operations are implemented in
  `server/research-api/`.
- Local and PostgreSQL stores share the same state-machine interface.
- The PostgreSQL migration applied successfully twice in disposable PostgreSQL
  16; owner RLS, anonymous safe-column reads, and client write denial passed.
- The fixed Docker runner produced the same prediction hashes as the local
  runner under no-network/read-only/CPU/memory/PID restrictions.
- Full test suite: 14 passed with PostgreSQL integration enabled.
- Real HTTP/Docker smoke: completed, 8 artifacts, honest verdict
  `partially_supported`, publish/withdraw successful, final run version 11.
- Local Research API is running at `http://127.0.0.1:8790` in development
  memory-store mode. Production Supabase and frontend behavior are unchanged.

---

# Task Plan: ResearchRun P1 Frontend Integration

## Goal
Drive all five research-island workspaces from the real ResearchRun API while
retaining the current localStorage experience as an explicit demo fallback.

## Phases
- [x] P0: Add typed API client, Supabase/dev token resolution, idempotency, and
  authenticated SSE parsing.
- [x] P0: Add a ResearchRun hook with active-run persistence, event refresh,
  command states, and structured errors.
- [x] P0: Map API run states and artifacts onto the five map hotspots.
- [x] P0: Implement connected topic, hypothesis, experiment, analysis,
  verification, retry/cancel, and publication workspaces.
- [x] P0: Add a Vite same-origin development proxy and preserve the unconfigured
  local demo path.
- [x] P0: Run TypeScript lint/build plus connected and fallback Playwright flows.

## Boundaries
- Do not apply the Supabase migration or deploy the Research API in this pass.
- Do not expose the development token in production builds.
- Do not remove `research-game:p1-state` or change the fallback behavior.
- Do not implement a public feed, likes, rankings, or arbitrary experiments.

## Result
- `VITE_RESEARCH_API_URL` explicitly enables connected mode; without it, the
  original localStorage workflow renders unchanged.
- Connected mode restores the current owner's latest run, streams authenticated
  SSE events, enforces optimistic versions, and surfaces structured API errors.
- The five dedicated workspaces now execute the complete fixed workflow from
  topic creation through verified research-card publication.
- Vite proxies `/research-api` to `127.0.0.1:8790`, so remote browser access
  needs only the frontend port tunnel during development.
- The real Docker-runner browser flow completed with two successful jobs,
  `partially_supported`, reproduction passed, publication succeeded, and map
  progress `5/5`.

## Verification
- `npm run lint`: passed.
- `npm run build`: passed; existing large-chunk warnings remain.
- Backend Ruff: passed.
- Backend tests: 13 passed, 1 PostgreSQL integration test skipped.
- Connected Playwright flow: passed on desktop and 390 px mobile, with no page
  errors, console errors, broken images, or horizontal overflow.
- Explicit fallback Playwright flow: passed and made zero Research API requests.
- Screenshots: `output/research-game-p1/`.

---

# Task Plan: Codebase RAG Capstone ResearchRun

## Goal
Replace the user-facing Wine classification example with Capstone 02,
`codebase_rag_retrieval_v1`, as a real, reproducible code-retrieval research
project across all five research-island zones.

## Experiment Contract
- Use a versioned, bundled public-code corpus and a fixed golden query set.
- Compare four cumulative configurations: fixed-window BM25, AST-aware BM25,
  AST-aware hybrid retrieval, and hybrid retrieval plus reranking.
- Primary metric: MRR@10. Secondary metrics: nDCG@10, Recall@10, citation
  precision, and query latency.
- Use identical queries for every variant and deterministic bootstrap seeds for
  uncertainty and paired comparisons.

## Phases
- [x] P0: Audit and remove Wine-specific public contracts.
- [x] P0: Build and hash the Code RAG corpus, golden queries, and retrieval
  implementations.
- [x] P0: Update runner, analysis, verification, artifacts, and OpenAPI.
- [x] P0: Adapt all five connected research-island workspaces.
- [x] P0: Verify backend, disposable PostgreSQL, runner image, frontend, and
  desktop/mobile browser flows.
- [x] P0: Restart the persistent Research API and frontend preview services.

## Result
- Active template: `codebase_rag_retrieval_v1` version `1.0.0`.
- Benchmark: 24 bundled public-code chunks, 22 golden queries, four fixed
  retrieval configurations, and five paired Bootstrap samples.
- Docker runner image: `ai-eng-studio-research-runner:0.2.0`, local digest
  `sha256:3c066ed5712f0f6ac5c1e3902b610520fe5ea1c4bc20a38c9eea20b271db5f58`.
- Real vertical slice: `supported`, best variant `hybrid`, mean full-pipeline
  versus body-only MRR@10 delta `+0.363636`, direction agreement `5/5`.
- The `hybrid_rerank` stage remains visible as a negative `-0.019697`
  incremental ablation relative to `hybrid`.
- Production Supabase was not modified.

## Verification
- Frontend `npm run lint` and `npm run build` passed.
- Backend Ruff passed; memory tests passed `13 passed, 1 skipped`.
- Fresh and repeat migration plus PostgreSQL/RLS tests passed `14 passed`.
- The no-network, read-only Docker runner completed all four variants and the
  HTTP smoke completed experiment, analysis, reproduction, publish, and
  withdrawal.
- Connected Playwright checks passed on 1440 px desktop and 390 px mobile with
  zero browser errors, failed responses, broken images, or horizontal overflow.
- Explicit local fallback passed with zero Research API requests.
- Screenshots: `output/research-game-code-rag/`.

---

# Task Plan: Real CodeSearchNet ResearchRun

## Goal
Upgrade the current curated Code RAG teaching benchmark into a provenance-auditable,
zero-paid-API retrieval experiment built from real source code and independent
human relevance judgments.

## Experiment Contract
- Source benchmark: GitHub CodeSearchNet commit
  `106e827405c968597da938f6b373d30183918869`.
- Language: Python for the first real vertical slice so structural fields can
  be derived with the standard-library `ast` parser.
- Queries and relevance: the official CodeSearchNet human annotation store,
  using graded relevance `0..3` instead of locally authored labels.
- Corpus: source snippets fetched from commit-pinned GitHub URLs at build time,
  frozen into the runner image with URL and SHA-256 provenance.
- Retrieval runtime: local BM25, local character n-gram LSA, RRF, and a
  deterministic AST-feature reranker; no paid or remote API calls.
- Primary metric: graded `nDCG@10`. Secondary metrics: MRR@10 using positive
  mean relevance, Recall@10, top-3 citation precision, and diagnostic latency.
- Uncertainty: direct query-set point estimates plus 1,000 paired bootstrap
  replicates and a 95% confidence interval.

## Phases
- [x] P0: Freeze official provenance and build a reproducible source snapshot.
- [x] P0: Replace curated summaries with actual code and AST-derived fields.
- [x] P0: Upgrade metrics, paired bootstrap, claim rules, verification, SQL,
  OpenAPI, and tests.
- [x] P0: Update all connected workspaces and clearly label fallback scores as
  a local demonstration.
- [x] P0: Verify memory/PostgreSQL stores, Docker reproduction, frontend build,
  and desktop/mobile browser behavior.
- [x] P0: Restart persistent preview services with the new immutable image ID.

## Boundaries
- No network access inside experiment or verifier containers.
- No OpenAI, embedding, reranking, generation, or judge API calls.
- No user repositories, paths, URLs, packages, or arbitrary experiment config.
- No production Supabase migration or deployment in this pass.
- Build-time source failures must be reported; they must not silently become
  fabricated snippets or labels.

## Result
- Template `codebase_rag_retrieval_v1` is now version `2.0.0` over a frozen
  CodeSearchNet Python human-evaluation snapshot: 938 real snippets, 85 test
  queries, and 738 repository/commit pairs.
- Primary result: full pipeline nDCG@10 `0.830193` versus body-only BM25
  `0.794021`, delta `+0.036172`, paired Bootstrap 95% CI
  `[0.013880, 0.057347]`, verdict `supported`.
- Best variant is `hybrid` at nDCG@10 `0.835275`; the reranker remains visible
  as a negative incremental ablation of `-0.005082`.
- Runtime API cost is `$0`: all retrieval, AST parsing, reranking, analysis,
  and verification are local and the runner network is disabled.
- Immutable local runner image ID:
  `sha256:343baa568dec35151515588e80f7597efd4683d73a0e74085515c162c33e56f5`.
- Production Supabase was not modified.

## Verification
- Backend Ruff passed; memory tests passed `14 passed, 1 skipped`.
- Fresh and repeat migration plus PostgreSQL/RLS tests passed `15 passed`.
- The restricted Docker runner completed all four variants by immutable image
  ID; the HTTP smoke completed experiment, analysis, reproduction, publish,
  and withdrawal.
- Frontend lint and production build passed. The existing large-chunk warning
  remains unchanged.
- Connected Playwright checks passed on 1440 px desktop and 390 px mobile with
  no browser errors, failed responses, broken images, or horizontal overflow.
- Explicit fallback checks passed on desktop and mobile with zero Research API
  requests.
- Screenshots: `output/research-game-codesearchnet/`.
- Persistent preview services: web `0.0.0.0:5181`, API `127.0.0.1:8790`.

---

# Task Plan: Zero-API Capstone Portfolio

## Goal
Add a real, reproducible ResearchRun vertical slice for every remaining
top-level capstone project that has a measurable offline subsystem, while
keeping paid API cost at exactly `$0`.

## Interpretation
- Implement bounded research experiments, not pretend that external production
  deployments, human studies, or subjective reviews have already happened.
- Keep Capstone 02 as the real CodeSearchNet benchmark already completed.
- Add one offline experiment for each of Capstones 01 and 03-17:
  harness recovery, VAD robustness, document evidence retrieval, bounded search,
  Kubernetes RCA, local fine-tuning, ACL-aware RAG, migration recipes,
  multi-agent coordination, trace regression detection, temporal video search,
  MCP governance, speculative decoding, safety gating, patch sandboxing, and
  Bayesian knowledge tracing.
- Every template must expose fixture/public provenance, a primary metric,
  paired uncertainty, explicit limitations, and runtime cost metadata.

## Phases
- [x] P0: Audit all 17 top-level capstones and freeze the zero-API experiment matrix.
- [x] P0: Generalize ResearchRun models, registry, runner, analysis, and verifier
  from one retrieval template to multiple metric-compatible templates.
- [x] P0: Implement and test all 16 additional offline benchmark packs.
- [x] P0: Add capstone selection and dynamic five-zone content to the frontend.
- [x] P0: Update SQL, OpenAPI, design docs, and migration/reproduction tests.
- [x] P0: Verify every template locally and in the restricted Docker runner,
  then run desktop/mobile browser checks.
- [x] P0: Restart persistent preview services with the new immutable runner image.

## Boundaries
- No paid model, embedding, speech, vision, judge, reranking, or generation API.
- No network inside experiment or verifier containers.
- No fabricated claim of production readiness or human-study completion.
- No production Supabase migration in this pass.
- Synthetic fixtures are allowed only when deterministic, versioned, disclosed,
  and appropriate for a systems property with known ground truth.

## Result
- Added 16 offline benchmark packs and retained the real CodeSearchNet benchmark,
  for 17 top-level capstones and 52 fixed variants total.
- All templates use generic case metrics, 1,000 paired Bootstrap replicates,
  explicit effect thresholds, deterministic result hashes, limitations, and
  external-validation fields.
- Current offline verdicts are 14 `supported` and 3 `partially_supported`;
  Capstones 03, 05, and 06 remain partial. Intermediate winners and negative
  incremental ablations are preserved.
- Added the connected frontend capstone selector and dynamic content across all
  five research zones. The local fallback remains explicitly labeled as a
  static Code RAG demonstration.
- Added `supabase/014-zero-api-capstone-portfolio.sql`; production Supabase was
  not modified.
- Runtime paid API cost remains `$0`; experiment and verifier containers have
  no network access.

## Verification
- Backend Ruff passed. Memory/API tests passed `16 passed, 1 skipped`.
- Fresh and repeat 013+014 migration plus PostgreSQL/RLS tests passed
  `17 passed` in an isolated disposable Postgres 16 container.
- Restricted Docker verification passed all 17 templates and 52 variants
  against host-side deterministic hashes.
- Immutable runner image ID:
  `sha256:961fec4e12716c0f3a6bd1d7aa250f75a524e70b7a0344ae62cb4a7bcff63b81`.
- A live BKT HTTP smoke completed experiment, analysis, independent
  reproduction, publication, and withdrawal at run version 11.
- Frontend lint and production build passed. The existing large-chunk warning
  remains unchanged.
- Connected Playwright checks passed on 1440 px desktop and 390 px mobile with
  no browser errors, failed responses, broken images, or horizontal overflow.
- Screenshots: `output/research-game-zero-api/`.
- Persistent preview: web `0.0.0.0:5181`, API `127.0.0.1:8790`.

---

# Task Plan: Playable Research World

## Goal
Turn Research Island into a player-controlled 2.5D social research game. Merge
the former observatory and personal lab into one research lab, then add a
curriculum-grounded Research Academy for learners without systematic research
training.

## Phases
- [x] P0: Freeze the six-scene world and interaction contract.
- [x] P0: Add Phaser, avatar selection, movement, world-bound collision, camera, portals,
  and scene persistence.
- [x] P0: Add the Research Academy mentor over all 503 curriculum lessons.
- [x] P0: Embed the existing ResearchRun workspaces into in-world objects.
- [x] P0: Add local/cloud game profiles and Realtime presence/broadcast.
- [x] P0: Verify desktop/mobile behavior and restart the persistent preview.

## Boundaries
- ResearchRun remains the only source of truth for experiment state and metrics.
- NPC tool actions remain human-gated and server-validated.
- P0 social movement is noncompetitive and client-authoritative.
- Production Supabase is not modified in this pass.

## Current Result

- Six scenes are live: lobby, academy, merged research lab, compute, paper, and plaza.
- The academy mentor exposes four structured research-training paths and retrieves
  real lesson content from the 503-lesson catalog before optional Tutor calls.
- The merged research lab contains separate project/data and hypothesis objects,
  but both operate on the same authoritative ResearchRun.
- Six role presets persist per local/cloud user. Presence and position/emote
  broadcast activate only when Supabase and migration 015 are configured.
- Phaser is dynamically imported and destination art is loaded only on scene entry.

## Verification

- Course audit: 20 phases, 503 lessons, 503 Chinese lessons; all 24 curated
  research-training lesson IDs exist.
- `npm run build`: passed. Phaser is emitted as a separate lazy runtime chunk;
  existing large-chunk warnings remain.
- `npm run lint`: passed with no warnings.
- Playwright playable-world flow: passed on 1440x900 and 390x844, including
  avatar selection, click movement, portal entry, lobby guide, academy mentor,
  503 lessons, merged-lab terminals, connected ResearchRun creation, nonblank
  canvas, image integrity, and horizontal overflow checks.
- Full HTTP ResearchRun smoke: `completed`, `supported`, 8 artifacts,
  1,000/1,000 same-direction Bootstrap samples, publish and withdrawal passed,
  final run version 11.
- Screenshots: `output/research-game-world/`.
- Persistent preview: web `0.0.0.0:5181`, Research API `127.0.0.1:8790`.

---

# Task Plan: Playable World P0.5 Motion and Scene Clarity

## Goal

Resolve four observed usability gaps: identical role art, static environments,
static walking, and portals that do not visibly enter a destination scene.

## Phases

- [x] P0: Generate six distinct low-poly role cutouts and bind each preset to
  its own asset in setup, local play, and multiplayer rendering.
- [x] P0: Add scene-aware ambient animation, including shoreline wave motion,
  leaf movement, light pulses, and restrained interior particles.
- [x] P0: Add a readable walking cycle using step cadence, body rise/fall,
  weight shift, shadow compression, and footstep feedback.
- [x] P0: Auto-enter portals at the inner threshold, show a scene-arrival
  transition, and verify that each destination route/background changes.
- [x] P0: Run lint/build and desktop/mobile Playwright motion and scene checks,
  then restart the persistent preview service.

## Boundaries

- Keep ResearchRun and NPC contracts unchanged.
- Do not replace the selected world or interior backgrounds.
- Ambient motion must stay subtle and respect reduced-motion preferences.
- Production Supabase remains unchanged.

## Current Result

- Six role presets now use six distinct transparent low-poly character assets.
- The lobby has animated shoreline foam and tree leaves; interior scenes have
  breathing NPCs, indicator lights, and restrained ambient particles.
- Local and remote characters animate while moving and settle into a subtle
  idle breath when stopped.
- Crossing a portal threshold now loads the destination scene, updates the URL,
  shows an arrival band, persists the destination, and survives refresh.

## Verification

- `npm run build`: passed; the existing large-chunk warning remains.
- `npm run lint`: passed with no warnings.
- `git diff --check`: passed.
- Desktop and mobile Playwright checks passed, including distinct role assets,
  actual idle-frame changes, walking state, automatic academy entry, destination
  canvas replacement, route persistence after refresh, and viewport integrity.
- Screenshots: `output/research-game-world/`.

---

# Task Plan: Research Game 3D Vertical Slice

## Goal

Replace the lobby and academy presentation with a playable Three.js graybox
whose default camera follows the player and whose world-map mode shows the full
scene, while retaining the Phaser world as an explicit fallback.

## Phases

- [x] P0: Audit the current world, installed Three.js game skills, controller
  options, and dependency compatibility.
- [x] P0: Freeze the core loop, scene plan, camera modes, architecture, fallback,
  performance budget, and acceptance checks.
- [x] P0: Implement the 3D Commons, Academy, local player, collision, follow
  camera, portal routing, and world map.
- [x] P0: Integrate existing profile, presence, guide, and academy mentor
  contracts without changing Supabase or backend schemas.
- [x] P0: Add responsive desktop/mobile HUD, input, loading, and reduced-motion
  behavior.
- [x] P0: Run lint/build plus desktop/mobile Playwright, nonblank canvas,
  movement, camera, portal, fallback, diagnostics, and performance checks.
- [x] P0: Restart the persistent preview only after the verification gate passes.

## Boundaries

- `src/game/` remains the working 2D fallback and `renderer=2d` forces it.
- ResearchRun, Tutor, and migration `015` remain unchanged.
- This is a graybox vertical slice, not a claim that all six scenes or final art
  have been migrated to 3D.

## Design Contract

See `docs/research-game-3d-vertical-slice-plan.md` for the player loop, camera
state machine, scene layout, module boundaries, budgets, and acceptance checks.

## Current Result

- The Commons and Academy now use the default 3D renderer. The four remaining
  destinations continue to use the existing Phaser implementation.
- The player has camera-relative movement, a collision-aware follow camera,
  orbit and zoom controls, and a paused orthographic world-map view.
- The procedural island includes dynamic water and trees, five destinations,
  six distinct player roles, guide and mentor interactions, desktop controls,
  and mobile controls.
- Existing profile and presence data pass through a reversible coordinate
  adapter. ResearchRun, Tutor, Supabase schemas, and migration `015` were not
  changed for this slice.
- `?renderer=2d` keeps the original Phaser world available as an explicit
  fallback.

## Verification

- `npm run build`, `npm run lint`, `git diff --check`, and
  `npm audit --omit=dev` pass. The production dependency audit reports zero
  vulnerabilities.
- `scripts/check-research-game-3d.py` passes at 1440x900 and 390x844 against
  the persistent preview. It checks canvas pixels, movement, follow camera,
  world map, interactions, portal travel, Academy content, mobile input, and
  the 2D fallback.
- Final screenshots are in `output/research-game-3d-preview/`.
- `ai-eng-studio-preview.service` serves `0.0.0.0:5181`; the unchanged research
  API remains healthy on `127.0.0.1:8790`.

---

# Task Plan: Research Game 3D Interaction Expansion

## Goal

Replace keyboard-first navigation with point-and-click movement, ground the
player model visibly, introduce an in-world game dialogue flow, improve the
Commons material language, and migrate the four remaining destinations to 3D.

## Phases

- [x] P0: Audit the current controller, collision, NPC panel, scene contracts,
  and retained 2D fallback.
- [x] P0: Add click navigation with grid A* routing, destination feedback,
  interaction approach behavior, and right-drag camera orbit.
- [x] P0: Correct the controller float height and avatar foot offset, then add
  diagnostics that expose the grounded state.
- [x] P0: Add a compact game-style NPC dialogue stage before opening each
  Agent's detailed research interface.
- [x] P0: Upgrade the Commons materials, lighting, architecture, paths, and
  environmental details without replacing the full-bleed world structure.
- [x] P0: Implement distinct 3D worlds and clickable workstations for Research,
  Compute, Paper, and Plaza.
- [x] P0: Extend desktop/mobile Playwright checks for click movement, grounded
  feet, NPC dialogue, all scene routes, workspace interactions, fallback, and
  nonblank canvases.
- [x] P0: Run lint/build/audit, restart the persistent preview, and verify the
  Mac tunnel-facing endpoint.

## Boundaries

- Existing ResearchRun, Tutor, profile, presence, and Supabase schemas remain
  unchanged.
- `renderer=2d` remains an explicit Phaser fallback for all six scenes.
- Procedural geometry remains the asset source for this pass; final authored
  character animation, photogrammetry, and audio are separate production work.

## Acceptance

- Desktop movement requires no movement keys: a click produces a visible target
  and the player follows a collision-aware path.
- Clicking an interaction walks into range before opening it; clicks while
  orbiting the camera never issue movement commands.
- The avatar's feet visually meet the floor and runtime diagnostics report an
  on-ground controller state.
- Every route under `/research-game/:sceneId` defaults to a distinct nonblank 3D
  scene with its existing Agent and workspace behavior available.

## Current Result

- Left-clicking the ground, an NPC, a workstation, or a portal now creates an
  A* route and approaches the target automatically. Right/middle drag orbits the
  camera, the wheel zooms, and movement keys no longer move the player.
- Ecctrl grounding and avatar visual offsets now place the feet on the floor;
  runtime diagnostics report `grounded=true` and a near-zero `footY`.
- NPC interaction first opens an in-world dialogue overlay, then continues into
  the existing detailed Agent interface.
- Campus, Academy, Research, Compute, Paper, and Plaza now have distinct 3D
  worlds. Research, Compute, Paper, and Plaza expose their existing work through
  clickable 3D workstations.
- The explicit `renderer=2d` fallback remains available. ResearchRun, Tutor,
  profile, presence, and Supabase data contracts were not changed.

## Verification

- `npm run build`: passed.
- `npm run lint`: passed.
- `npm audit --omit=dev`: zero vulnerabilities.
- Desktop and mobile Playwright checks passed for click navigation, camera
  follow/orbit, grounding, NPC dialogue, all six scenes, all four workspaces,
  map overlays, fallback rendering, and nonblank canvas pixels.
- Final screenshots: `output/research-game-3d-expansion-final/`.

---

# Task Plan: Persistent NPC Dialogue And Larger Commons

## Goal

Keep every NPC interaction inside one game dialogue surface and replace the
small decorative Commons with a larger, decision-shaping exploration map.

## Phases

- [x] P0: Inspect current dialogue, Agent actions, navigation, map framing,
  rendering budget, and browser checks.
- [x] P0: Research current public games built with Claude Code/Codex and extract
  patterns relevant to this project.
- [x] P0: Unify greeting, Agent chat, curriculum recommendations, and structured
  actions inside the game dialogue overlay.
- [x] P0: Expand the Commons topology, navigation grid, paths, landmarks,
  destination silhouettes, vegetation, and map camera.
- [x] P0: Add locally persisted field-note discoveries and compact objective/map
  feedback without changing backend contracts.
- [x] P0: Extend desktop/mobile Playwright checks and collect active-play,
  dialogue, discovery, and world-map screenshots plus renderer diagnostics.
- [x] P0: Run build, lint, audit, production-preview QA, then restart and verify
  the persistent `5181` preview service.

## Boundaries

- Existing ResearchRun, Tutor, profile, presence, Supabase, and 2D fallback
  contracts remain unchanged.
- This pass does not claim final premium character/building assets or audio.
- Detailed design and evidence live in
  `docs/research-game-world-expansion-plan.md`.

## Current Result

- NPC greetings, free-form Tutor chat, guide travel choices, 503-course mentor
  recommendations, and workspace actions now remain in one game dialogue box.
- Campus bounds grew from `42 x 28` to `96 x 68`, with an authored road network,
  five destinations, six optional field-note landmarks, and local discovery
  persistence per user.
- Full-detail follow view and a simplified world-map representation share the
  same authored coordinates. Merged road geometry and map-only silhouettes
  reduce the campus overview from 454 to 56 draw calls in SwiftShader.
- ResearchRun, Tutor, profile, presence, Supabase, and explicit 2D fallback
  contracts remain unchanged.

## Verification

- `npm run build`, `npm run lint`, `npm audit --omit=dev`, Python script syntax,
  and `git diff --check` pass; the production dependency audit reports zero
  vulnerabilities.
- `scripts/check-research-game-3d.py` passes at `1440 x 900` and `390 x 844`
  against a production preview. It covers click/tap movement, long navigation,
  grounding, camera orbit/follow, all six scenes, all four workspaces, guide and
  mentor dialogue, field-note persistence, map labels/budget, 2D fallback,
  canvas pixels, console errors, and viewport overflow.
- Final screenshots are in `output/research-game-world-expansion-final/`.

---

# Task Plan: Ground Flicker Regression

## Goal

Remove visible ground flicker in the 3D Commons without flattening the scene or
changing navigation, collision, ResearchRun, Tutor, Supabase, or fallback
contracts.

## Phases

- [x] P0: Reproduce with a fixed camera and capture consecutive-frame pixel
  differences, console errors, and renderer diagnostics.
- [x] P0: Isolate coplanar path/plaza depth conflicts from animated shadow or
  material effects using reduced-motion and controlled scene comparisons.
- [x] P0: Fix the owning scene/material layer and add a focused browser
  regression check.
- [x] P0: Run build, lint, visual desktop/mobile checks, then restart and smoke
  test the persistent `5181` preview.

## Debug Reference Ledger

- `threejs-debug-profiler/references/debug-profile-checklists.md`: loaded; used
  for reproduction order, renderer checks, root-cause ownership, and retest.
- `threejs-debug-profiler/references/checklists/scene-debugging.md`: loaded; used
  for material depth, camera, loop, console, screenshot, and pixel checks.

## Result

- Separated island ground, road edge, road surface, plaza border, and plaza
  surface using explicit Y layers, render order, and polygon offsets.
- Replaced path-local UVs with world-space UVs so overlapping road ribbons use
  identical texture samples at junctions.
- Added a reduced-motion, fixed-camera consecutive-frame regression check to
  `scripts/check-research-game-3d.py`.
- Controlled baseline measured mean absolute difference `3.623` and `6.27%`
  changed pixels in the plaza ROI. Production-preview verification measured
  `0.067` and `0.34%`, respectively.

## Verification

- `npm run build`, `npm run lint`, `npm audit --omit=dev`, and
  `git diff --check` pass; the production dependency audit reports zero
  vulnerabilities.
- Full desktop/mobile Playwright QA passes against the production preview,
  including all six scenes, map, navigation, dialogue, workspaces, 2D fallback,
  canvas pixels, and the new ground-stability threshold.
- Evidence: `output/ground-flicker-debug/` and `output/ground-flicker-final/`.

---

# Task Plan: Readable Research Game UI And DeepSeek NPCs

## Goal

Make the Research Game interface readable on desktop and mobile, enlarge NPC
portraits beyond the dialogue box where useful, and route every Research Game
NPC through the existing DeepSeek tutor service with a server-owned persona and
bounded conversational response length.

## Phases

- [x] P0: Audit the 3D and 2D dialogue call sites, game typography, tutor
  contract, live DeepSeek service, and persistent preview configuration.
- [x] P0: Add a validated NPC identifier, six independent server-side personas,
  NPC-specific token budgets, and focused backend tests without changing the
  existing Cici tutor behavior.
- [x] P0: Pass the NPC identifier from every 3D and 2D NPC interaction, enlarge
  Research Game typography, and give dialogue portraits a larger stable frame
  that may extend above the dialogue box.
- [x] P0: Add the local tutor proxy/configuration, run backend tests, build,
  lint, diff checks, and one real DeepSeek smoke request.
- [x] P0: Restart the persistent preview and verify dialogue readability,
  portrait framing, streaming, canvas pixels, console state, and viewport fit
  at desktop and mobile sizes.

## Boundaries

- Typography changes are scoped to the Research Game surfaces rather than the
  course site's unrelated learning pages.
- NPC identity is selected from a server-side allowlist. Browser-provided text
  cannot define or replace system personas.
- The service remains stateless: each NPC conversation sends only its own local
  history, preserving user and conversation isolation.
- ResearchRun, Supabase, Spark account handling, and the normal Cici AI tutor
  contract remain unchanged when no NPC identifier is supplied.

## UI Reference Ledger

- `threejs-game-ui-designer/references/ui-patterns.md`: loaded; used for dialogue
  hierarchy, world-label readability, stable control sizing, and overlays.
- `threejs-game-ui-designer/references/checklists/game-ui-quality.md`: loaded;
  used for in-world/UI cohesion, state coverage, and screenshot verification.
- `threejs-game-ui-designer/references/checklists/hud-readability.md`: loaded;
  used for HUD font floors, contrast, and information priority.
- `threejs-game-ui-designer/references/checklists/responsive-ui-fit.md`: loaded;
  used for desktop/mobile text fit, portrait framing, and overflow checks.
- `threejs-game-ui-designer/references/checklists/mobile-input.md`: loaded; used
  to preserve touch targets and dialogue controls while increasing text size.

## Result

- All six Research Game NPCs now select a validated server-side DeepSeek
  persona. Each persona has an independent name, role, focus, voice, and
  `180-280` token response cap; browser context cannot replace those identities.
- NPC requests disable DeepSeek thinking and ask for 1-4 conversational
  sentences. The normal Cici tutor path keeps its existing identity, Spark
  handling, thinking configuration, and optional global token limit.
- Every 3D and 2D NPC call now sends its own `assistant_id`; keyed dialogue
  components prevent one NPC's local history from carrying into another.
- Research Game metadata, secondary text, and normal body copy now have 12, 14,
  and 16 px floors. Dialogue body copy is 14-17 px, with larger HUD headings,
  world labels, controls, role art, and workspace copy.
- Desktop dialogue portraits render at roughly 350-410 px visual height. Mobile
  dialogue portraits render at roughly 230-290 px and may overflow their grid
  column without covering the separate text column.
- The persistent `5181` preview uses `/tutor-api` as a same-origin proxy to the
  host Tutor service on `8787`.

## Verification

- `npm run build`, `npm run lint`, Python compilation, `git diff --check`, and
  six focused backend persona tests pass.
- A real DeepSeek `campus-guide` request returned a 67-character answer as 林澈,
  did not identify as 茜茜, and stayed below the 300-character acceptance limit.
- Browser request interception confirmed the regular NPC and curriculum mentor
  branches send `campus-guide` and `research-mentor`, respectively, and render
  SSE replies inside the shared game dialogue box.
- Desktop QA covered all six 3D scenes, canvas pixels, grounding, maps, dialogue,
  four workspaces, and the 2D fallback. Focused mobile QA passes at `390 x 844`
  for canvas pixels, tap movement, map layout, dialogue readability, portrait
  size, and viewport overflow.
- Screenshots are in `output/deepseek-npc-readable-ui-final/` and
  `output/deepseek-npc-readable-ui-mobile-final/`.
