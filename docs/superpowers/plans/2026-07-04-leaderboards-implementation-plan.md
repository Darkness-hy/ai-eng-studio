# Leaderboards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build three privacy-bounded leaderboards for progress, placement, and post-quiz performance.

**Architecture:** Supabase owns cross-user aggregation through one `security definer` RPC. The React app sends the current catalog lesson ids to that RPC, calls it from a small typed API module, and renders a dedicated `/leaderboards` page; the frontend never reads full `profiles`, `progress`, or `placement` tables for students.

**Tech Stack:** Vite, React 19, TypeScript, Tailwind CSS 4, Supabase SQL/RPC.

---

## File Structure

- Create `supabase/012-leaderboards.sql`: idempotent SQL function `public.get_leaderboard_summary(lesson_ids text[])` returning one JSONB summary with three boards.
- Create `src/lib/leaderboards.ts`: typed Supabase RPC wrapper plus UI-facing board metadata and fallback messages.
- Create `src/pages/LeaderboardsPage.tsx`: dedicated page showing three medal boards and the current user's status.
- Modify `src/App.tsx`: add `/leaderboards` route.
- Modify `src/components/Layout.tsx`: add nav link.
- Modify `src/lib/i18n.tsx`: add `nav_leaderboards`.

## Decisions

- Public top three use `profiles.display_name`; empty names render as `未命名学习者`.
- Do not expose total user count, exact rank, or full ranking list.
- Progress board score: completed lessons in current catalog, with `lesson_ids` supplied by the frontend.
- Placement board score: current `placement.total`.
- Unit test board score: average first-attempt post-quiz percentage, eligible after at least 10 post quizzes.
- Percentile is returned as a coarse 10-point ceiling bucket (`10`, `20`, ..., `100`).
- Gap to podium is hidden for users already in top three.

## Tasks

### Task 1: SQL RPC

**Files:**
- Create: `supabase/012-leaderboards.sql`

- [x] Add an idempotent `public.get_leaderboard_summary(lesson_ids text[])` function.
- [x] In SQL, require `auth.uid()` to be non-null.
- [x] Build per-user scores for three boards from `profiles`, `progress`, and `placement`.
- [x] Return only: board id, top three display names/scores, current user's score/eligibility/percentile/gap.
- [x] Do not return student totals, exact rank, emails, or non-top-three names.

### Task 2: Typed Client API

**Files:**
- Create: `src/lib/leaderboards.ts`

- [x] Define `LeaderboardId`, `LeaderboardSummary`, `LeaderboardBoard`, `LeaderboardEntry`, and `LeaderboardMine`.
- [x] Add `fetchLeaderboardSummary(lessonIds: string[])`.
- [x] Throw a clear error when Supabase is not configured.
- [x] Add label/format helpers for board titles, score labels, and gap labels.

### Task 3: Leaderboards Page

**Files:**
- Create: `src/pages/LeaderboardsPage.tsx`

- [x] Fetch the course index to get the current lesson ids.
- [x] Fetch the leaderboard RPC after the index is available.
- [x] Render loading, auth-disabled, error, and empty states.
- [x] Render three board sections with gold/silver/bronze medals.
- [x] Render each board's "my status": percentile bucket, score, and gap or next action.

### Task 4: Route And Navigation

**Files:**
- Modify: `src/App.tsx`
- Modify: `src/components/Layout.tsx`
- Modify: `src/lib/i18n.tsx`

- [x] Add `/leaderboards` route.
- [x] Add `nav_leaderboards` translations.
- [x] Add a nav link that is visible when cloud auth is enabled and a user is signed in.
- [x] Keep the existing route basename behavior unchanged.

### Task 5: Verification

**Commands:**
- `npm run lint`
- `npm run build`

- [x] Run lint and fix any new issues.
- [x] Run build and fix any type or bundling issues.
- [x] Inspect git diff to confirm only leaderboard-related files changed.

## Notes

- There is no configured test runner in this repo. Baseline `npm run lint` and `npm run build` pass before this work.
- Do not modify or revert existing unrelated changes in `server/tutor/*`, `src/lib/auth.tsx`, `src/lib/progress.ts`, `src/lib/placement.ts`, or `src/lib/sync.ts`.
- Local UI/RPC runtime limits: this machine has no Playwright/Chromium and no `.env.local` Supabase config, so the authenticated RPC path was not browser-tested locally.

## Deployment Addendum

- [x] Cherry-pick the leaderboard commit into a clean worktree based on `origin/main`.
- [x] Re-run `npm run lint` and `npm run build` in the clean worktree.
- [x] Push the clean source commit to GitHub.
- [x] Build the GitHub Pages bundle with the existing public Supabase and tutor env values.
- [x] Replace the `gh-pages` branch contents with the generated `dist/` bundle.
- [x] Verify the local Pages bundle contains the leaderboard route/RPC and the public env values.
- [x] Verify the published bundle contains the leaderboard route/RPC and the public env values.

## Deployment Notes

- Clean deploy worktree initially had no `node_modules`, so `npm run lint` failed with `eslint: not found` and `npm run build` failed with `tsc: not found`.
- Resolution: ran `npm ci` in the clean deploy worktree before repeating verification.
- `git push origin HEAD:main` failed because the machine has no non-interactive GitHub credential for the HTTPS remote.
- Online GitHub Pages still serves `index-uKu4h2CX.js`, which does not contain `get_leaderboard_summary` or `/leaderboards`.
- Local Pages preview is running at `http://127.0.0.1:5182/ai-eng-studio/leaderboards` and returned `200 OK`.
- User supplied a token for this deployment run. It was used only through a temporary `GIT_ASKPASS` shell script, then `/tmp/git-askpass-ai-eng-studio.sh` and the token environment variable were removed.
- `main` pushed to `a020cf3676d84a639fcd33b0b595b80282f5f342`; `gh-pages` pushed to `785215864bbeea1db7393287c91f7da2826d9d35`.
- Published Pages HTML now references `index-HYJPXFon.js`; that bundle contains `get_leaderboard_summary`, `/leaderboards`, the production Supabase URL/key marker, and the tutor endpoint.
- Direct route `https://darkness-hy.github.io/ai-eng-studio/leaderboards` returns GitHub Pages HTTP `404` with the SPA fallback body, and that fallback references the new bundle.
- After launch, the live RPC returned `PGRST202`: Supabase could not find `public.get_leaderboard_summary(lesson_ids)` in the schema cache.
- Frontend error formatting was patched and deployed so this condition no longer renders as `[object Object]`; latest refs are `main=7af1a69b64ef9499bcf06a11dc406c297b7752b0`, `gh-pages=e7a8a47b01e6f8ef1d8575419a887ca61295dbde`.
- Remaining database action: execute `supabase/012-leaderboards.sql` in Supabase SQL Editor, then reload/retry the leaderboard page after schema cache refresh.
