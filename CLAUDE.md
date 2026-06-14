# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`ai-eng-studio` is a Chinese-first, interactive learning frontend for the open-source
[`ai-engineering-from-scratch`](https://github.com/rohitg00/ai-engineering-from-scratch)
curriculum (503 lessons, 20 phases). Pure static SPA (Vite + React 19 + TS +
Tailwind 4), deployed to GitHub Pages at `https://darkness-hy.github.io/ai-eng-studio/`.
Optional Supabase backend adds accounts, cross-device sync, and an admin dashboard.

## Commands

```bash
npm run build:content   # MUST run first: regenerates public/data/ from the upstream repo
npm run dev             # dev server on http://localhost:5180
npm run build           # tsc -b && vite build (+ SPA 404.html + .nojekyll for Pages)
npm run deploy:build    # build with base path /ai-eng-studio/ for GitHub Pages
npm run lint            # eslint (strict — see pitfalls below)
```

No test runner is configured. Verification is done by `tsc -b`, `npm run lint`, and Playwright
screenshot/E2E scripts run ad hoc (Python `playwright`, headless chromium).

**Deploy** (no CI): push `main`, then `npm run deploy:build` and force-push `dist/` to the
`gh-pages` branch. `vite.config.ts` reads `DEPLOY_BASE`; `App.tsx` uses `import.meta.env.BASE_URL`
for the router basename and all asset URLs — never hardcode `/ai-eng-studio/`.

## Local environment gotchas

- A local proxy (clash on `127.0.0.1:7890`) intercepts localhost. Use `curl --noproxy '*'` and
  launch Playwright chromium with `args=["--no-proxy-server"]`. BUT keep the proxy when reaching
  external CDNs (Pyodide, Supabase) — drop `--no-proxy-server` for those tests.
- `.env.local` holds `VITE_SUPABASE_URL` / `VITE_SUPABASE_ANON_KEY` (gitignored). Without them the
  app runs fully local — cloud features go dormant, no auth gate.

## Architecture

**Content pipeline** (`scripts/build-content.mjs`) is the spine. It scans the sibling
`../ai-engineering-from-scratch/phases/` plus the local `content/zh/` translation overlay and emits
static JSON into `public/data/`:
- `index.json` — catalog: phases (with `deps` DAG, `hours` from ROADMAP.md), per-lesson metadata
- `lessons/<phase>/<lesson>.json` — full payload: en + zh body, quizzes, code files
- `glossary.json` — terms with zh overlay

Two hardcoded lists live in this script and matter:
- `PHASE_META` — zh titles/descriptions and the phase dependency DAG (mirrors the upstream README flowchart).
- `EXCLUDED_LESSONS` — capstones dropped for NVIDIA DGX Spark (aarch64) incompatibility. The audit
  rationale is in `../plan/task_plan.md`.
- `pyodideRunnable()` whitelists stdlib+numpy imports to mark which Python code can run in-browser.

**Translations** live in `content/zh/` (never edit the upstream repo). One `.md` + optional
`.quiz.json` per lesson, plus `titles.json` per phase and `glossary.json`. Untranslated lessons fall
back to English with a "translation in progress" notice. Large translation batches are run as
Workflow fan-outs (one agent per lesson). After translating, re-run `build:content`.

**Frontend** (`src/`):
- `pages/` — route components; `components/` — shared UI; `lib/` — state and data.
- `lib/data.ts` fetches the static JSON (cached). `lib/i18n.tsx` holds the bilingual UI dictionary
  and `lang` context; lesson/phase content uses zh with en fallback.
- `lib/progress.ts` — local-first progress store (localStorage, `useSyncExternalStore`). Quiz scores
  are **first-attempt-only**: `saveQuizScore` no-ops if a score already exists; retakes are practice.
- `lib/sync.ts` + `lib/supabase.ts` + `lib/auth.tsx` — cloud layer. On login: pull remote, merge
  (newest `updatedAt` wins per lesson, visits unioned), push local; thereafter 2s-debounced upsert.
  `App.tsx`'s `Gate` forces unauthenticated users to `/login` when cloud is enabled.
- `lib/placement.ts` — the "find your level" quiz (based on the upstream `/find-your-level` skill,
  scaled to 50 hard Q / 10 per area, score 0-50 → entry-phase, 20-phase path table). Questions are
  authored+verified via workflow; `QUESTIONS_PER_AREA` and `entryPhase` boundaries drive scaling.
  Result syncs via the `placement` table on the same login flow.
- Heavy deps are lazy-loaded: shiki (fine-grained core, `lib/highlight.ts`), mermaid
  (`components/Mermaid.tsx`), Pyodide from CDN (`lib/pyodide.ts`), `AdminPage`.
- Admin dashboard charts (`components/charts/`) are dependency-free SVG.

**Supabase** schema is `supabase/schema.sql` (+ `002-placement.sql`), applied by hand in the
Dashboard SQL Editor. RLS: students read/write only their own rows; `is_admin()` grants admins
read-all. A signup trigger auto-creates the profile and bootstraps admin role for a hardcoded email
allowlist in `handle_new_user()` — adding an admin means editing that list AND running an `update`.

## Pitfalls

- **ESLint is strict on React rules.** `react-hooks/set-state-in-effect` forbids synchronous
  `setState` in an effect body — derive render-local values or set state only in async `.then`
  callbacks. `react-hooks/refs` forbids reading `ref.current` during render — use `useState` for
  values needed during render. Hooks must run before any early `return` (compute quiz arrays etc.
  unconditionally).
- **Design system** (minimalist editorial): theme tokens only — `bg-canvas/bone/paper`,
  `text-ink/faint`, `border-hairline`, pastel pairs `bg-pale-{red,blue,green,yellow}` +
  `text-ink-{...}`. Fonts `font-serif` (Newsreader/Noto Serif SC), `font-mono` (JetBrains Mono),
  `font-brand` (DingTalk Sans, footer only). No heavy shadows, no gradients, no emojis, 8px radii.
- **Markdown heading IDs** in `components/Markdown.tsx` must stay in sync with `extractToc` in
  `lib/md.ts` (same `slugify` + duplicate-suffix logic) or the TOC scroll-spy breaks.
- Progress counts iterate the catalog, not raw localStorage keys, so stale lesson IDs (from removed
  capstones) don't inflate totals. Keep new aggregations index-based.
