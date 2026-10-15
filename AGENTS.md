# AGENTS.md

Guidance for coding agents working in this repository.

## Project

`ai-eng-studio` is a Chinese-first interactive learning frontend for the
`ai-engineering-from-scratch` curriculum. It is a Vite + React 19 + TypeScript +
Tailwind CSS 4 static SPA, deployed to GitHub Pages. Supabase is optional and
adds accounts, sync, and admin features.

## Commands

```bash
npm run build:content   # regenerate public/data/ from upstream + local zh overlay
npm run dev             # Vite dev server, normally http://localhost:5180
npm run build           # tsc -b && vite build, then SPA 404.html + .nojekyll
npm run deploy:build    # build with DEPLOY_BASE=/ai-eng-studio/
npm run lint            # strict ESLint
```

No test runner is configured. Verify frontend changes with `npm run build`,
`npm run lint`, and targeted Playwright checks when UI behavior matters.

## Architecture

- `scripts/build-content.mjs` is the content pipeline. It reads the sibling
  upstream repo at `../ai-engineering-from-scratch/phases/`, applies the local
  Chinese overlay in `content/zh/`, and emits static JSON into `public/data/`.
- `content/zh/` contains translations and quiz overlays. Do not edit the
  upstream curriculum for translations.
- `src/pages/` contains route components.
- `src/components/` contains shared UI.
- `src/lib/` contains data loading, i18n, progress, auth, sync, placement logic,
  markdown helpers, highlighting, and Pyodide loading.
- `supabase/` contains hand-applied SQL schema and migrations.
- `server/tutor/` and `server/spark-agent/` are separate server-side helpers.

## Local Gotchas

- Local proxy settings can affect localhost. Use `curl --noproxy '*'` for local
  dev-server checks. For Playwright local screenshots, Chromium may need
  `--no-proxy-server`; keep proxy behavior for external CDNs.
- `.env.local` is gitignored and may contain `VITE_SUPABASE_URL` and
  `VITE_SUPABASE_ANON_KEY`. Without these, the app runs local-only.
- `vite.config.ts` uses `DEPLOY_BASE`; route basenames and assets should use
  `import.meta.env.BASE_URL`. Do not hardcode `/ai-eng-studio/`.

## Implementation Rules

- Preserve the minimalist editorial design system: theme tokens such as
  `bg-canvas`, `bg-bone`, `bg-paper`, `text-ink`, `text-faint`,
  `border-hairline`, and pastel pairs. Avoid heavy shadows, gradients, emojis,
  and radii above 8px unless the existing component already requires it.
- React hooks linting is strict. Avoid synchronous `setState` in effect bodies,
  avoid reading `ref.current` during render, and call hooks before early returns.
- Keep markdown heading ID logic in `components/Markdown.tsx` aligned with
  `extractToc` in `src/lib/md.ts`.
- Progress aggregations should iterate the catalog, not raw localStorage keys,
  so removed lesson IDs do not inflate totals.
- Translation changes require `npm run build:content` before the generated
  `public/data/` output reflects them.

## Workflow

- Prefer focused changes that follow existing patterns.
- Do not revert unrelated user changes. The repo may contain generated files,
  local server artifacts, or untracked cc-connect assets.
- Before finishing code changes, report which verification commands ran and any
  commands that could not be run.
