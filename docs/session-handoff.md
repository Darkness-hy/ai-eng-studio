# Session Handoff

Timestamp: 2026-07-02 04:14:11 CST

## Current Project

- Repository: `/home/nvme03/dhy/workspace/code/ai-eng-studio`
- Production site: `https://darkness-hy.github.io/ai-eng-studio/`
- Production entry bundle last confirmed: `index-uKu4h2CX.js`

## Latest Remote State

- `main`: `7c717ded034459f894572dbb8f328edd6d9f4e2d`
- `gh-pages`: `de75c2e564712c56a16f17c1397fe30e82ffdc9e`

## Completed Work

### DeepSeek Tutor / Caddy

- Public tutor endpoint is served through Caddy.
- `tutor.hongyuding.site` proxies to `127.0.0.1:18787`.
- `127.0.0.1:18787` is a reverse SSH tunnel from spark-32e2 to Spark-side tutor service on `127.0.0.1:8787`.
- Tutor smoke test returned correct SSE:

```text
data: {"type": "delta", "text": "OK"}
data: {"type": "done"}
```

### AI Tutor Frontend Config

- A broken GitHub Pages build without public env had temporarily disabled Supabase and the AI tutor.
- Rebuilt Pages with public runtime config restored:
  - Supabase URL present.
  - Supabase publishable key present in built bundle.
  - Tutor endpoint present: `https://tutor.hongyuding.site/chat`.
- AI tutor button and account features were restored.

### Spark Class Eligibility

- Fixed frontend Spark class detection so the tutor does not incorrectly say the user has not joined `Spark 使用班级`.
- Changed logic in `src/lib/sparkAccount.ts` to match Spark class by:
  - class id
  - invite code `FRB2XC`
  - normalized class name `spark使用班级`
- Changed `src/components/TutorWidget.tsx` so Spark account context refreshes when users ask Spark/account-related questions.

### Cross-Account Progress Leak

Bug:

- Same browser had old account `skyhyding`.
- New test account `xq` inherited:
  - learning progress `20/503`
  - placement result `10/50`

Root cause:

- Old code stored all local progress under shared browser keys:
  - `aes:progress:v1`
  - `aes:placement:v1`
- Login sync merged local browser state into the signed-in account and pushed it to Supabase.
- After the first cleanup, stale browser placement cache could re-upload to Supabase.

Final fix:

- `src/lib/progress.ts`
  - Added per-user localStorage scope.
  - Signed-in users use `aes:progress:v1:user:<userId>`.
- `src/lib/placement.ts`
  - Added per-user localStorage scope.
  - Signed-in users use `aes:placement:v1:user:<userId>`.
  - Login pull treats cloud as authoritative, including no-placement state.
- `src/lib/auth.tsx`
  - Switches local storage scope on login/logout.
- `src/lib/sync.ts`
  - Login sync now replaces scoped local cache with cloud rows.
  - Supabase admin reset to empty progress will clear browser cache instead of being undone by local data.

Verification:

- `npm run build` passed.
- `npm run lint` passed.
- Online bundle contains:
  - Supabase config
  - Tutor config
  - `:user:` scoped localStorage logic
  - no undefined Supabase initialization
- `xq` was finally confirmed in the UI as:
  - progress `0/503`
  - no placement result, showing the placement entry card instead.

## xq Cleanup Details

- `xq` user id:

```text
321bb96d-28ec-4818-8f56-f8b1e7878bf0
```

Cleanup SQL used:

```sql
delete from public.progress
where user_id = '321bb96d-28ec-4818-8f56-f8b1e7878bf0';

delete from public.activity
where user_id = '321bb96d-28ec-4818-8f56-f8b1e7878bf0';

delete from public.placement
where user_id = '321bb96d-28ec-4818-8f56-f8b1e7878bf0';
```

Browser-local cleanup that was attempted:

```js
localStorage.removeItem('aes:placement:v1:user:321bb96d-28ec-4818-8f56-f8b1e7878bf0');
localStorage.removeItem('aes:placement:v1');
location.reload();
```

Final deployed code should make this local cleanup unnecessary for future Supabase resets.

## Important Warnings

- A GitHub token was pasted into chat during this session. It should be revoked/rotated.
- Local worktree may still be dirty and should not be treated as canonical.
- Remote `origin/main` and `origin/gh-pages` are the source of truth for deployed frontend fixes.

## Known Local Worktree State

The local worktree may still include unrelated or pre-existing changes:

- Modified `server/tutor/*` files from earlier DeepSeek tutor work.
- Untracked `.cc-connect/`.
- Untracked `AGENTS.md`.
- Untracked `aes-handoff.zip`.
- Untracked `assets/`.
- Untracked `docs/spark-32e2-deepseek-tutor-fix.md`.
- Untracked `server/tutor/.venv/`.
- Local branch may be divergent from `origin/main`; avoid rebasing/resetting unless explicitly requested.

## If Continuing In A New Session

Recommended first instruction:

```text
请先阅读 docs/session-handoff.md，然后继续当前 ai-eng-studio 线上修复工作。
```

Suggested checks:

```bash
git fetch origin main gh-pages
git ls-remote origin refs/heads/main refs/heads/gh-pages
curl -fsSL --noproxy '*' 'https://darkness-hy.github.io/ai-eng-studio/?v=handoff-check' | sed -n 's/.*src="\/ai-eng-studio\/assets\/\([^"]*\.js\)".*/\1/p'
```
