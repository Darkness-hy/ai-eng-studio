import { useCallback, useEffect, useState, type CSSProperties } from 'react';
import { useAuth } from '../lib/auth';
import { fetchIndex } from '../lib/data';
import {
  fetchLeaderboardSummary,
  formatLeaderboardScore,
  leaderboardSubtitle,
  leaderboardTitle,
  percentileLabel,
  statusLabel,
  type LeaderboardBoard,
  type LeaderboardEntry,
  type LeaderboardSummary,
} from '../lib/leaderboards';
import { useLang } from '../lib/i18n';

interface PageState {
  loading: boolean;
  summary: LeaderboardSummary | null;
  error: string | null;
}

const MEDALS: Record<1 | 2 | 3, { zh: string; en: string; cls: string }> = {
  1: {
    zh: '金',
    en: 'Gold',
    cls: 'border-ink-yellow/30 bg-pale-yellow text-ink-yellow',
  },
  2: {
    zh: '银',
    en: 'Silver',
    cls: 'border-hairline bg-bone text-faint',
  },
  3: {
    zh: '铜',
    en: 'Bronze',
    cls: 'border-[#d8b79a] bg-[#f6eee7] text-[#8a5a2b]',
  },
};

export function LeaderboardsPage() {
  const { enabled, loading: authLoading, profile } = useAuth();
  const { lang } = useLang();
  const zh = lang === 'zh';
  const [state, setState] = useState<PageState>({ loading: true, summary: null, error: null });

  const load = useCallback(async (canCommit: () => boolean = () => true) => {
    try {
      const index = await fetchIndex();
      const lessonIds = index.phases.flatMap((phase) =>
        phase.lessons.map((lesson) => `${phase.slug}/${lesson.slug}`),
      );
      const summary = await fetchLeaderboardSummary(lessonIds);
      if (canCommit()) {
        setState({ loading: false, summary: { ...summary, lessonTotal: lessonIds.length }, error: null });
      }
    } catch (err) {
      if (canCommit()) {
        setState({
          loading: false,
          summary: null,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }
  }, []);

  useEffect(() => {
    if (!enabled || !profile) return;
    let live = true;
    const timer = window.setTimeout(() => {
      void load(() => live);
    }, 0);
    return () => {
      live = false;
      window.clearTimeout(timer);
    };
  }, [enabled, profile, load]);

  if (!enabled) {
    return <Notice text={zh ? '云同步未配置，排行榜不可用' : 'Cloud sync is not configured.'} />;
  }
  if (authLoading || !profile) {
    return <Notice text={zh ? '加载中…' : 'Loading…'} />;
  }
  if (state.loading) {
    return <Notice text={zh ? '排行榜加载中…' : 'Loading leaderboards…'} />;
  }
  if (state.error || !state.summary) {
    return (
      <div className="mx-auto max-w-4xl px-5 py-14">
        <header className="border-b border-hairline pb-8">
          <h1 className="font-serif text-[38px] font-semibold tracking-tight">
            {zh ? '排行榜' : 'Leaderboards'}
          </h1>
        </header>
        <div className="mt-8 rounded-lg border border-ink-red/20 bg-pale-red px-5 py-4 text-[14px] text-ink-red">
          {zh ? '加载失败：' : 'Failed to load: '}
          {state.error}
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-6xl px-5 py-14">
      <header className="flex flex-wrap items-end justify-between gap-6 border-b border-hairline pb-8">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-faint">
            {zh ? '公开前三名 · 个人区间' : 'Public podium · private band'}
          </p>
          <h1 className="mt-2 font-serif text-[38px] font-semibold tracking-tight">
            {zh ? '排行榜' : 'Leaderboards'}
          </h1>
        </div>
        <button
          type="button"
          onClick={() => {
            setState((prev) => ({ ...prev, loading: true, error: null }));
            void load();
          }}
          className="rounded-md border border-hairline bg-paper px-3 py-1.5 font-mono text-[11.5px] text-faint transition-colors hover:bg-bone hover:text-ink"
        >
          {zh ? '刷新' : 'Refresh'}
        </button>
      </header>

      <section className="mt-8 grid gap-4 lg:grid-cols-3">
        {state.summary.boards.map((board, i) => (
          <LeaderboardCard
            key={board.id}
            board={board}
            lessonTotal={state.summary?.lessonTotal ?? 0}
            lang={lang}
            stagger={i}
          />
        ))}
      </section>
    </div>
  );
}

function LeaderboardCard({
  board,
  lessonTotal,
  lang,
  stagger,
}: {
  board: LeaderboardBoard;
  lessonTotal: number;
  lang: 'zh' | 'en';
  stagger: number;
}) {
  const zh = lang === 'zh';
  return (
    <article
      className="rise rounded-lg border border-hairline bg-paper p-5 shadow-lift"
      style={{ '--stagger': stagger } as CSSProperties}
    >
      <header className="border-b border-hairline pb-4">
        <div className="font-mono text-[10.5px] uppercase tracking-[0.16em] text-faint">
          {leaderboardSubtitle(board.id, lang)}
        </div>
        <h2 className="mt-2 font-serif text-[25px] font-semibold leading-tight">
          {leaderboardTitle(board.id, lang)}
        </h2>
      </header>

      <div className="mt-4 space-y-2">
        {board.top.length > 0 ? (
          board.top.map((entry) => (
            <PodiumRow key={`${board.id}-${entry.position}`} entry={entry} board={board} lessonTotal={lessonTotal} lang={lang} />
          ))
        ) : (
          <div className="rounded-md border border-dashed border-hairline bg-bone/40 px-4 py-6 text-center text-[13px] text-faint">
            {zh ? '榜单数据积累中' : 'Collecting leaderboard data'}
          </div>
        )}
      </div>

      <section className="mt-5 rounded-lg border border-hairline bg-bone/45 p-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-faint">
              {zh ? '我的位置' : 'My standing'}
            </div>
            <div className="mt-1 font-serif text-[25px] font-semibold leading-none">
              {percentileLabel(board.mine, lang)}
            </div>
          </div>
          <div className="shrink-0 text-right">
            <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-faint">
              {zh ? '成绩' : 'Score'}
            </div>
            <div className="mt-1 font-mono text-[13px] text-ink">
              {formatLeaderboardScore(board.id, board.mine.score, lessonTotal, lang)}
            </div>
          </div>
        </div>
        <p className="mt-3 text-[13px] leading-relaxed text-faint">{statusLabel(board, lang)}</p>
      </section>
    </article>
  );
}

function PodiumRow({
  entry,
  board,
  lessonTotal,
  lang,
}: {
  entry: LeaderboardEntry;
  board: LeaderboardBoard;
  lessonTotal: number;
  lang: 'zh' | 'en';
}) {
  const medal = MEDALS[entry.position];
  return (
    <div className="grid grid-cols-[42px_1fr_auto] items-center gap-3 rounded-md border border-hairline bg-canvas px-3 py-3">
      <span
        className={`flex h-8 w-8 items-center justify-center rounded-full border font-mono text-[11px] font-medium ${medal.cls}`}
        title={lang === 'zh' ? medal.zh : medal.en}
      >
        {entry.position}
      </span>
      <span className="min-w-0">
        <span className="block truncate text-[14px] font-medium">{entry.displayName}</span>
        <span className="font-mono text-[10.5px] uppercase tracking-[0.14em] text-faint">
          {lang === 'zh' ? medal.zh : medal.en}
        </span>
      </span>
      <span className="font-mono text-[12px] text-faint">
        {formatLeaderboardScore(board.id, entry.score, lessonTotal, lang)}
      </span>
    </div>
  );
}

function Notice({ text }: { text: string }) {
  return <div className="py-32 text-center text-[14px] text-faint">{text}</div>;
}
