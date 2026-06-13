import type { ReactElement } from 'react';
import type { Badge, BadgeCategory } from '../lib/achievements';
import type { Lang } from '../lib/types';

const CAT: Record<BadgeCategory, { bg: string; ring: string; ink: string }> = {
  streak: { bg: 'bg-pale-red', ring: 'border-ink-red/30', ink: 'text-ink-red' },
  lessons: { bg: 'bg-pale-blue', ring: 'border-ink-blue/30', ink: 'text-ink-blue' },
  phase: { bg: 'bg-pale-green', ring: 'border-ink-green/30', ink: 'text-ink-green' },
  placement: { bg: 'bg-pale-yellow', ring: 'border-ink-yellow/30', ink: 'text-ink-yellow' },
  quiz: { bg: 'bg-pale-blue', ring: 'border-ink-blue/30', ink: 'text-ink-blue' },
};

function BadgeIcon({ category, on }: { category: BadgeCategory; on: boolean }) {
  const c = CAT[category];
  const paths: Record<BadgeCategory, ReactElement> = {
    streak: <path d="M9 2c1.5 2.2.6 3.7-.3 5C7.6 8.8 7 10 7 11.5a5 5 0 1 0 9.6-2C15.4 6 13 4.4 12.5 2c-.9 2-2 2.6-3.5 0Z" />,
    lessons: <path d="M4 4.5A1.5 1.5 0 0 1 5.5 3H15v13H5.5A1.5 1.5 0 0 0 4 17.5v-13ZM15 3l4 1.5V18l-4-2" />,
    phase: <path d="M3 6h14M3 11h14M3 16h9" />,
    placement: <><circle cx="10" cy="10" r="7" /><circle cx="10" cy="10" r="3" /></>,
    quiz: <path d="M4 10.5 8 15l8-9" />,
  };
  return (
    <span
      className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full border ${
        on ? `${c.bg} ${c.ring}` : 'border-hairline bg-bone'
      }`}
    >
      <svg
        width="18"
        height="18"
        viewBox="0 0 20 20"
        fill="none"
        stroke={on ? 'currentColor' : '#b8b7b3'}
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
        className={on ? c.ink : ''}
      >
        {paths[category]}
      </svg>
    </span>
  );
}

export function BadgeWall({ badges, lang }: { badges: Badge[]; lang: Lang }) {
  const unlocked = badges.filter((b) => b.unlocked).length;
  return (
    <section>
      <div className="mb-4 flex items-baseline justify-between">
        <span className="font-mono text-[10.5px] uppercase tracking-[0.16em] text-faint">
          {lang === 'zh' ? '成就徽章' : 'Achievements'}
        </span>
        <span className="font-mono text-[11.5px] text-faint">
          {unlocked} / {badges.length}
        </span>
      </div>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 md:grid-cols-3">
        {badges.map((b) => (
          <div
            key={b.key}
            className={`flex items-center gap-3 rounded-lg border px-4 py-3 ${
              b.unlocked ? 'border-hairline bg-paper' : 'border-hairline bg-bone/40'
            }`}
          >
            <BadgeIcon category={b.category} on={b.unlocked} />
            <div className="min-w-0">
              <div className={`truncate text-[13.5px] font-medium ${b.unlocked ? 'text-ink' : 'text-faint'}`}>
                {lang === 'zh' ? b.zh : b.en}
              </div>
              <div className="mt-0.5 font-mono text-[10.5px] text-faint">
                {b.unlocked
                  ? lang === 'zh'
                    ? '已解锁'
                    : 'unlocked'
                  : `${Math.min(b.current, b.target)} / ${b.target}`}
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
