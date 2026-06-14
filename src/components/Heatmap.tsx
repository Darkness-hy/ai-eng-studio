import type { Lang } from '../lib/types';

// Tailwind needs these class strings present literally to emit them.
const LEVELS = ['bg-bone', 'bg-pale-green', 'bg-ink-green/40', 'bg-ink-green'];

function level(count: number): number {
  if (count === 0) return 0;
  if (count <= 2) return 1;
  if (count <= 5) return 2;
  return 3;
}

function dayKey(d: Date): string {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

/** GitHub-style contribution calendar: the last ~month of lessons completed. */
export function Heatmap({ counts, lang }: { counts: Map<string, number>; lang: Lang }) {
  const today = new Date();
  const start = new Date(today);
  start.setDate(start.getDate() - (4 * 7 + today.getDay())); // Sunday-aligned, ~1 month (5 weeks)

  const cells: { key: string; count: number }[] = [];
  const cursor = new Date(start);
  while (cursor <= today) {
    const key = dayKey(cursor);
    cells.push({ key, count: counts.get(key) ?? 0 });
    cursor.setDate(cursor.getDate() + 1);
  }

  return (
    <div className="overflow-x-auto">
      <div className="grid grid-flow-col grid-rows-7 gap-[3px]" style={{ gridAutoColumns: '11px' }}>
        {cells.map(({ key, count }) => {
          const lv = level(count);
          return (
            <span
              key={key}
              title={`${key} · ${count} ${lang === 'zh' ? '课' : 'lessons'}`}
              className={`h-[11px] w-[11px] rounded-[2px] ${LEVELS[lv]} ${lv === 0 ? 'border border-hairline' : ''}`}
            />
          );
        })}
      </div>
      <div className="mt-3 flex items-center gap-1.5 font-mono text-[11px] text-faint">
        <span>{lang === 'zh' ? '少' : 'less'}</span>
        {LEVELS.map((cls, i) => (
          <span
            key={cls}
            className={`h-[10px] w-[10px] rounded-[2px] ${cls} ${i === 0 ? 'border border-hairline' : ''}`}
          />
        ))}
        <span>{lang === 'zh' ? '多' : 'more'}</span>
      </div>
    </div>
  );
}
