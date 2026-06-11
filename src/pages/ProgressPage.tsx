import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { fetchIndex } from '../lib/data';
import { phaseTitle, useLang } from '../lib/i18n';
import { exportProgress, importProgress, streakDays, useProgress } from '../lib/progress';
import type { CourseIndex } from '../lib/types';

export function ProgressPage() {
  const { lang, t } = useLang();
  const [index, setIndex] = useState<CourseIndex | null>(null);
  const progress = useProgress();
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchIndex().then(setIndex);
  }, []);

  if (!index) return <div className="py-32 text-center text-faint">{t('loading')}</div>;

  const doneTotal = Object.values(progress.lessons).filter((l) => l.done).length;
  const pct = Math.round((doneTotal / index.stats.lessons) * 100);

  const quizEntries = Object.values(progress.lessons).filter((l) => l.postTotal);
  const quizAvg =
    quizEntries.length > 0
      ? Math.round(
          (quizEntries.reduce((acc, l) => acc + (l.postScore ?? 0) / (l.postTotal ?? 1), 0) /
            quizEntries.length) *
            100,
        )
      : null;

  const download = () => {
    const blob = new Blob([exportProgress()], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ai-eng-studio-progress.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const upload = (file: File) => {
    file.text().then((text) => {
      if (!importProgress(text)) alert(lang === 'zh' ? '导入失败：文件格式不正确' : 'Import failed: bad format');
    });
  };

  return (
    <div className="mx-auto max-w-4xl px-5 py-14">
      <header className="flex flex-wrap items-end justify-between gap-6 border-b border-hairline pb-8">
        <div>
          <h1 className="font-serif text-[38px] font-semibold tracking-tight">{t('progress_title')}</h1>
          <div className="mt-4 flex items-baseline gap-8">
            <Big n={`${pct}%`} label={t('progress_overall')} />
            <Big n={`${doneTotal}/${index.stats.lessons}`} label={t('lessons')} />
            <Big n={`${streakDays()}`} label={`${t('streak')} (${t('days')})`} />
            {quizAvg != null && <Big n={`${quizAvg}`} label={t('quiz_avg')} />}
          </div>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={download}
            className="rounded-md border border-hairline bg-paper px-3 py-1.5 text-[13px] text-faint transition-colors hover:bg-bone hover:text-ink"
          >
            {t('export')}
          </button>
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            className="rounded-md border border-hairline bg-paper px-3 py-1.5 text-[13px] text-faint transition-colors hover:bg-bone hover:text-ink"
          >
            {t('import')}
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="application/json"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
          />
        </div>
      </header>

      <ol className="mt-2">
        {index.phases.map((phase) => {
          const done = phase.lessons.filter(
            (l) => progress.lessons[`${phase.slug}/${l.slug}`]?.done,
          ).length;
          const phasePct = phase.lessons.length ? (done / phase.lessons.length) * 100 : 0;
          return (
            <li key={phase.slug} className="border-b border-hairline">
              <Link
                to={`/phase/${phase.slug}`}
                className="group grid grid-cols-[44px_1fr_140px_64px] items-center gap-4 py-3.5"
              >
                <span className="font-mono text-[12px] text-faint">
                  {String(phase.num).padStart(2, '0')}
                </span>
                <span className="truncate text-[14.5px] font-medium group-hover:underline group-hover:decoration-hairline group-hover:underline-offset-4">
                  {phaseTitle(phase, lang)}
                </span>
                <span className="h-[3px] overflow-hidden rounded-full bg-bone">
                  <span
                    className="block h-full rounded-full bg-ink-green"
                    style={{ width: `${phasePct}%` }}
                  />
                </span>
                <span className="text-right font-mono text-[11.5px] text-faint">
                  {done}/{phase.lessons.length}
                </span>
              </Link>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function Big({ n, label }: { n: string; label: string }) {
  return (
    <div>
      <div className="font-serif text-[30px] font-semibold leading-none">{n}</div>
      <div className="mt-1 text-[12px] text-faint">{label}</div>
    </div>
  );
}
