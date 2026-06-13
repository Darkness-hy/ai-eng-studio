import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { fetchIndex } from '../lib/data';
import { lessonTitle, phaseTitle, useLang } from '../lib/i18n';
import { useProgress } from '../lib/progress';
import type { CourseIndex } from '../lib/types';

export function PhasePage() {
  const { phaseSlug } = useParams();
  const { lang, t } = useLang();
  const [index, setIndex] = useState<CourseIndex | null>(null);
  const progress = useProgress();

  useEffect(() => {
    fetchIndex().then(setIndex);
  }, []);

  if (!index) return <div className="py-32 text-center text-faint">{t('loading')}</div>;
  const phase = index.phases.find((p) => p.slug === phaseSlug);
  if (!phase) return <div className="py-32 text-center text-faint">{t('load_failed')}</div>;

  const done = phase.lessons.filter((l) => progress.lessons[`${phase.slug}/${l.slug}`]?.done).length;
  const pct = phase.lessons.length ? (done / phase.lessons.length) * 100 : 0;

  return (
    <div className="mx-auto max-w-4xl px-5 py-14">
      <Link to="/" className="font-mono text-[11px] tracking-[0.14em] text-faint hover:text-ink">
        ← {t('back_to_map')}
      </Link>

      <header className="mt-6 border-b border-hairline pb-8">
        <div className="flex items-baseline gap-4">
          <span className="font-serif text-[56px] font-semibold leading-none text-hairline">
            {String(phase.num).padStart(2, '0')}
          </span>
          <div>
            <h1 className="font-serif text-[34px] font-semibold tracking-tight">
              {phaseTitle(phase, lang)}
            </h1>
            <p className="font-mono text-[11px] tracking-[0.12em] text-faint">
              {lang === 'zh' ? phase.titleEn.toUpperCase() : phase.titleZh}
            </p>
          </div>
        </div>
        {lang === 'zh' && <p className="mt-4 max-w-2xl text-[15px] text-faint">{phase.descZh}</p>}
        <div className="mt-5 flex items-center gap-3">
          <div className="h-[3px] w-48 overflow-hidden rounded-full bg-bone">
            <div className="h-full bg-ink-green" style={{ width: `${pct}%` }} />
          </div>
          <span className="font-mono text-[11.5px] text-faint">
            {t('done_count')(done, phase.lessons.length)}
          </span>
        </div>
      </header>

      <ol>
        {phase.lessons.map((lesson, i) => {
          const id = `${phase.slug}/${lesson.slug}`;
          const isDone = Boolean(progress.lessons[id]?.done);
          return (
            <li
              key={lesson.slug}
              className="rise group flex items-center gap-4 border-b border-hairline py-4"
              style={{ ['--stagger' as string]: Math.min(i, 10) }}
            >
              <span className="w-7 shrink-0 font-mono text-[12px] text-faint">
                {String(i + 1).padStart(2, '0')}
              </span>
              <Link to={`/lesson/${phase.slug}/${lesson.slug}`} className="min-w-0 flex-1">
                <div className="truncate text-[15.5px] font-medium group-hover:underline group-hover:decoration-hairline group-hover:underline-offset-4">
                  {lessonTitle(lesson, lang)}
                </div>
                {lang === 'zh' && lesson.titleZh && (
                  <div className="truncate text-[12px] text-faint">{lesson.title}</div>
                )}
              </Link>
              <div className="flex shrink-0 items-center gap-2">
                {lesson.hasZh && lang === 'zh' && (
                  <span className="rounded-full bg-pale-blue px-2 py-0.5 text-[10.5px] text-ink-blue">
                    {t('zh_ready')}
                  </span>
                )}
                {lesson.hasQuiz && (
                  <span className="rounded-full bg-pale-yellow px-2 py-0.5 text-[10.5px] text-ink-yellow">
                    {t('quiz_label')}
                  </span>
                )}
                {lesson.runnable && (
                  <span className="rounded-full bg-pale-green px-2 py-0.5 text-[10.5px] text-ink-green">
                    {t('runnable_label')}
                  </span>
                )}
                {lesson.time && (
                  <span className="hidden font-mono text-[11px] text-faint sm:inline">
                    {lesson.time.replace('~', '').replace('minutes', 'min')}
                  </span>
                )}
                <span
                  aria-label={isDone ? t('done') : undefined}
                  className={`flex h-5 w-5 items-center justify-center rounded border ${
                    isDone
                      ? 'border-ink-green/40 bg-pale-green text-ink-green'
                      : 'border-hairline bg-paper text-transparent'
                  }`}
                >
                  <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                    <path d="M1.5 5.5L4 8L8.5 2.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                  </svg>
                </span>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
