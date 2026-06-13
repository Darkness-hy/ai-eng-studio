import { Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { ChallengeWidget } from '../components/ChallengeWidget';
import { CodeTabs } from '../components/CodeTabs';
import { INTERACTIVE } from '../components/interactive/registry';
import { Markdown } from '../components/Markdown';
import { Quiz } from '../components/Quiz';
import { fetchIndex, fetchLesson } from '../lib/data';
import { lessonTitle, phaseTitle, useLang } from '../lib/i18n';
import { extractToc, stripLessonHeader } from '../lib/md';
import { recordVisit, setLessonDone, useProgress } from '../lib/progress';
import type { CourseIndex, Lesson } from '../lib/types';

export function LessonPage() {
  const { phaseSlug, lessonSlug } = useParams();
  const { lang, t } = useLang();
  const routeId = `${phaseSlug}/${lessonSlug}`;
  const [index, setIndex] = useState<CourseIndex | null>(null);
  const [loaded, setLoaded] = useState<{ key: string; lesson?: Lesson; failed?: boolean } | null>(null);
  const [originalFor, setOriginalFor] = useState<string | null>(null);
  const progress = useProgress();

  const lesson = loaded?.key === routeId ? (loaded.lesson ?? null) : null;
  const failed = loaded?.key === routeId && loaded.failed === true;
  const showOriginal = originalFor === routeId;

  useEffect(() => {
    fetchIndex().then(setIndex);
  }, []);

  useEffect(() => {
    if (!phaseSlug || !lessonSlug) return;
    let live = true;
    const key = `${phaseSlug}/${lessonSlug}`;
    fetchLesson(phaseSlug, lessonSlug)
      .then((l) => {
        if (!live) return;
        setLoaded({ key, lesson: l });
        recordVisit(l.id);
      })
      .catch(() => {
        if (live) setLoaded({ key, failed: true });
      });
    return () => {
      live = false;
    };
  }, [phaseSlug, lessonSlug]);

  const phase = index?.phases.find((p) => p.slug === phaseSlug);
  const lessonIdx = phase?.lessons.findIndex((l) => l.slug === lessonSlug) ?? -1;
  const prev = phase && lessonIdx > 0 ? phase.lessons[lessonIdx - 1] : null;
  const next = phase && lessonIdx >= 0 && lessonIdx < phase.lessons.length - 1 ? phase.lessons[lessonIdx + 1] : null;

  const useZh = lang === 'zh' && lesson?.bodyZh != null && !showOriginal;
  const body = useMemo(() => {
    if (!lesson) return '';
    return stripLessonHeader(useZh ? lesson.bodyZh! : lesson.bodyEn);
  }, [lesson, useZh]);
  const toc = useMemo(() => extractToc(body), [body]);
  const activeId = useScrollSpy(toc.map((item) => item.id), body);

  // Hooks must run unconditionally — compute quiz arrays before any early return.
  // Upstream uses three stages: pre (warm-up), post and check (both verify
  // after reading) — render post+check together so no questions are dropped.
  const quiz = lesson ? ((lang === 'zh' && lesson.quizZh) || lesson.quizEn) : null;
  const preQuiz = useMemo(() => quiz?.filter((q) => q.stage === 'pre') ?? [], [quiz]);
  const postQuiz = useMemo(() => quiz?.filter((q) => q.stage !== 'pre') ?? [], [quiz]);

  if (failed) return <div className="py-32 text-center text-faint">{t('load_failed')}</div>;
  if (!lesson || !index || !phase) {
    return <div className="py-32 text-center text-faint">{t('loading')}</div>;
  }
  const isDone = Boolean(progress.lessons[lesson.id]?.done);
  const title = lessonTitle(lesson, lang);

  return (
    <div className="mx-auto grid max-w-6xl grid-cols-1 gap-10 px-5 py-12 xl:grid-cols-[1fr_210px]">
      <article className="mx-auto w-full max-w-3xl min-w-0">
        {/* breadcrumb + header */}
        <nav className="font-mono text-[11px] tracking-[0.1em] text-faint">
          <Link to="/" className="hover:text-ink">
            {t('nav_map')}
          </Link>
          <span className="mx-2">/</span>
          <Link to={`/phase/${phase.slug}`} className="hover:text-ink">
            {t('phase')} {phase.num} · {phaseTitle(phase, lang)}
          </Link>
        </nav>

        <header className="mt-5 border-b border-hairline pb-7">
          <div className="flex items-start justify-between gap-6">
            <h1 className="font-serif text-[34px] font-semibold leading-tight tracking-tight md:text-[40px]">
              {title}
            </h1>
            {isDone && (
              <span className="mt-2 shrink-0 rounded-md border border-ink-green/30 bg-pale-green px-3 py-1.5 text-[12.5px] text-ink-green">
                ✓ {t('done')}
              </span>
            )}
          </div>
          {lang === 'zh' && lesson.titleZh && (
            <p className="mt-1 font-mono text-[11.5px] tracking-[0.08em] text-faint">{lesson.title}</p>
          )}
          {lesson.quote && (
            <p className="mt-4 font-serif text-[17px] italic leading-relaxed text-faint">
              {lesson.quote}
            </p>
          )}
          <div className="mt-5 flex flex-wrap items-center gap-2 font-mono text-[11px] text-faint">
            {lesson.meta.type && (
              <span className="rounded-full bg-pale-blue px-2.5 py-0.5 text-ink-blue">
                {lesson.meta.type}
              </span>
            )}
            {lesson.meta.time && (
              <span className="rounded-full border border-hairline px-2.5 py-0.5">{lesson.meta.time}</span>
            )}
            {lesson.meta.languages && (
              <span className="rounded-full border border-hairline px-2.5 py-0.5">
                {lesson.meta.languages}
              </span>
            )}
          </div>
        </header>

        {/* translation status */}
        {lang === 'zh' && !lesson.bodyZh && (
          <p className="mt-6 rounded-md bg-pale-yellow px-4 py-3 text-[13.5px] text-ink-yellow">
            {t('not_translated')}
          </p>
        )}
        {lang === 'zh' && lesson.bodyZh && (
          <p className="mt-6 text-right">
            <button
              type="button"
              onClick={() => setOriginalFor(showOriginal ? null : routeId)}
              className="font-mono text-[11.5px] text-faint underline decoration-hairline underline-offset-2 hover:text-ink"
            >
              {showOriginal ? t('show_translation') : t('show_original')}
            </button>
          </p>
        )}

        {preQuiz.length > 0 && (
          <Quiz key={`${lesson.id}:pre`} lessonId={lesson.id} stage="pre" questions={preQuiz} />
        )}

        {INTERACTIVE[lesson.id] && (
          <Suspense
            fallback={<div className="my-10 h-72 animate-pulse rounded-lg border border-hairline bg-bone" />}
          >
            {(() => {
              const Widget = INTERACTIVE[lesson.id];
              return <Widget />;
            })()}
          </Suspense>
        )}

        <Markdown content={body} />

        {lesson.code.length > 0 && <CodeTabs files={lesson.code} />}

        {lesson.challenge && (
          <ChallengeWidget key={lesson.id} challenge={lesson.challenge} lessonId={lesson.id} />
        )}

        {postQuiz.length > 0 && (
          <Quiz key={`${lesson.id}:post`} lessonId={lesson.id} stage="post" questions={postQuiz} />
        )}

        {/* Completion rule: a lesson WITH a post-quiz completes only by
            finishing that quiz. Lessons with NO post-quiz have no such signal,
            so they complete when the reader reaches the bottom. */}
        {postQuiz.length === 0 && <CompletionSentinel lessonId={lesson.id} done={isDone} />}

        {/* prev / next */}
        <nav className="mt-14 grid grid-cols-2 gap-3 border-t border-hairline pt-6">
          {prev ? (
            <Link
              to={`/lesson/${phase.slug}/${prev.slug}`}
              className="group rounded-lg border border-hairline bg-paper px-4 py-3 transition-all hover:-translate-y-0.5 hover:shadow-lift"
            >
              <div className="font-mono text-[10.5px] tracking-[0.12em] text-faint">← {t('prev_lesson')}</div>
              <div className="mt-1 truncate text-[14px] font-medium">{lessonTitle(prev, lang)}</div>
            </Link>
          ) : (
            <span />
          )}
          {next ? (
            <Link
              to={`/lesson/${phase.slug}/${next.slug}`}
              className="group rounded-lg border border-hairline bg-paper px-4 py-3 text-right transition-all hover:-translate-y-0.5 hover:shadow-lift"
            >
              <div className="font-mono text-[10.5px] tracking-[0.12em] text-faint">{t('next_lesson')} →</div>
              <div className="mt-1 truncate text-[14px] font-medium">{lessonTitle(next, lang)}</div>
            </Link>
          ) : (
            <span />
          )}
        </nav>
      </article>

      {/* TOC */}
      <aside className="hidden xl:block">
        <div className="sticky top-20">
          <p className="font-mono text-[10.5px] tracking-[0.16em] text-faint">{t('toc').toUpperCase()}</p>
          <ul className="mt-3 space-y-1.5 border-l border-hairline">
            {toc
              .filter((item) => item.depth === 2)
              .map((item) => (
                <li key={item.id}>
                  <a
                    href={`#${item.id}`}
                    className={`-ml-px block border-l py-0.5 pl-3 text-[12.5px] leading-snug transition-colors ${
                      activeId === item.id
                        ? 'border-ink text-ink'
                        : 'border-transparent text-faint hover:text-ink'
                    }`}
                  >
                    {item.text}
                  </a>
                </li>
              ))}
          </ul>
        </div>
      </aside>
    </div>
  );
}

/** A 1px marker just before the prev/next nav. When it scrolls into view the
 *  reader has reached the end of the lesson, so mark it complete. */
function CompletionSentinel({ lessonId, done }: { lessonId: string; done: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (done) return;
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setLessonDone(lessonId, true);
          io.disconnect();
        }
      },
      { rootMargin: '0px 0px -5% 0px' },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [lessonId, done]);
  return <div ref={ref} aria-hidden className="h-px w-full" />;
}

function useScrollSpy(ids: string[], resetKey: string): string | null {
  const [active, setActive] = useState<string | null>(null);

  useEffect(() => {
    if (ids.length === 0) return;
    let ticking = false;
    const onScroll = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        ticking = false;
        let current: string | null = ids[0] ?? null;
        for (const id of ids) {
          const el = document.getElementById(id);
          if (el && el.getBoundingClientRect().top <= 120) current = id;
        }
        setActive(current);
      });
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
    return () => window.removeEventListener('scroll', onScroll);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resetKey, ids.join('|')]);

  return active;
}
