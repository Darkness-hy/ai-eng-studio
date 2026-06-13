import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { RoadmapList, RoadmapSpine } from '../components/RoadmapSpine';
import { fetchIndex } from '../lib/data';
import { lessonTitle, phaseTitle, useLang } from '../lib/i18n';
import { AREAS, QUESTIONS_PER_AREA, usePlacement } from '../lib/placement';
import { useProgress } from '../lib/progress';
import type { CourseIndex } from '../lib/types';

export function Dashboard() {
  const { lang, t } = useLang();
  const [index, setIndex] = useState<CourseIndex | null>(null);
  const progress = useProgress();

  useEffect(() => {
    fetchIndex().then(setIndex);
  }, []);

  if (!index) {
    return <div className="py-32 text-center text-faint">{t('loading')}</div>;
  }

  const continueTarget = (() => {
    if (!progress.lastLesson) return null;
    const [phaseSlug, lessonSlug] = progress.lastLesson.split('/');
    const phase = index.phases.find((p) => p.slug === phaseSlug);
    const lesson = phase?.lessons.find((l) => l.slug === lessonSlug);
    return phase && lesson ? { phase, lesson } : null;
  })();

  // Count only lessons that still exist in the catalog — stale ids in
  // localStorage (e.g. removed capstones) must not inflate the total.
  const doneTotal = index.phases.reduce(
    (acc, p) => acc + p.lessons.filter((l) => progress.lessons[`${p.slug}/${l.slug}`]?.done).length,
    0,
  );
  const donePct = index.stats.lessons ? (doneTotal / index.stats.lessons) * 100 : 0;

  return (
    <div className="mx-auto max-w-6xl px-5">
      {/* hero */}
      <section className="grid gap-10 py-16 md:grid-cols-[1.4fr_1fr] md:py-24">
        <div>
          <p className="rise font-mono text-[11px] tracking-[0.22em] text-faint" style={{ ['--stagger' as string]: 0 }}>
            AI ENGINEERING · FROM SCRATCH
          </p>
          <h1
            className="rise mt-5 font-serif text-[44px] font-semibold leading-[1.12] tracking-tight md:text-[60px]"
            style={{ ['--stagger' as string]: 1 }}
          >
            {lang === 'zh' ? (
              <>
                从零开始，
                <br />
                亲手造出 <em className="italic">AI</em>
              </>
            ) : (
              <>
                Build AI
                <br />
                <em className="italic">by hand</em>, from zero
              </>
            )}
          </h1>
          <p
            className="rise mt-6 max-w-xl text-[16px] leading-relaxed text-faint"
            style={{ ['--stagger' as string]: 2 }}
          >
            {lang === 'zh'
              ? '反向传播、分词器、注意力机制、Agent 循环——每一个算法先用裸数学推导，再亲手写出来，最后跑过测试。框架出场之前，你已经知道它在做什么。'
              : 'Backprop, tokenizers, attention, the agent loop — derive the math first, write the code by hand, run the test. By the time the framework shows up, you already know what it does.'}
          </p>
          <div
            className="rise mt-8 flex flex-wrap items-center gap-x-6 gap-y-2 font-mono text-[12px] text-faint"
            style={{ ['--stagger' as string]: 3 }}
          >
            <Stat n={index.stats.lessons} label={t('lessons')} />
            <Stat n={index.stats.phases} label={t('phases')} />
            <Stat n={index.stats.quizzes} label={t('quizzes')} />
            <Stat n={320} label={`+ ${t('hours')}`} />
          </div>
        </div>

        <div className="flex flex-col justify-end gap-3">
          <Link
            to={
              continueTarget
                ? `/lesson/${continueTarget.phase.slug}/${continueTarget.lesson.slug}`
                : `/lesson/${index.phases[0].slug}/${index.phases[0].lessons[0].slug}`
            }
            className="rise group rounded-lg border border-hairline bg-paper p-5 transition-all hover:-translate-y-0.5 hover:shadow-lift"
            style={{ ['--stagger' as string]: 4 }}
          >
            <div className="font-mono text-[10.5px] tracking-[0.16em] text-faint">
              {continueTarget ? t('continue_learning').toUpperCase() : t('start_learning').toUpperCase()}
            </div>
            <div className="mt-2 font-serif text-[20px] font-semibold leading-snug">
              {continueTarget
                ? lessonTitle(continueTarget.lesson, lang)
                : lessonTitle(index.phases[0].lessons[0], lang)}
            </div>
            <div className="mt-1 text-[13px] text-faint">
              {continueTarget
                ? `${t('phase')} ${continueTarget.phase.num} · ${phaseTitle(continueTarget.phase, lang)}`
                : `${t('phase')} 0 · ${phaseTitle(index.phases[0], lang)}`}
            </div>
            <div className="mt-4 text-right font-serif text-[18px] text-faint transition-transform group-hover:translate-x-1">
              →
            </div>
          </Link>
          <PlacementCard index={index} />
          <div
            className="rise rounded-lg border border-hairline bg-bone/60 px-5 py-4"
            style={{ ['--stagger' as string]: 5 }}
          >
            <div className="flex items-baseline justify-between">
              <span className="text-[13px] text-faint">{t('progress_overall')}</span>
              <span className="font-mono text-[12px] text-faint">
                {doneTotal} / {index.stats.lessons} · {donePct < 1 && doneTotal > 0 ? '<1' : Math.round(donePct)}%
              </span>
            </div>
            <div className="mt-3 h-[6px] overflow-hidden rounded-full bg-hairline/60">
              <div
                className="h-full rounded-full bg-ink-green transition-[width] duration-500"
                style={{ width: `${Math.max(donePct, doneTotal > 0 ? 1 : 0)}%` }}
              />
            </div>
          </div>
        </div>
      </section>

      {/* roadmap */}
      <section className="border-t border-hairline py-14">
        <div className="mb-10 flex items-baseline justify-between">
          <h2 className="font-serif text-[30px] font-semibold tracking-tight">{t('nav_map')}</h2>
          <span className="font-mono text-[11px] tracking-[0.14em] text-faint">
            {lang === 'zh' ? '数学是地板，Agent 是屋顶' : 'MATH IS THE FLOOR, AGENTS ARE THE ROOF'}
          </span>
        </div>
        <RoadmapSpine index={index} />
        <RoadmapList index={index} />
      </section>
    </div>
  );
}

function PlacementCard({ index }: { index: CourseIndex }) {
  const { lang } = useLang();
  const zh = lang === 'zh';
  const placement = usePlacement();
  const entry = placement ? index.phases.find((p) => p.num === placement.entry) : null;

  if (placement && entry) {
    return (
      <Link
        to="/find-your-level"
        className="rise flex items-baseline justify-between rounded-lg border border-hairline bg-paper px-5 py-3.5 transition-all hover:-translate-y-0.5 hover:shadow-lift"
        style={{ ['--stagger' as string]: 5 }}
      >
        <span className="text-[13px] text-faint">{zh ? '你的起点' : 'Your level'}</span>
        <span className="text-[13.5px] font-medium">
          {zh ? `阶段 ${entry.num} · ${entry.titleZh}` : `Phase ${entry.num} · ${entry.titleEn}`}
          <span className="ml-2 font-mono text-[11px] text-faint">{placement.total}/{AREAS.length * QUESTIONS_PER_AREA}</span>
        </span>
      </Link>
    );
  }
  return (
    <Link
      to="/find-your-level"
      className="rise group flex items-center justify-between rounded-lg border border-ink-blue/25 bg-pale-blue px-5 py-3.5 transition-all hover:-translate-y-0.5 hover:shadow-lift"
      style={{ ['--stagger' as string]: 5 }}
    >
      <span className="text-[13.5px] font-medium text-ink-blue">
        {zh ? '不知道从哪开始？50 题找到你的起点' : 'Not sure where to start? Find your level'}
      </span>
      <span className="font-serif text-[16px] text-ink-blue transition-transform group-hover:translate-x-1">
        →
      </span>
    </Link>
  );
}

function Stat({ n, label }: { n: number; label: string }) {
  return (
    <span>
      <span className="font-serif text-[20px] font-semibold text-ink">{n}</span>{' '}
      <span>{label}</span>
    </span>
  );
}
