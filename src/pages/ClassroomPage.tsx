import { lazy, Suspense, useEffect, useState } from 'react';
import { Link, Navigate, useParams } from 'react-router-dom';
import { TutorAvatar } from '../components/TutorAvatar';
import { classroomPack } from '../classroom/packs';
import {
  markClassroomComplete,
  tx,
  type ClassroomPack,
  type ClassroomScene,
} from '../lib/classroom';
import { fetchIndex } from '../lib/data';
import { lessonTitle, phaseTitle, useLang } from '../lib/i18n';
import type { CourseIndex } from '../lib/types';

const VectorPlayground = lazy(() =>
  import('../components/interactive/VectorPlayground').then((m) => ({ default: m.VectorPlayground })),
);
const AttentionHeatmap = lazy(() =>
  import('../components/interactive/AttentionHeatmap').then((m) => ({ default: m.AttentionHeatmap })),
);
const AgentLoopSim = lazy(() =>
  import('../components/interactive/AgentLoopSim').then((m) => ({ default: m.AgentLoopSim })),
);

const KIND_LABEL = {
  lecture: { zh: '讲解', en: 'Lecture' },
  lab: { zh: '实验', en: 'Lab' },
  quiz: { zh: '练习', en: 'Practice' },
  wrap: { zh: '小结', en: 'Wrap' },
} as const;

function Widget({ name }: { name: 'vector' | 'attention' | 'agent-loop' }) {
  if (name === 'vector') return <VectorPlayground />;
  if (name === 'attention') return <AttentionHeatmap />;
  return <AgentLoopSim />;
}

export function ClassroomPage() {
  const { phaseSlug, lessonSlug } = useParams();
  if (!phaseSlug || !lessonSlug) return <Navigate to="/" replace />;
  const lessonId = `${phaseSlug}/${lessonSlug}`;
  const pack = classroomPack(lessonId);
  if (!pack) return <Navigate to={`/lesson/${phaseSlug}/${lessonSlug}`} replace />;
  return <ClassroomSession key={lessonId} pack={pack} phaseSlug={phaseSlug} lessonSlug={lessonSlug} />;
}

function ClassroomSession({
  pack,
  phaseSlug,
  lessonSlug,
}: {
  pack: ClassroomPack;
  phaseSlug: string;
  lessonSlug: string;
}) {
  const { lang, t } = useLang();
  const [index, setIndex] = useState<CourseIndex | null>(null);
  const [sceneIndex, setSceneIndex] = useState(0);
  const [picks, setPicks] = useState<Record<string, number>>({});

  useEffect(() => {
    fetchIndex().then(setIndex);
  }, []);

  const scene = pack.scenes[sceneIndex] ?? pack.scenes[0];
  const total = pack.scenes.length;
  const last = sceneIndex >= total - 1;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target) {
        const tag = target.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable) return;
      }
      if (e.key === 'ArrowRight' || e.key === ' ') {
        e.preventDefault();
        setSceneIndex((i) => Math.min(total - 1, i + 1));
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        setSceneIndex((i) => Math.max(0, i - 1));
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [total]);

  const phase = index?.phases.find((p) => p.slug === phaseSlug);
  const lessonMeta = phase?.lessons.find((l) => l.slug === lessonSlug);
  const backTo = `/lesson/${phaseSlug}/${lessonSlug}`;

  const go = (next: number) => {
    const clamped = Math.max(0, Math.min(pack.scenes.length - 1, next));
    const target = pack.scenes[clamped];
    if (target?.kind === 'wrap') markClassroomComplete(pack.lessonId);
    setSceneIndex(clamped);
  };

  if (!scene) {
    return <div className="py-32 text-center text-faint">{t('loading')}</div>;
  }

  const kindLabel = lang === 'zh' ? KIND_LABEL[scene.kind].zh : KIND_LABEL[scene.kind].en;
  const avatarMode =
    scene.kind === 'lab'
      ? 'listening'
      : scene.kind === 'quiz' && picks[scene.id] != null
        ? picks[scene.id] === scene.correct
          ? 'happy'
          : 'confused'
        : scene.kind === 'wrap'
          ? 'happy'
          : 'talking';

  return (
    <div className="mx-auto max-w-5xl px-5 py-8 md:py-10">
      <nav className="font-mono text-[11px] tracking-[0.1em] text-faint">
        <Link to="/" className="hover:text-ink">
          {t('nav_map')}
        </Link>
        {phase && (
          <>
            <span className="mx-2">/</span>
            <Link to={`/phase/${phase.slug}`} className="hover:text-ink">
              {t('phase')} {phase.num} · {phaseTitle(phase, lang)}
            </Link>
          </>
        )}
        <span className="mx-2">/</span>
        <Link to={backTo} className="hover:text-ink">
          {lessonMeta ? lessonTitle(lessonMeta, lang) : tx(pack.title, lang)}
        </Link>
        <span className="mx-2">/</span>
        <span>{t('classroom')}</span>
      </nav>

      <header className="mt-4 flex flex-wrap items-end justify-between gap-3 border-b border-hairline pb-5">
        <div>
          <p className="font-mono text-[11px] tracking-[0.14em] text-faint">
            {kindLabel} · {sceneIndex + 1}/{total} · {pack.durationMin}
            {lang === 'zh' ? ' 分钟' : ' min'}
          </p>
          <h1 className="mt-1 font-serif text-[26px] font-semibold tracking-tight md:text-[32px]">
            {tx(pack.title, lang)}
          </h1>
          <p className="mt-1 max-w-2xl text-[14px] leading-relaxed text-faint">{tx(pack.subtitle, lang)}</p>
        </div>
        <Link
          to={backTo}
          className="rounded-md border border-hairline bg-paper px-3 py-1.5 text-[13px] text-faint hover:text-ink"
        >
          {t('classroom_back')}
        </Link>
      </header>

      <ol className="mt-5 flex flex-wrap gap-1.5" aria-label={lang === 'zh' ? '课堂页' : 'Scenes'}>
        {pack.scenes.map((item, i) => {
          const active = i === sceneIndex;
          const past = i < sceneIndex;
          return (
            <li key={item.id}>
              <button
                type="button"
                onClick={() => go(i)}
                className={`h-7 min-w-7 rounded-md px-2 font-mono text-[11px] ${
                  active
                    ? 'bg-ink text-paper'
                    : past
                      ? 'bg-pale-green text-ink-green'
                      : 'border border-hairline bg-paper text-faint hover:text-ink'
                }`}
                aria-current={active ? 'step' : undefined}
              >
                {i + 1}
              </button>
            </li>
          );
        })}
      </ol>

      <section className="mt-6 overflow-hidden rounded-lg border border-hairline bg-paper">
        <div className="border-b border-hairline bg-bone/50 px-5 py-4 md:px-7">
          <div className="flex items-start gap-3">
            <TutorAvatar mode={avatarMode} size={52} />
            <div className="min-w-0 flex-1">
              <p className="font-mono text-[11px] tracking-[0.12em] text-faint">{t('classroom_teacher')}</p>
              <p className="mt-1 text-[15px] leading-relaxed">{tx(scene.teacher, lang)}</p>
              {scene.kind === 'lecture' && scene.peer && (
                <p className="mt-3 rounded-md bg-pale-yellow px-3 py-2 text-[13.5px] leading-relaxed text-ink-yellow">
                  <span className="font-mono text-[10.5px] tracking-[0.12em]">{t('classroom_peer')} · </span>
                  {tx(scene.peer, lang)}
                </p>
              )}
            </div>
          </div>
        </div>

        <div className="px-5 py-6 md:px-7 md:py-8">
          <SceneBody
            scene={scene}
            pack={pack}
            pick={picks[scene.id] ?? null}
            onPick={(value) => setPicks((prev) => ({ ...prev, [scene.id]: value }))}
            backTo={backTo}
          />
        </div>
      </section>

      <nav className="mt-5 flex items-center justify-between gap-3">
        <button
          type="button"
          onClick={() => go(sceneIndex - 1)}
          disabled={sceneIndex === 0}
          className="rounded-md border border-hairline bg-paper px-4 py-2 text-[13.5px] disabled:opacity-40"
        >
          ← {t('classroom_prev')}
        </button>
        <p className="hidden font-mono text-[11px] text-faint sm:block">
          {lang === 'zh' ? '方向键翻页' : 'Arrow keys to turn'}
        </p>
        {last ? (
          <Link
            to={backTo}
            onClick={() => markClassroomComplete(pack.lessonId)}
            className="rounded-md bg-ink px-4 py-2 text-[13.5px] text-paper"
          >
            {t('classroom_finish')}
          </Link>
        ) : (
          <button
            type="button"
            onClick={() => go(sceneIndex + 1)}
            className="rounded-md bg-ink px-4 py-2 text-[13.5px] text-paper"
          >
            {t('classroom_next')} →
          </button>
        )}
      </nav>
    </div>
  );
}

function SceneBody({
  scene,
  pack,
  pick,
  onPick,
  backTo,
}: {
  scene: ClassroomScene;
  pack: ClassroomPack;
  pick: number | null;
  onPick: (value: number) => void;
  backTo: string;
}) {
  const { lang, t } = useLang();

  if (scene.kind === 'lecture') {
    return (
      <div>
        <h2 className="font-serif text-[22px] font-semibold tracking-tight">{tx(scene.title, lang)}</h2>
        <ul className="mt-4 space-y-2">
          {scene.bullets.map((item) => (
            <li key={item.zh} className="flex gap-2 text-[15px] leading-relaxed">
              <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-ink" />
              <span>{tx(item, lang)}</span>
            </li>
          ))}
        </ul>
        {scene.board && (
          <pre className="mt-5 overflow-x-auto rounded-md border border-hairline bg-bone px-4 py-3 font-mono text-[12.5px] leading-relaxed text-ink">
            {tx(scene.board, lang)}
          </pre>
        )}
      </div>
    );
  }

  if (scene.kind === 'lab') {
    return (
      <div>
        <h2 className="mb-2 font-serif text-[22px] font-semibold tracking-tight">{tx(scene.title, lang)}</h2>
        <div className="overflow-x-auto [&>section]:my-0">
          <Suspense fallback={<div className="h-72 animate-pulse rounded-lg border border-hairline bg-bone" />}>
            <Widget name={scene.widget} />
          </Suspense>
        </div>
      </div>
    );
  }

  if (scene.kind === 'quiz') {
    const revealed = pick != null;
    return (
      <div>
        <p className="font-mono text-[11px] tracking-[0.12em] text-faint">{t('classroom_practice')}</p>
        <h2 className="mt-1 font-serif text-[22px] font-semibold tracking-tight">{tx(scene.question, lang)}</h2>
        <ol className="mt-5 space-y-2">
          {scene.options.map((option, i) => {
            const chosen = pick === i;
            const right = i === scene.correct;
            let cls = 'border-hairline bg-paper hover:bg-bone';
            if (revealed && right) cls = 'border-ink-green/40 bg-pale-green text-ink-green';
            else if (revealed && chosen && !right) cls = 'border-ink-red/40 bg-pale-red text-ink-red';
            return (
              <li key={option.zh}>
                <button
                  type="button"
                  onClick={() => onPick(i)}
                  className={`flex w-full items-start gap-3 rounded-md border px-4 py-3 text-left text-[14.5px] leading-relaxed ${cls}`}
                >
                  <span className="font-mono text-[12px] text-faint">{String.fromCharCode(65 + i)}</span>
                  <span>{tx(option, lang)}</span>
                </button>
              </li>
            );
          })}
        </ol>
        {revealed && (
          <p className="mt-4 rounded-md bg-bone px-4 py-3 text-[13.5px] leading-relaxed text-faint">
            {tx(scene.explanation, lang)}
          </p>
        )}
      </div>
    );
  }

  return (
    <div>
      <h2 className="font-serif text-[22px] font-semibold tracking-tight">{tx(scene.title, lang)}</h2>
      <ul className="mt-4 space-y-2">
        {scene.takeaways.map((item) => (
          <li key={item.zh} className="flex gap-2 text-[15px] leading-relaxed">
            <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-ink-green" />
            <span>{tx(item, lang)}</span>
          </li>
        ))}
      </ul>
      <p className="mt-6 text-[13px] text-faint">
        {lang === 'zh'
          ? '课文仍是完整版本。互动课只过要点，测验不计入成绩。'
          : 'The lesson text remains the full version. This class covers the spine; practice does not count toward your score.'}
      </p>
      <Link
        to={backTo}
        onClick={() => markClassroomComplete(pack.lessonId)}
        className="mt-5 inline-flex rounded-md bg-ink px-4 py-2 text-[13.5px] text-paper"
      >
        {t('classroom_finish')}
      </Link>
    </div>
  );
}
