import { useEffect, useMemo, useState } from 'react';
import { useLang } from '../lib/i18n';
import { getProgress, saveQuizScore, setLessonDone } from '../lib/progress';
import type { QuizQuestion } from '../lib/types';

const LETTERS = ['A', 'B', 'C', 'D', 'E', 'F'];

interface QuizProps {
  lessonId: string;
  stage: 'pre' | 'post';
  questions: QuizQuestion[];
}

export function Quiz({ lessonId, stage, questions }: QuizProps) {
  const { lang, t } = useLang();
  const [picked, setPicked] = useState<(number | null)[]>(() => questions.map(() => null));
  // Score policy: first attempt counts; anything after is practice.
  const [hadRecordAtMount] = useState<boolean>(() => {
    const p = getProgress().lessons[lessonId];
    return stage === 'pre' ? p?.preTotal != null : p?.postTotal != null;
  });
  const [retried, setRetried] = useState(false);

  // Reset only when the question COUNT changes (e.g. zh/en fallback differs).
  // Same-length swaps (re-renders, language toggle) keep the user's answers —
  // an identity check here wiped all answers whenever the parent re-rendered.
  // Navigation between lessons remounts via the parent's `key` prop.
  if (picked.length !== questions.length) {
    setPicked(questions.map(() => null));
  }

  const answered = picked.filter((p) => p != null).length;
  const score = useMemo(
    () => picked.filter((p, i) => p === questions[i].correct).length,
    [picked, questions],
  );
  const finished = answered === questions.length;

  useEffect(() => {
    // Warm-up (pre) is practice only and is never recorded. Only the
    // post-lesson check counts toward grades, and saveQuizScore itself keeps
    // the first attempt (retakes don't overwrite).
    if (finished && questions.length > 0 && stage === 'post') {
      saveQuizScore(lessonId, stage, score, questions.length);
      setLessonDone(lessonId, true); // finishing the post-lesson check completes the lesson
    }
  }, [finished, score, lessonId, stage, questions.length]);

  if (questions.length === 0) return null;

  return (
    <section className="my-10 rounded-lg border border-hairline bg-paper">
      <header className="flex items-baseline justify-between border-b border-hairline px-6 py-4">
        <div className="flex items-baseline gap-3">
          <span
            className={`rounded-full px-2.5 py-0.5 text-[11px] font-medium tracking-[0.05em] uppercase ${
              stage === 'pre' ? 'bg-pale-yellow text-ink-yellow' : 'bg-pale-green text-ink-green'
            }`}
          >
            {stage === 'pre' ? t('pre_quiz') : t('post_quiz')}
          </span>
          <span className="text-[13px] text-faint">
            {stage === 'pre' ? t('pre_quiz_hint') : t('post_quiz_hint')}
          </span>
        </div>
        <span className="font-mono text-[12px] text-faint">
          {answered}/{questions.length}
        </span>
      </header>

      <ol className="divide-y divide-hairline">
        {questions.map((q, qi) => {
          const chosen = picked[qi];
          const revealed = chosen != null;
          return (
            <li key={qi} className="px-6 py-5">
              <p className="mb-3 font-medium leading-relaxed">
                <span className="mr-2 font-mono text-[12px] text-faint">{qi + 1}.</span>
                {q.question}
              </p>
              <div className="space-y-1.5">
                {q.options.map((opt, oi) => {
                  const isCorrect = oi === q.correct;
                  const isChosen = oi === chosen;
                  let cls = 'border-hairline bg-paper hover:bg-bone';
                  if (revealed) {
                    if (isCorrect) cls = 'border-ink-green/30 bg-pale-green';
                    else if (isChosen) cls = 'border-ink-red/30 bg-pale-red';
                    else cls = 'border-hairline bg-paper opacity-60';
                  }
                  return (
                    <button
                      key={oi}
                      type="button"
                      disabled={revealed}
                      onClick={() => setPicked((prev) => prev.map((p, i) => (i === qi ? oi : p)))}
                      className={`flex w-full items-start gap-3 rounded-md border px-4 py-2.5 text-left text-[15px] leading-relaxed transition-colors ${cls} ${revealed ? '' : 'cursor-pointer'}`}
                    >
                      <span className="mt-0.5 font-mono text-[12px] text-faint">{LETTERS[oi]}</span>
                      <span className="flex-1">{opt}</span>
                      {revealed && isCorrect && (
                        <span className="mt-0.5 font-mono text-[12px] text-ink-green">✓</span>
                      )}
                      {revealed && isChosen && !isCorrect && (
                        <span className="mt-0.5 font-mono text-[12px] text-ink-red">✗</span>
                      )}
                    </button>
                  );
                })}
              </div>
              {revealed && (
                <p className="mt-3 rounded-md bg-bone px-4 py-3 text-[14px] leading-relaxed text-faint">
                  <span className={`mr-2 font-medium ${chosen === q.correct ? 'text-ink-green' : 'text-ink-red'}`}>
                    {chosen === q.correct ? t('correct') : t('wrong')}
                  </span>
                  {q.explanation}
                </p>
              )}
            </li>
          );
        })}
      </ol>

      {finished && (
        <footer className="flex flex-wrap items-center justify-between gap-3 border-t border-hairline px-6 py-4">
          <span className="text-[14px]">
            {t('score')}{' '}
            <span className="font-serif text-xl font-semibold">
              {score}/{questions.length}
            </span>
            {(() => {
              if (!hadRecordAtMount && !retried) return null;
              const p = getProgress().lessons[lessonId];
              const rec =
                stage === 'pre'
                  ? p?.preTotal != null
                    ? { s: p.preScore ?? 0, t: p.preTotal }
                    : null
                  : p?.postTotal != null
                    ? { s: p.postScore ?? 0, t: p.postTotal }
                    : null;
              if (!rec) return null;
              return (
                <span className="ml-3 rounded-full bg-pale-yellow px-2.5 py-0.5 text-[11.5px] text-ink-yellow">
                  {lang === 'zh'
                    ? `成绩以首次作答为准（${rec.s}/${rec.t}），重做不计入`
                    : `First attempt counts (${rec.s}/${rec.t}); retakes are practice`}
                </span>
              );
            })()}
          </span>
          <button
            type="button"
            onClick={() => {
              setRetried(true);
              setPicked(questions.map(() => null));
            }}
            className="rounded-md border border-hairline px-3 py-1.5 text-[13px] text-faint transition-colors hover:bg-bone hover:text-ink"
          >
            {t('retry')}
          </button>
        </footer>
      )}
    </section>
  );
}
