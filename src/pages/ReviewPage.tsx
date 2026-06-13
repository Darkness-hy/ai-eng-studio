import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { fetchLesson } from '../lib/data';
import { useLang } from '../lib/i18n';
import { dueItems, gradeReview } from '../lib/review';
import type { QuizQuestion } from '../lib/types';

const LETTERS = ['A', 'B', 'C', 'D', 'E', 'F'];

interface ReviewCard {
  id: string;
  phaseSlug: string;
  lessonSlug: string;
  lessonTitle: string;
  lessonTitleZh: string | null;
  q: QuizQuestion; // English (canonical correct index)
  qZh: QuizQuestion | null;
}

export function ReviewPage() {
  const { lang, t } = useLang();
  const zh = lang === 'zh';
  const [cards, setCards] = useState<ReviewCard[] | null>(null);
  const [cursor, setCursor] = useState(0);
  const [picked, setPicked] = useState<number | null>(null);
  const [reviewed, setReviewed] = useState(0);

  // Snapshot the due queue once on mount (grading changes due dates live).
  useEffect(() => {
    const items = dueItems();
    const lessonIds = [...new Set(items.map((i) => i.lessonId))];
    Promise.all(
      lessonIds.map((lid) => {
        const [phaseSlug, lessonSlug] = lid.split('/');
        return fetchLesson(phaseSlug, lessonSlug)
          .then((l) => [lid, l] as const)
          .catch(() => [lid, null] as const);
      }),
    ).then((pairs) => {
      const byId = new Map(pairs);
      const built: ReviewCard[] = [];
      for (const it of items) {
        const l = byId.get(it.lessonId);
        if (!l) continue;
        const postEn = (l.quizEn ?? []).filter((q) => q.stage !== 'pre');
        const postZh = l.quizZh ? l.quizZh.filter((q) => q.stage !== 'pre') : null;
        const q = postEn[it.postIdx];
        if (!q) continue;
        const [phaseSlug, lessonSlug] = it.lessonId.split('/');
        built.push({
          id: it.id,
          phaseSlug,
          lessonSlug,
          lessonTitle: l.title,
          lessonTitleZh: l.titleZh,
          q,
          qZh: postZh?.[it.postIdx] ?? null,
        });
      }
      setCards(built);
    });
  }, []);

  if (cards === null) return <Shell><p className="text-center text-faint">{t('loading')}</p></Shell>;

  if (cards.length === 0) {
    return (
      <Shell>
        <div className="rounded-lg border border-hairline bg-paper px-6 py-10 text-center">
          <p className="font-serif text-[20px]">{zh ? '今天没有需要复习的题' : 'Nothing due today'}</p>
          <p className="mt-2 text-[14px] text-faint">
            {zh
              ? '课后检验里答错的题会自动进入复习队列，按记忆曲线在合适的时间提醒你重做。'
              : 'Questions you miss in post-lesson checks enter this queue and resurface on a spaced schedule.'}
          </p>
        </div>
      </Shell>
    );
  }

  if (cursor >= cards.length) {
    return (
      <Shell>
        <div className="rounded-lg border border-ink-green/30 bg-pale-green px-6 py-10 text-center">
          <p className="font-serif text-[22px] font-semibold text-ink-green">
            {zh ? `本次复习完成 ${reviewed} 题` : `Reviewed ${reviewed} question(s)`}
          </p>
          <p className="mt-2 text-[14px] text-ink-green/80">
            {zh ? '答对的题会拉长下次复习间隔，答错的会很快再出现。' : 'Right answers push the next review further out; wrong ones come back soon.'}
          </p>
        </div>
      </Shell>
    );
  }

  const card = cards[cursor];
  const q = zh && card.qZh ? card.qZh : card.q;
  const revealed = picked != null;

  const advance = () => {
    gradeReview(card.id, picked === card.q.correct);
    setReviewed((n) => n + 1);
    setPicked(null);
    setCursor((c) => c + 1);
  };

  return (
    <Shell>
      <div className="mb-4 flex items-baseline justify-between">
        <span className="font-mono text-[12px] text-faint">
          {cursor + 1} / {cards.length}
        </span>
        <Link
          to={`/lesson/${card.phaseSlug}/${card.lessonSlug}`}
          className="font-mono text-[11.5px] text-faint underline decoration-hairline underline-offset-2 hover:text-ink"
        >
          {zh ? (card.lessonTitleZh ?? card.lessonTitle) : card.lessonTitle} ↗
        </Link>
      </div>

      <section className="rounded-lg border border-hairline bg-paper px-6 py-5">
        <p className="font-medium leading-relaxed">{zh ? q.question : card.q.question}</p>
        <div className="mt-3 space-y-1.5">
          {(zh ? q.options : card.q.options).map((opt, oi) => {
            const isCorrect = oi === card.q.correct;
            const isChosen = oi === picked;
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
                onClick={() => setPicked(oi)}
                className={`flex w-full items-start gap-3 rounded-md border px-4 py-2.5 text-left text-[15px] leading-relaxed transition-colors ${cls}`}
              >
                <span className="mt-0.5 font-mono text-[12px] text-faint">{LETTERS[oi]}</span>
                <span className="flex-1">{opt}</span>
              </button>
            );
          })}
        </div>
        {revealed && (
          <p className="mt-3 rounded-md bg-bone px-4 py-3 text-[14px] leading-relaxed text-faint">
            <span className={`mr-2 font-medium ${picked === card.q.correct ? 'text-ink-green' : 'text-ink-red'}`}>
              {picked === card.q.correct ? (zh ? '答对了' : 'Correct') : zh ? '不对' : 'Not quite'}
            </span>
            {zh ? q.explanation : card.q.explanation}
          </p>
        )}
      </section>

      {revealed && (
        <div className="mt-4 text-right">
          <button
            type="button"
            onClick={advance}
            className="rounded-md bg-ink px-5 py-2 text-[14px] text-white transition-colors hover:bg-ink/85"
          >
            {cursor + 1 >= cards.length ? (zh ? '完成' : 'Finish') : zh ? '下一题' : 'Next'}
          </button>
        </div>
      )}
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  const { lang } = useLang();
  return (
    <div className="mx-auto max-w-2xl px-5 py-14">
      <p className="font-mono text-[11px] tracking-[0.18em] text-faint">SPACED REVIEW</p>
      <h1 className="mt-2 mb-8 font-serif text-[34px] font-semibold tracking-tight">
        {lang === 'zh' ? '错题复习' : 'Review'}
      </h1>
      {children}
    </div>
  );
}
