import { useEffect, useState } from 'react';
import { fetchGlossary } from '../lib/data';
import { buildSession, gradeCard } from '../lib/flashcards';
import { useLang } from '../lib/i18n';
import type { GlossaryTerm } from '../lib/types';

export function FlashcardsPage() {
  const { lang, t } = useLang();
  const zh = lang === 'zh';
  const [session, setSession] = useState<GlossaryTerm[] | null>(null);
  const [cursor, setCursor] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [reviewed, setReviewed] = useState(0);

  // Snapshot the session once (grading changes due dates live).
  useEffect(() => {
    fetchGlossary().then((g) => {
      const byTerm = new Map(g.terms.map((x) => [x.term, x]));
      const names = buildSession(g.terms.map((x) => x.term));
      setSession(names.map((n) => byTerm.get(n)!).filter(Boolean));
    });
  }, []);

  if (!session) return <Shell><p className="text-center text-faint">{t('loading')}</p></Shell>;

  if (session.length === 0) {
    return (
      <Shell>
        <div className="rounded-lg border border-hairline bg-paper px-6 py-10 text-center">
          <p className="font-serif text-[20px]">{zh ? '今天的闪卡都复习完了' : 'No cards due today'}</p>
          <p className="mt-2 text-[14px] text-faint">
            {zh ? '记得的术语会拉长复习间隔,明天再来巩固新的。' : 'Remembered cards return later; new ones tomorrow.'}
          </p>
        </div>
      </Shell>
    );
  }

  if (cursor >= session.length) {
    return (
      <Shell>
        <div className="rounded-lg border border-ink-green/30 bg-pale-green px-6 py-10 text-center">
          <p className="font-serif text-[22px] font-semibold text-ink-green">
            {zh ? `本次复习了 ${reviewed} 张闪卡` : `Reviewed ${reviewed} cards`}
          </p>
        </div>
      </Shell>
    );
  }

  const term = session[cursor];
  const front = zh && term.zh ? term.zh.term : term.term;
  const meaning = zh && term.zh ? term.zh.meaning : term.meaning;
  const origin = zh && term.zh ? term.zh.origin : term.origin;

  const grade = (remembered: boolean) => {
    gradeCard(term.term, remembered);
    setReviewed((n) => n + 1);
    setFlipped(false);
    setCursor((c) => c + 1);
  };

  return (
    <Shell>
      <div className="mb-4 font-mono text-[12px] text-faint">
        {cursor + 1} / {session.length}
      </div>
      <button
        type="button"
        onClick={() => setFlipped((f) => !f)}
        className="flex min-h-56 w-full flex-col items-center justify-center rounded-lg border border-hairline bg-paper px-6 py-10 text-center transition-colors hover:bg-bone/40"
      >
        {!flipped ? (
          <>
            <span className="font-serif text-[28px] font-semibold tracking-tight">{front}</span>
            {zh && term.zh && <span className="mt-1 font-mono text-[12px] text-faint">{term.term}</span>}
            <span className="mt-6 font-mono text-[11px] uppercase tracking-[0.14em] text-faint">
              {zh ? '点击翻面' : 'tap to flip'}
            </span>
          </>
        ) : (
          <>
            <p className="max-w-xl text-[16px] leading-relaxed">{meaning}</p>
            {origin && <p className="mt-3 max-w-xl text-[13.5px] leading-relaxed text-faint">{origin}</p>}
          </>
        )}
      </button>

      {flipped && (
        <div className="mt-4 flex justify-center gap-3">
          <button
            type="button"
            onClick={() => grade(false)}
            className="rounded-md border border-ink-red/30 bg-pale-red px-6 py-2 text-[14px] text-ink-red transition-colors hover:bg-pale-red/70"
          >
            {zh ? '忘了' : 'Forgot'}
          </button>
          <button
            type="button"
            onClick={() => grade(true)}
            className="rounded-md border border-ink-green/30 bg-pale-green px-6 py-2 text-[14px] text-ink-green transition-colors hover:bg-pale-green/70"
          >
            {zh ? '记得' : 'Got it'}
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
      <p className="font-mono text-[11px] tracking-[0.18em] text-faint">FLASHCARDS</p>
      <h1 className="mt-2 mb-8 font-serif text-[34px] font-semibold tracking-tight">
        {lang === 'zh' ? '术语闪卡' : 'Flashcards'}
      </h1>
      {children}
    </div>
  );
}
