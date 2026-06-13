import { useEffect, useMemo, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { fetchGlossary } from '../lib/data';
import { useLang } from '../lib/i18n';
import { slugify } from '../lib/md';
import type { GlossaryTerm } from '../lib/types';

export function GlossaryPage() {
  const { lang, t } = useLang();
  const [terms, setTerms] = useState<GlossaryTerm[] | null>(null);
  const [query, setQuery] = useState('');
  const { hash } = useLocation();

  useEffect(() => {
    fetchGlossary().then((g) => setTerms(g.terms));
  }, []);

  useEffect(() => {
    if (!terms || !hash) return;
    const el = document.getElementById(hash.slice(1));
    if (el) el.scrollIntoView({ block: 'start' });
  }, [terms, hash]);

  const filtered = useMemo(() => {
    if (!terms) return [];
    const q = query.trim().toLowerCase();
    if (!q) return terms;
    return terms.filter((term) =>
      `${term.term} ${term.zh?.term ?? ''} ${term.meaning} ${term.zh?.meaning ?? ''}`
        .toLowerCase()
        .includes(q),
    );
  }, [terms, query]);

  if (!terms) return <div className="py-32 text-center text-faint">{t('loading')}</div>;

  return (
    <div className="mx-auto max-w-4xl px-5 py-14">
      <header className="border-b border-hairline pb-8">
        <div className="flex items-start justify-between gap-4">
          <h1 className="font-serif text-[38px] font-semibold tracking-tight">{t('glossary_title')}</h1>
          <Link
            to="/flashcards"
            className="mt-2 shrink-0 rounded-md border border-hairline bg-paper px-3 py-1.5 text-[13px] text-faint transition-colors hover:bg-bone hover:text-ink"
          >
            {lang === 'zh' ? '术语闪卡 →' : 'Flashcards →'}
          </Link>
        </div>
        <p className="mt-2 text-[15px] text-faint">{t('glossary_sub')}</p>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={t('search_placeholder')}
          className="mt-6 w-full max-w-sm rounded-md border border-hairline bg-paper px-4 py-2 text-[14px] outline-none transition-colors placeholder:text-faint focus:border-faint"
        />
      </header>

      <div className="divide-y divide-hairline">
        {filtered.map((term, i) => {
          const zh = lang === 'zh' ? term.zh : undefined;
          return (
            <section
              key={term.term}
              id={`term-${slugify(term.term)}`}
              className="rise scroll-mt-24 py-7"
              style={{ ['--stagger' as string]: Math.min(i, 8) }}
            >
              <h2 className="font-serif text-[22px] font-semibold tracking-tight">
                {zh ? (
                  <>
                    {zh.term}
                    <span className="ml-2 font-mono text-[12px] font-normal tracking-[0.06em] text-faint">
                      {term.term}
                    </span>
                  </>
                ) : (
                  term.term
                )}
              </h2>
              <dl className="mt-3 space-y-2.5 text-[14.5px] leading-relaxed">
                <Row label={t('g_saying')} value={zh?.saying || term.saying} tone="faint" quoted />
                <Row label={t('g_meaning')} value={zh?.meaning || term.meaning} tone="ink" />
                {(zh?.origin || term.origin) && (
                  <Row label={t('g_origin')} value={zh?.origin || term.origin} tone="faint" />
                )}
              </dl>
            </section>
          );
        })}
        {filtered.length === 0 && (
          <p className="py-16 text-center text-[14px] text-faint">{t('search_empty')}</p>
        )}
      </div>
    </div>
  );
}

function Row({
  label,
  value,
  tone,
  quoted,
}: {
  label: string;
  value: string;
  tone: 'ink' | 'faint';
  quoted?: boolean;
}) {
  return (
    <div className="grid grid-cols-[96px_1fr] gap-3">
      <dt className="pt-0.5 font-mono text-[10.5px] tracking-[0.08em] text-faint">{label}</dt>
      <dd className={tone === 'ink' ? 'text-ink' : 'text-faint'}>
        {quoted ? <span className="font-serif italic">“{value}”</span> : value}
      </dd>
    </div>
  );
}
