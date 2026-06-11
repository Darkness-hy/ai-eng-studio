import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { fetchGlossary, fetchIndex } from '../lib/data';
import { lessonTitle, phaseTitle, useLang } from '../lib/i18n';
import { slugify } from '../lib/md';
import type { CourseIndex, GlossaryTerm } from '../lib/types';

interface Hit {
  key: string;
  kind: 'lesson' | 'term';
  primary: string;
  secondary: string;
  to: string;
  rank: number;
}

export function CommandPalette({ onClose }: { onClose: () => void }) {
  const { lang, t } = useLang();
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [index, setIndex] = useState<CourseIndex | null>(null);
  const [terms, setTerms] = useState<GlossaryTerm[]>([]);
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchIndex().then(setIndex);
    fetchGlossary().then((g) => setTerms(g.terms));
    const id = setTimeout(() => inputRef.current?.focus(), 30);
    return () => clearTimeout(id);
  }, []);

  const hits = useMemo<Hit[]>(() => {
    if (!index) return [];
    const q = query.trim().toLowerCase();
    const out: Hit[] = [];
    for (const phase of index.phases) {
      for (const lesson of phase.lessons) {
        const zh = lesson.titleZh ?? '';
        const en = lesson.title;
        const hay = `${zh} ${en} ${phase.titleZh} ${phase.titleEn}`.toLowerCase();
        if (q && !hay.includes(q)) continue;
        const rank = q
          ? Number(!zh.toLowerCase().startsWith(q) && !en.toLowerCase().startsWith(q))
          : 1;
        out.push({
          key: `${phase.slug}/${lesson.slug}`,
          kind: 'lesson',
          primary: lessonTitle(lesson, lang),
          secondary: `${t('phase')} ${phase.num} · ${phaseTitle(phase, lang)}`,
          to: `/lesson/${phase.slug}/${lesson.slug}`,
          rank,
        });
      }
    }
    if (q) {
      for (const term of terms) {
        const hay = `${term.term} ${term.zh?.term ?? ''}`.toLowerCase();
        if (!hay.includes(q)) continue;
        out.push({
          key: `term-${term.term}`,
          kind: 'term',
          primary: term.zh && lang === 'zh' ? `${term.zh.term} · ${term.term}` : term.term,
          secondary: t('nav_glossary'),
          to: `/glossary#term-${slugify(term.term)}`,
          rank: 0,
        });
      }
    }
    out.sort((a, b) => a.rank - b.rank);
    return out.slice(0, 12);
  }, [index, terms, query, lang, t]);

  const go = (hit: Hit) => {
    onClose();
    navigate(hit.to);
  };

  return (
    <div
      className="fixed inset-0 z-50 bg-ink/15 backdrop-blur-[1px]"
      onClick={onClose}
      onKeyDown={(e) => {
        if (e.key === 'Escape') onClose();
        if (e.key === 'ArrowDown') {
          e.preventDefault();
          setCursor((c) => Math.min(c + 1, hits.length - 1));
        }
        if (e.key === 'ArrowUp') {
          e.preventDefault();
          setCursor((c) => Math.max(c - 1, 0));
        }
        if (e.key === 'Enter' && hits[cursor]) go(hits[cursor]);
      }}
    >
      <div
        className="mx-auto mt-[12vh] w-[min(620px,92vw)] overflow-hidden rounded-xl border border-hairline bg-paper shadow-[0_16px_48px_rgba(0,0,0,0.08)]"
        onClick={(e) => e.stopPropagation()}
      >
        <input
          ref={inputRef}
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setCursor(0);
          }}
          placeholder={t('search_placeholder')}
          className="w-full border-b border-hairline bg-transparent px-5 py-4 text-[15px] outline-none placeholder:text-faint"
        />
        <ul className="max-h-[50vh] overflow-y-auto py-2">
          {hits.length === 0 && (
            <li className="px-5 py-6 text-center text-[14px] text-faint">{t('search_empty')}</li>
          )}
          {hits.map((hit, i) => (
            <li key={hit.key}>
              <button
                type="button"
                onMouseEnter={() => setCursor(i)}
                onClick={() => go(hit)}
                className={`flex w-full items-baseline justify-between gap-4 px-5 py-2.5 text-left transition-colors ${
                  i === cursor ? 'bg-bone' : ''
                }`}
              >
                <span className="truncate text-[14.5px]">{hit.primary}</span>
                <span className="shrink-0 font-mono text-[11px] text-faint">{hit.secondary}</span>
              </button>
            </li>
          ))}
        </ul>
        <div className="flex gap-4 border-t border-hairline bg-bone/50 px-5 py-2 font-mono text-[10.5px] text-faint">
          <span>↑↓ 选择</span>
          <span>↵ 打开</span>
          <span>esc 关闭</span>
        </div>
      </div>
    </div>
  );
}
