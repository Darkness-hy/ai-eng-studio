import { useEffect, useState } from 'react';
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom';
import { useLang } from '../lib/i18n';
import { CommandPalette } from './CommandPalette';

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'instant' as ScrollBehavior });
  }, [pathname]);
  return null;
}

export function Layout() {
  const { lang, setLang, t } = useLang();
  const [paletteOpen, setPaletteOpen] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setPaletteOpen((v) => !v);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const navCls = ({ isActive }: { isActive: boolean }) =>
    `whitespace-nowrap text-[13px] transition-colors sm:text-[14px] ${isActive ? 'text-ink' : 'text-faint hover:text-ink'}`;

  return (
    <div className="min-h-dvh">
      <ScrollToTop />
      <header className="sticky top-0 z-40 border-b border-hairline bg-canvas/85 backdrop-blur-sm">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-5">
          <Link to="/" className="group flex min-w-0 items-baseline gap-3">
            <span className="truncate font-serif text-[16px] font-semibold tracking-tight sm:text-[19px]">
              {lang === 'zh' ? '从零开始的 AI 工程' : 'AI Engineering from Scratch'}
            </span>
            <span className="hidden font-mono text-[10px] tracking-[0.18em] text-faint sm:inline">
              STUDIO
            </span>
          </Link>
          <nav className="flex shrink-0 items-center gap-3 sm:gap-5">
            <NavLink to="/" className={navCls} end>
              {t('nav_map')}
            </NavLink>
            <NavLink to="/glossary" className={navCls}>
              {t('nav_glossary')}
            </NavLink>
            <NavLink to="/progress" className={navCls}>
              {t('nav_progress')}
            </NavLink>
            <button
              type="button"
              onClick={() => setPaletteOpen(true)}
              className="hidden items-center gap-2 rounded-md border border-hairline bg-paper px-2.5 py-1 text-[13px] text-faint transition-colors hover:bg-bone md:flex"
            >
              {t('search')}
              <kbd>⌘K</kbd>
            </button>
            <button
              type="button"
              onClick={() => setLang(lang === 'zh' ? 'en' : 'zh')}
              className="rounded-md border border-hairline bg-paper px-2 py-1 font-mono text-[11px] text-faint transition-colors hover:bg-bone hover:text-ink"
              title={lang === 'zh' ? 'Switch to English' : '切换到中文'}
            >
              {lang === 'zh' ? 'EN' : '中'}
            </button>
          </nav>
        </div>
      </header>

      <main>
        <Outlet />
      </main>

      <footer className="mt-24 border-t border-hairline">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-5 py-8 text-[12.5px] text-faint">
          <span>
            {lang === 'zh' ? '课程内容来自开源项目 ' : 'Course content from '}
            <a
              href="https://github.com/rohitg00/ai-engineering-from-scratch"
              target="_blank"
              rel="noreferrer"
              className="underline decoration-hairline underline-offset-2 hover:text-ink"
            >
              ai-engineering-from-scratch
            </a>
            {' · MIT License'}
          </span>
          <span className="font-mono text-[11px] tracking-[0.12em]">
            FROM SCRATCH · 20 PHASES · BUILT BY HAND
          </span>
        </div>
        <div className="border-t border-hairline bg-bone/40">
          <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-8 gap-y-4 px-5 py-6">
            <div className="flex items-center gap-4">
              {/* LaViRA mark, redrawn flat for the light canvas (no dark plate) */}
              <svg width="34" height="34" viewBox="0 0 512 512" aria-label="LaViRA logo" role="img">
                <rect x="96" y="176" width="224" height="36" rx="18" fill="#2f3437" />
                <rect x="96" y="240" width="156" height="36" rx="18" fill="#2f3437" />
                <rect x="96" y="304" width="92" height="36" rx="18" fill="#2f3437" />
                <circle cx="388" cy="322" r="26" fill="#f2335d" />
              </svg>
              <div>
                <div className="text-[15px] leading-snug">
                  <span className="font-mono text-[10.5px] tracking-[0.16em] text-faint">
                    {lang === 'zh' ? '独家冠名' : 'PRESENTED BY'}
                  </span>{' '}
                  <span className="ml-1 font-semibold text-ink">LaViRA</span>
                  <span className="text-faint">: Language-Vision-Robot Actions Translation</span>
                </div>
                <div className="mt-0.5 font-serif text-[13.5px] italic text-faint">
                  Like A Very Intelligent Real Assistant
                </div>
              </div>
            </div>
            <div className="flex items-center gap-4 text-[13px]">
              <a
                href="https://robo-lavira.github.io/lavira-zs-vln/"
                target="_blank"
                rel="noreferrer"
                className="rounded-md border border-hairline bg-paper px-3 py-1.5 text-faint transition-colors hover:bg-bone hover:text-ink"
              >
                LaViRA ZS-VLN ↗
              </a>
              <a
                href="https://xetroubadour.github.io/Uni-LaViRA/"
                target="_blank"
                rel="noreferrer"
                className="rounded-md border border-hairline bg-paper px-3 py-1.5 text-faint transition-colors hover:bg-bone hover:text-ink"
              >
                Uni-LaViRA ↗
              </a>
            </div>
          </div>
        </div>
      </footer>

      {paletteOpen && <CommandPalette onClose={() => setPaletteOpen(false)} />}
    </div>
  );
}
