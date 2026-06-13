import { useEffect, useState } from 'react';
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom';
import { useAuth } from '../lib/auth';
import { useLang } from '../lib/i18n';
import { dueCount, useReview } from '../lib/review';
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
  const { enabled: authEnabled, profile, signOut } = useAuth();
  const [paletteOpen, setPaletteOpen] = useState(false);
  useReview(); // re-render the due badge when the review queue changes
  const due = dueCount();

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
            <NavLink to="/review" className={navCls}>
              <span className="inline-flex items-center gap-1.5">
                {lang === 'zh' ? '复习' : 'Review'}
                {due > 0 && (
                  <span className="rounded-full bg-pale-red px-1.5 py-0.5 font-mono text-[10px] leading-none text-ink-red">
                    {due}
                  </span>
                )}
              </span>
            </NavLink>
            {authEnabled && profile && (
              <NavLink to="/classes" className={navCls}>
                {lang === 'zh' ? '班级' : 'Classes'}
              </NavLink>
            )}
            {authEnabled && profile?.role === 'admin' && (
              <NavLink to="/admin" className={navCls}>
                {lang === 'zh' ? '管理后台' : 'Admin'}
              </NavLink>
            )}
            {authEnabled &&
              (profile ? (
                <span className="flex items-center gap-2">
                  <span
                    className="hidden max-w-[120px] truncate rounded-full bg-pale-green px-2.5 py-0.5 text-[11.5px] text-ink-green sm:inline"
                    title={profile.email}
                  >
                    {profile.display_name ?? profile.email}
                  </span>
                  <button
                    type="button"
                    onClick={() => void signOut()}
                    className="whitespace-nowrap text-[12.5px] text-faint transition-colors hover:text-ink"
                  >
                    {lang === 'zh' ? '退出' : 'Sign out'}
                  </button>
                </span>
              ) : (
                <NavLink to="/login" className={navCls}>
                  {lang === 'zh' ? '登录' : 'Sign in'}
                </NavLink>
              ))}
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
        <div className="bg-bone/40">
          <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-8 gap-y-4 px-5 py-6">
            <div className="flex items-center gap-4">
              <img
                src={`${import.meta.env.BASE_URL}brand/lavira-logo.png`}
                alt="LaViRA logo"
                className="h-9 w-9 rounded-lg"
              />
              <div>
                <div className="font-mono text-[13px] font-medium tracking-[0.18em] text-faint">
                  {lang === 'zh' ? '独家冠名' : 'PRESENTED BY'}
                </div>
                <div className="font-brand mt-0.5 text-[16px] leading-snug">
                  <span className="font-semibold text-ink">LaViRA</span>
                  <span className="text-faint">: Language-Vision-Robot Actions Translation</span>
                </div>
              </div>
            </div>
            <div className="ml-auto flex items-center justify-end gap-3 text-[13.5px]">
              <span className="hidden items-center gap-2 sm:flex">
                <span className="text-right text-[12.5px] leading-tight text-faint">
                  {lang === 'zh' ? (
                    <>
                      觉得有用的话，
                      <br />
                      给我们点个 Star 吧
                    </>
                  ) : (
                    <>
                      Find it useful?
                      <br />
                      Give us a Star
                    </>
                  )}
                </span>
                <span className="text-[22px] leading-none">👉</span>
              </span>
              <a
                href="https://github.com/NJU-R-L-Group-Embodied-Lab/lavira-code"
                target="_blank"
                rel="noreferrer"
                className="font-brand rounded-md border border-hairline bg-paper px-3 py-1.5 text-faint transition-colors hover:bg-bone hover:text-ink"
              >
                LaViRA ★
              </a>
              <a
                href="https://github.com/NJU-R-L-Group-Embodied-Lab/uni-lavira-code"
                target="_blank"
                rel="noreferrer"
                className="font-brand rounded-md border border-hairline bg-paper px-3 py-1.5 text-faint transition-colors hover:bg-bone hover:text-ink"
              >
                Uni-LaViRA ★
              </a>
            </div>
          </div>
        </div>
        <div className="border-t border-hairline">
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
        </div>
      </footer>

      {paletteOpen && <CommandPalette onClose={() => setPaletteOpen(false)} />}
    </div>
  );
}
