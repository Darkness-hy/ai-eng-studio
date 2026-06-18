import { useEffect, useRef, useState, useSyncExternalStore, type FormEvent } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useAuth } from '../lib/auth';
import { fetchIndex } from '../lib/data';
import { useLang } from '../lib/i18n';
import { useProgress } from '../lib/progress';
import {
  askTutor,
  buildUserProfile,
  saveTutorMessages,
  subscribeTutorContext,
  tutorContextSnapshot,
  type ChatMessage,
} from '../lib/tutor';
import type { CourseIndex } from '../lib/types';
import { TutorAvatar, type AvatarMode } from './TutorAvatar';

interface Turn {
  role: 'user' | 'assistant';
  content: string;
}

/** Compact markdown for chat bubbles — bold, lists, inline/blocks, links. */
function ChatMarkdown({ text }: { text: string }) {
  return (
    <div className="space-y-2 [&_p]:m-0 [&>:last-child]:mb-0">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          strong: ({ children }) => <strong className="font-semibold text-ink">{children}</strong>,
          em: ({ children }) => <em className="italic">{children}</em>,
          ul: ({ children }) => <ul className="list-disc space-y-0.5 pl-5">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal space-y-0.5 pl-5">{children}</ol>,
          a: ({ children, href }) => (
            <a href={href} target="_blank" rel="noreferrer" className="underline decoration-faint underline-offset-2 hover:text-ink">
              {children}
            </a>
          ),
          h1: ({ children }) => <div className="font-serif text-[14.5px] font-semibold">{children}</div>,
          h2: ({ children }) => <div className="font-serif text-[14px] font-semibold">{children}</div>,
          h3: ({ children }) => <div className="font-semibold">{children}</div>,
          code: ({ children }) => (
            <code className="rounded bg-bone px-1 py-0.5 font-mono text-[12px] text-ink">{children}</code>
          ),
          pre: ({ children }) => (
            <pre className="overflow-x-auto rounded-md bg-bone p-2.5 font-mono text-[12px] leading-relaxed [&_code]:bg-transparent [&_code]:p-0">
              {children}
            </pre>
          ),
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

/** Three softly pulsing dots — shows the tutor is thinking before the first token. */
function ThinkingDots() {
  return (
    <span className="inline-flex items-center gap-1 py-1" aria-label="thinking">
      <span className="h-1.5 w-1.5 rounded-full bg-faint animate-pulse" style={{ animationDuration: '1.1s' }} />
      <span className="h-1.5 w-1.5 rounded-full bg-faint animate-pulse" style={{ animationDuration: '1.1s', animationDelay: '0.18s' }} />
      <span className="h-1.5 w-1.5 rounded-full bg-faint animate-pulse" style={{ animationDuration: '1.1s', animationDelay: '0.36s' }} />
    </span>
  );
}

export function TutorWidget() {
  const { lang } = useLang();
  const zh = lang === 'zh';
  const { profile } = useAuth();
  const progress = useProgress();
  const ctx = useSyncExternalStore(subscribeTutorContext, tutorContextSnapshot, tutorContextSnapshot);

  const [index, setIndex] = useState<CourseIndex | null>(null);
  const [open, setOpen] = useState(false);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [streamShown, setStreamShown] = useState('');
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [inputFocused, setInputFocused] = useState(false);
  const [justAnswered, setJustAnswered] = useState(false); // transient "happy" pulse after a reply

  const abortRef = useRef<AbortController | null>(null);
  const happyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const fullRef = useRef(''); // all text received from the network so far
  const shownLenRef = useRef(0); // chars currently revealed (typewriter cursor)
  const doneRef = useRef(false); // network stream finished
  const drainRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pendingRef = useRef<{ q: string; lessonId: string | null } | null>(null);
  const atBottomRef = useRef(true); // is the user pinned to the bottom (stick-to-bottom)?
  const okRef = useRef(false); // did the last request complete without error/abort?

  useEffect(() => () => {
    abortRef.current?.abort();
    if (drainRef.current) clearInterval(drainRef.current);
    if (happyTimerRef.current) clearTimeout(happyTimerRef.current);
  }, []);
  useEffect(() => { fetchIndex().then(setIndex).catch(() => {}); }, []);

  // Stick to the bottom only while the user is already there — so they can scroll up
  // to re-read while the tutor is still streaming without being yanked back down.
  const scrollDown = () => {
    const el = scrollRef.current;
    if (el && atBottomRef.current) el.scrollTop = el.scrollHeight;
  };

  const onScroll = () => {
    const el = scrollRef.current;
    if (el) atBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  };

  const stopDrain = () => {
    if (drainRef.current) {
      clearInterval(drainRef.current);
      drainRef.current = null;
    }
  };

  // Move the finished in-flight message into the committed list + persist it.
  const commit = () => {
    const full = fullRef.current;
    stopDrain();
    setStreamShown('');
    setBusy(false);
    if (full) {
      setTurns((prev) => [...prev, { role: 'assistant', content: full }]);
      const p = pendingRef.current;
      if (profile && p) {
        void saveTutorMessages(profile.id, p.lessonId, [
          { role: 'user', content: p.q },
          { role: 'assistant', content: full },
        ]);
      }
      if (okRef.current) {
        // brief "happy" celebration after a complete answer
        setJustAnswered(true);
        if (happyTimerRef.current) clearTimeout(happyTimerRef.current);
        happyTimerRef.current = setTimeout(() => setJustAnswered(false), 1500);
      }
    }
    pendingRef.current = null;
    scrollDown();
  };

  // Reveal received text at a steady pace so bursty network delivery looks like
  // smooth typing (decouples render cadence from arrival).
  const startDrain = () => {
    stopDrain();
    drainRef.current = setInterval(() => {
      const target = fullRef.current.length;
      if (shownLenRef.current < target) {
        shownLenRef.current = Math.min(target, shownLenRef.current + Math.max(1, Math.min(3, target - shownLenRef.current)));
        setStreamShown(fullRef.current.slice(0, shownLenRef.current));
        scrollDown();
      } else if (doneRef.current) {
        commit();
      }
    }, 30);
  };

  const send = async (text: string) => {
    const q = text.trim();
    if (!q || busy) return;
    setError(null);
    setInput('');
    const history: ChatMessage[] = turns.map((t) => ({ role: t.role, content: t.content }));
    setTurns((prev) => [...prev, { role: 'user', content: q }]);
    fullRef.current = '';
    shownLenRef.current = 0;
    doneRef.current = false;
    pendingRef.current = { q, lessonId: ctx?.lessonId ?? null };
    setStreamShown('');
    setBusy(true);
    setJustAnswered(false);
    okRef.current = false;
    atBottomRef.current = true; // a freshly sent question always scrolls into view
    startDrain();
    const ac = new AbortController();
    abortRef.current = ac;
    const userProfile = profile && index ? buildUserProfile(profile, progress, index, lang) : null;
    try {
      await askTutor(q, history, ctx, userProfile, lang, {
        signal: ac.signal,
        onDelta: (delta) => {
          fullRef.current += delta;
        },
      });
      okRef.current = true;
      doneRef.current = true; // drain finishes revealing, then commits
    } catch (err) {
      doneRef.current = true;
      if (ac.signal.aborted) {
        commit(); // keep whatever streamed so far
      } else {
        if (fullRef.current) {
          commit(); // keep the partial answer that already streamed, then show the error
        } else {
          stopDrain();
          setStreamShown('');
          setBusy(false);
          pendingRef.current = null;
        }
        setError(err instanceof Error ? err.message : String(err));
      }
    }
    abortRef.current = null;
  };

  const stop = () => abortRef.current?.abort();

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void send(input);
  };

  const suggestions = ctx
    ? zh
      ? ['用一句话概括这一课', '这部分能举个例子吗?', '我没看懂,换个说法讲讲']
      : ['Summarize this lesson', 'Give me an example', 'Explain this more simply']
    : zh
      ? ['我该从哪个阶段开始?', '什么是 Transformer?', '帮我规划学习路线']
      : ['Where should I start?', 'What is a Transformer?', 'Plan my learning path'];

  const userBubble =
    'max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-ink px-3.5 py-2 text-[13.5px] leading-relaxed text-white';
  const asstBubble =
    'max-w-[90%] rounded-2xl rounded-bl-sm bg-paper px-3.5 py-2 text-[13.5px] leading-relaxed text-ink';

  // Avatar mood derived purely from the widget's live state (no extra source of truth).
  const mode: AvatarMode = !open
    ? 'sleeping'
    : error
      ? 'confused'
      : busy && streamShown
        ? 'talking'
        : busy
          ? 'thinking'
          : justAnswered
            ? 'happy'
            : inputFocused || input.trim()
              ? 'listening'
              : turns.length === 0
                ? 'greeting'
                : 'idle';

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-label={zh ? '打开 AI 辅导' : 'Open AI tutor'}
        className="fixed bottom-6 right-6 z-50 flex h-14 w-14 items-center justify-center rounded-full bg-canvas shadow-lg ring-1 ring-hairline transition-transform hover:scale-105"
      >
        <TutorAvatar mode="sleeping" size={52} />
      </button>
    );
  }

  return (
    <div className="fixed bottom-6 right-6 z-50 flex h-[540px] w-[min(92vw,400px)] flex-col overflow-hidden rounded-2xl border border-hairline bg-canvas shadow-xl">
      <header className="flex items-center justify-between border-b border-hairline bg-paper px-4 py-3">
        <div className="flex min-w-0 items-center gap-2.5">
          <TutorAvatar mode={mode} size={38} />
          <div className="min-w-0">
            <div className="font-serif text-[15px] font-semibold">{zh ? 'AI 辅导' : 'AI Tutor'}</div>
            <div className="truncate font-mono text-[10.5px] text-faint">
              {ctx ? `${zh ? '正在讨论' : 'on'} · ${ctx.title}` : zh ? '课程助教' : 'course assistant'}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-1">
          {turns.length > 0 && (
            <button
              type="button"
              onClick={() => { setTurns([]); setError(null); }}
              className="rounded px-2 py-1 font-mono text-[10.5px] text-faint hover:text-ink"
            >
              {zh ? '清空' : 'clear'}
            </button>
          )}
          <button
            type="button"
            onClick={() => { abortRef.current?.abort(); setOpen(false); }}
            aria-label={zh ? '收起' : 'Close'}
            className="rounded p-1 text-faint hover:text-ink"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><path d="M6 6l12 12M18 6L6 18" /></svg>
          </button>
        </div>
      </header>

      <div ref={scrollRef} onScroll={onScroll} className="flex-1 space-y-3 overflow-y-auto overscroll-contain px-4 py-4">
        {turns.length === 0 && !busy && (
          <div className="space-y-2">
            <p className="text-[13px] leading-relaxed text-faint">
              {zh
                ? '我是这门课的助教,可以解释概念、举例、帮你规划学习。试试:'
                : "I'm the course assistant — I can explain concepts, give examples, and plan your study. Try:"}
            </p>
            {suggestions.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => void send(s)}
                className="block w-full rounded-lg border border-hairline bg-paper px-3 py-2 text-left text-[13px] text-ink hover:bg-bone"
              >
                {s}
              </button>
            ))}
          </div>
        )}
        {turns.map((t, i) => (
          <div key={i} className={t.role === 'user' ? 'flex justify-end' : 'flex justify-start'}>
            <div className={t.role === 'user' ? userBubble : asstBubble}>
              {t.role === 'user' ? t.content : <ChatMarkdown text={t.content} />}
            </div>
          </div>
        ))}
        {busy && (
          <div className="flex justify-start">
            <div className={asstBubble}>
              {streamShown ? <ChatMarkdown text={streamShown} /> : <ThinkingDots />}
            </div>
          </div>
        )}
        {error && <div className="rounded-lg bg-pale-red px-3 py-2 text-[12.5px] text-ink-red">{error}</div>}
      </div>

      <form onSubmit={onSubmit} className="border-t border-hairline bg-paper px-3 py-3">
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onFocus={() => setInputFocused(true)}
            onBlur={() => setInputFocused(false)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                void send(input);
              }
            }}
            rows={1}
            placeholder={zh ? '问我任何关于这门课的问题…' : 'Ask anything about this course…'}
            className="max-h-28 flex-1 resize-none rounded-lg border border-hairline bg-canvas px-3 py-2 text-[13.5px] outline-none placeholder:text-faint focus:border-faint"
          />
          {busy ? (
            <button type="button" onClick={stop} className="rounded-lg border border-hairline px-3 py-2 text-[12.5px] text-faint hover:text-ink-red">
              {zh ? '停止' : 'stop'}
            </button>
          ) : (
            <button type="submit" disabled={!input.trim()} className="rounded-lg bg-ink px-3.5 py-2 text-[13px] text-white hover:bg-ink/85 disabled:opacity-40">
              {zh ? '发送' : 'send'}
            </button>
          )}
        </div>
      </form>
    </div>
  );
}
