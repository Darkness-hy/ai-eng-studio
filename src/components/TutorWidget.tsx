import { useRef, useState, useSyncExternalStore, type FormEvent } from 'react';
import { useLang } from '../lib/i18n';
import {
  askTutor,
  subscribeTutorContext,
  tutorContextSnapshot,
  type ChatMessage,
} from '../lib/tutor';

interface Turn {
  role: 'user' | 'assistant';
  content: string;
}

export function TutorWidget() {
  const { lang } = useLang();
  const zh = lang === 'zh';
  const ctx = useSyncExternalStore(subscribeTutorContext, tutorContextSnapshot, tutorContextSnapshot);

  const [open, setOpen] = useState(false);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const scrollDown = () => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  };

  const send = async (text: string) => {
    const q = text.trim();
    if (!q || busy) return;
    setError(null);
    setInput('');
    const history: ChatMessage[] = turns.map((t) => ({ role: t.role, content: t.content }));
    setTurns((prev) => [...prev, { role: 'user', content: q }, { role: 'assistant', content: '' }]);
    setBusy(true);
    const ac = new AbortController();
    abortRef.current = ac;
    try {
      await askTutor(q, history, ctx, lang, {
        signal: ac.signal,
        onDelta: (delta) => {
          setTurns((prev) => {
            const next = prev.slice();
            const last = next[next.length - 1];
            if (last && last.role === 'assistant') next[next.length - 1] = { ...last, content: last.content + delta };
            return next;
          });
          scrollDown();
        },
      });
    } catch (err) {
      if (!ac.signal.aborted) {
        setError(err instanceof Error ? err.message : String(err));
        // drop the empty assistant placeholder on hard failure
        setTurns((prev) => (prev[prev.length - 1]?.content === '' ? prev.slice(0, -1) : prev));
      }
    }
    setBusy(false);
    abortRef.current = null;
    scrollDown();
  };

  const stop = () => {
    abortRef.current?.abort();
    setBusy(false);
  };

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

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-label={zh ? '打开 AI 辅导' : 'Open AI tutor'}
        className="fixed bottom-6 right-6 z-50 flex h-12 w-12 items-center justify-center rounded-full bg-ink text-white shadow-lg transition-transform hover:scale-105"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        </svg>
      </button>
    );
  }

  return (
    <div className="fixed bottom-6 right-6 z-50 flex h-[540px] w-[min(92vw,400px)] flex-col overflow-hidden rounded-2xl border border-hairline bg-canvas shadow-xl">
      <header className="flex items-center justify-between border-b border-hairline bg-paper px-4 py-3">
        <div className="min-w-0">
          <div className="font-serif text-[15px] font-semibold">{zh ? 'AI 辅导' : 'AI Tutor'}</div>
          <div className="truncate font-mono text-[10.5px] text-faint">
            {ctx ? `${zh ? '正在讨论' : 'on'} · ${ctx.title}` : zh ? '课程助教' : 'course assistant'}
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
            onClick={() => setOpen(false)}
            aria-label={zh ? '收起' : 'Close'}
            className="rounded p-1 text-faint hover:text-ink"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><path d="M6 6l12 12M18 6L6 18" /></svg>
          </button>
        </div>
      </header>

      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
        {turns.length === 0 && (
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
            <div
              className={
                t.role === 'user'
                  ? 'max-w-[85%] rounded-2xl rounded-br-sm bg-ink px-3.5 py-2 text-[13.5px] leading-relaxed text-white'
                  : 'max-w-[90%] whitespace-pre-wrap rounded-2xl rounded-bl-sm bg-paper px-3.5 py-2 text-[13.5px] leading-relaxed text-ink'
              }
            >
              {t.content || (busy && i === turns.length - 1 ? <span className="text-faint">…</span> : '')}
            </div>
          </div>
        ))}
        {error && <div className="rounded-lg bg-pale-red px-3 py-2 text-[12.5px] text-ink-red">{error}</div>}
      </div>

      <form onSubmit={onSubmit} className="border-t border-hairline bg-paper px-3 py-3">
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
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
