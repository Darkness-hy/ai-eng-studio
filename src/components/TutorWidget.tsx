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
import { getMySparkAccount, isInSparkClass, requestSparkAccount, type SparkAccountRow } from '../lib/sparkAccount';

/** A short status line fed to the tutor so it knows whether to start, skip, or
 *  block a Spark-account request. */
function sparkContextLine(inClass: boolean, acc: SparkAccountRow | null): string {
  const account =
    !acc || acc.status === 'revoked'
      ? '尚未申请'
      : acc.status === 'ready'
        ? `已开通(用户名 ${acc.ssh_username ?? acc.requested_username ?? ''})`
        : acc.status === 'failed'
          ? '上次开通失败'
          : '申请处理中';
  return `【Spark 账户信息(供你判断是否要发起"申请spark账号")】是否已加入「Spark 使用班级」:${inClass ? '是' : '否'};该用户的 Spark 账户:${account}。`;
}

async function loadSparkContextLine(userId: string): Promise<string> {
  const [inClass, acc] = await Promise.all([isInSparkClass(userId), getMySparkAccount(userId).catch(() => null)]);
  return sparkContextLine(inClass, acc);
}

const mentionsSparkAccount = (text: string) => /spark|账号|账户|毕业设计|申请/i.test(text);

/** The tutor emits [[spark-apply:<pinyin>]] once it has the learner's pinyin name. */
const SPARK_MARKER = /\[\[spark-apply:([a-z][a-z0-9]{1,31})\]\]/i;
const stripSparkMarker = (t: string) =>
  t
    .replace(/\[\[spark-apply:[^\]]*\]\]/gi, '')
    .replace(/\[\[spark-apply:[^\]]*$/i, '') // incomplete trailing marker while streaming
    .replace(/\n{3,}/g, '\n\n')
    .trimEnd();

const VPN_URL = 'https://itsc.nju.edu.cn/21601/listm.htm';

/** Once the Spark agent has provisioned (or failed), build the tutor's follow-up
 *  message. The temp password is read here from the authed session, never by the LLM. */
function sparkResultMessage(r: SparkAccountRow, zh: boolean): string {
  if (r.status === 'ready') {
    const ssh = `ssh -p ${r.ssh_port ?? 22} ${r.ssh_username}@${r.host}`;
    return zh
      ? `你的 Spark 账户已开通啦 🎉 登录方式:\n\n1. 先登录**南大 VPN**:${VPN_URL}\n2. 再 SSH 登录:\`${ssh}\`\n3. 临时密码:\`${r.temp_password}\`\n\n首次登录会要求改密码:在 **Current password** 处把上面这串临时密码**再原样输一遍**(建议直接复制粘贴、别手敲),然后设置你自己的新密码。临时密码仅用于第一次登录,勿外传~\n\n⚠️ Spark 是 **ARM 架构(aarch64)**,不是 x86——装依赖、编译、拉镜像时都请选 ARM/aarch64 版。`
      : `Your Spark account is ready 🎉\n\n1. Sign in to the **NJU VPN**: ${VPN_URL}\n2. SSH in: \`${ssh}\`\n3. Temp password: \`${r.temp_password}\`\n\nOn first login you must change the password: at the **Current password** prompt re-enter the temp password above **exactly** (copy-paste it, don't retype), then set your own. It's one-time — keep it private.\n\n⚠️ Spark is **ARM (aarch64)**, not x86 — pick ARM/aarch64 builds for deps, compilation, and images.`;
  }
  return zh
    ? `账户开通失败:${r.error || '请联系管理员'}。你可以在「学习进度」页撤回并重试。`
    : `Provisioning failed: ${r.error || 'please contact an admin'}. You can withdraw & retry on the Progress page.`;
}

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
  const [sparkLine, setSparkLine] = useState<string | null>(null); // spark status fed to the tutor

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
  const sparkPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => () => {
    abortRef.current?.abort();
    if (drainRef.current) clearInterval(drainRef.current);
    if (happyTimerRef.current) clearTimeout(happyTimerRef.current);
    if (sparkPollRef.current) clearInterval(sparkPollRef.current);
  }, []);
  useEffect(() => { fetchIndex().then(setIndex).catch(() => {}); }, []);

  // Tell the tutor whether the learner is in the Spark class and already has an
  // account, so it can guide/skip/block "申请spark账号" correctly. Refreshed on open.
  useEffect(() => {
    if (!open || !profile) return;
    let live = true;
    loadSparkContextLine(profile.id)
      .then((line) => {
        if (live) setSparkLine(line);
      })
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [open, profile]);

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

  // After a spark request is created, poll until the agent provisions (or fails)
  // and post the result — including the temp password (read here, not by the LLM).
  const watchSpark = (userId: string) => {
    if (sparkPollRef.current) clearInterval(sparkPollRef.current);
    let tries = 0;
    sparkPollRef.current = setInterval(() => {
      tries += 1;
      getMySparkAccount(userId)
        .then((r) => {
          if (!r) return;
          if (r.status === 'ready' || r.status === 'failed') {
            if (sparkPollRef.current) {
              clearInterval(sparkPollRef.current);
              sparkPollRef.current = null;
            }
            setTurns((prev) => [...prev, { role: 'assistant', content: sparkResultMessage(r, zh) }]);
            isInSparkClass(userId)
              .then((inClass) => setSparkLine(sparkContextLine(inClass, r)))
              .catch(() => {});
            scrollDown();
          }
        })
        .catch(() => {});
      if (tries > 40 && sparkPollRef.current) {
        clearInterval(sparkPollRef.current);
        sparkPollRef.current = null;
      }
    }, 6000);
  };

  // Move the finished in-flight message into the committed list + persist it.
  const commit = () => {
    const full = fullRef.current;
    stopDrain();
    setStreamShown('');
    setBusy(false);
    if (full) {
      // The tutor signals a capstone-account request with a hidden marker
      // [[spark-apply:<pinyin>]] once it has collected the learner's name. The
      // privileged write happens here (authed session); the agent gates on class
      // membership. Strip the marker before showing/persisting the message.
      const m = full.match(SPARK_MARKER);
      if (m && profile) {
        const learner = profile.id;
        void requestSparkAccount(learner, m[1])
          .catch(() => {})
          .finally(() => watchSpark(learner));
      }
      const shown = stripSparkMarker(full);
      setTurns((prev) => [...prev, { role: 'assistant', content: shown }]);
      const p = pendingRef.current;
      if (profile && p) {
        void saveTutorMessages(profile.id, p.lessonId, [
          { role: 'user', content: p.q },
          { role: 'assistant', content: shown },
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
    const base = profile && index ? buildUserProfile(profile, progress, index, lang) : null;
    let currentSparkLine = sparkLine;
    if (profile && (!currentSparkLine || mentionsSparkAccount(q))) {
      try {
        currentSparkLine = await loadSparkContextLine(profile.id);
        setSparkLine(currentSparkLine);
      } catch {
        currentSparkLine = sparkLine;
      }
    }
    const userProfile = [base, currentSparkLine].filter(Boolean).join('\n') || null;
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
            <div className="font-serif text-[15px] font-semibold">{zh ? 'AI 学习助教' : 'AI Tutor'}</div>
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
                ? '我是这门课的 AI 学习助教呀~有什么不懂的都可以问我:讲概念、举例子、帮你规划学习都行。试试看:'
                : "Hi! I'm your AI tutor for this course — ask me anything: concepts, examples, or planning your study. Try:"}
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
              {streamShown ? <ChatMarkdown text={stripSparkMarker(streamShown)} /> : <span className="text-faint">…</span>}
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
