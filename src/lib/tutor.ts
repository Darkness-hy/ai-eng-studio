import type { Lang, Lesson } from './types';

const endpoint = import.meta.env.VITE_AI_TUTOR_ENDPOINT as string | undefined;
const token = import.meta.env.VITE_AI_TUTOR_TOKEN as string | undefined;

/** AI 辅导是可选的:没有配置端点时整块功能保持休眠(浮窗不出现)。 */
export const tutorEnabled = Boolean(endpoint);

export interface TutorContext {
  lessonId: string;
  title: string;
  /** 注入给服务端的精简课文(前端侧轻量 RAG)。 */
  text: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

// ── 当前页面的 RAG 上下文(课文)。课程页登记,浮窗读取。 ──────────────
let context: TutorContext | null = null;
const listeners = new Set<() => void>();

export function setTutorContext(ctx: TutorContext | null): void {
  context = ctx;
  for (const l of listeners) l();
}
export function subscribeTutorContext(cb: () => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}
export function tutorContextSnapshot(): TutorContext | null {
  return context;
}

/** 从课程对象生成精简上下文(标题 + 正文截断到 ~4000 字)。 */
export function lessonContext(lesson: Lesson, lang: Lang): TutorContext {
  const title = (lang === 'zh' ? lesson.titleZh : null) ?? lesson.title;
  const body = (lang === 'zh' ? lesson.bodyZh : null) ?? lesson.bodyEn;
  return { lessonId: lesson.id, title, text: `# ${title}\n\n${body}`.slice(0, 4000) };
}

export interface AskOptions {
  signal?: AbortSignal;
  onDelta: (text: string) => void;
}

/**
 * 向用户自建的辅导服务端发起一次流式问答。
 * 协议见 docs/ai-tutor-server-contract.md:POST JSON,返回 SSE(data: {type,text})。
 */
export async function askTutor(
  message: string,
  history: ChatMessage[],
  ctx: TutorContext | null,
  lang: Lang,
  opts: AskOptions,
): Promise<void> {
  if (!endpoint) throw new Error('AI 辅导未配置(缺少 VITE_AI_TUTOR_ENDPOINT)');
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers.Authorization = `Bearer ${token}`;

  const res = await fetch(endpoint, {
    method: 'POST',
    headers,
    signal: opts.signal,
    body: JSON.stringify({
      message,
      history,
      lesson_id: ctx?.lessonId ?? null,
      context: ctx?.text ?? null,
      lang,
      stream: true,
    }),
  });
  if (res.status === 429 || res.status === 503) throw new Error('助教正忙(同时提问的人较多),请稍后再试');
  if (!res.ok || !res.body) throw new Error(`辅导服务返回错误 (${res.status})`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const frames = buf.split('\n\n');
    buf = frames.pop() ?? '';
    for (const frame of frames) {
      const dataLine = frame.split('\n').find((l) => l.startsWith('data:'));
      if (!dataLine) continue;
      const json = dataLine.slice(5).trim();
      if (!json) continue;
      let ev: { type?: string; text?: string; message?: string };
      try {
        ev = JSON.parse(json);
      } catch {
        continue; // 忽略心跳/空行
      }
      if (ev.type === 'delta' && typeof ev.text === 'string') opts.onDelta(ev.text);
      else if (ev.type === 'error') throw new Error(ev.message || '辅导服务出错');
    }
  }
}
