import { computeBadges } from './achievements';
import { lessonTitle } from './i18n';
import { AREAS, loadPlacement } from './placement';
import { streakDays, type ProgressState } from './progress';
import { cloudEnabled, getSupabase, type ProfileRow } from './supabase';
import type { CourseIndex, Lang, Lesson } from './types';

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

/**
 * 把「当前页面对应的整篇课文」作为 RAG 知识库注入:标题 + 完整正文 +
 * 本课测验的关键问答(浓缩的知识点)。截断到一个宽松上限,覆盖绝大多数整课。
 */
export function lessonContext(lesson: Lesson, lang: Lang): TutorContext {
  const zh = lang === 'zh';
  const title = (zh ? lesson.titleZh : null) ?? lesson.title;
  const body = (zh ? lesson.bodyZh : null) ?? lesson.bodyEn;
  const quiz = (zh ? lesson.quizZh : null) ?? lesson.quizEn ?? [];
  const parts = [`# ${title}`, '', body];
  if (quiz.length) {
    parts.push('', zh ? '## 本课测验要点' : '## Key quiz points');
    for (const q of quiz) {
      const ans = q.options[q.correct] ?? '';
      parts.push(`- ${q.question} → ${ans}${q.explanation ? ` —— ${q.explanation}` : ''}`);
    }
  }
  return { lessonId: lesson.id, title, text: parts.join('\n').slice(0, 12000) };
}

const SOLID_AREA = 8; // placement area mastery threshold (0–10)

function lessonTitleById(id: string, index: CourseIndex, lang: Lang): string | null {
  const [ps, ls] = id.split('/');
  const le = index.phases.find((p) => p.slug === ps)?.lessons.find((l) => l.slug === ls);
  return le ? lessonTitle(le, lang) : null;
}

/**
 * 「学习者画像」:把这位用户的整体学习信息汇总成精简文本,随每次提问注入,
 * 让助教个性化(称呼、难度、进度建议)。刻意排除 PII(不含完整邮箱/ID/角色/逐题作答)。
 */
export function buildUserProfile(
  profile: ProfileRow | null,
  progress: ProgressState,
  index: CourseIndex,
  lang: Lang,
): string {
  const zh = lang === 'zh';
  const none = zh ? '暂无' : 'none';
  const nick = profile?.display_name || profile?.email?.split('@')[0] || (zh ? '同学' : 'learner');

  // 进度:按 catalog 迭代(避免被移除的 capstone 残留 key 虚高总数)
  const totalLessons = index.stats.lessons;
  let doneCount = 0;
  for (const p of index.phases)
    doneCount += p.lessons.filter((l) => progress.lessons[`${p.slug}/${l.slug}`]?.done).length;
  const donePct = totalLessons > 0 ? Math.round((doneCount / totalLessons) * 100) : 0;

  // 测验均分(仅首测 post 分)
  const post = Object.values(progress.lessons).filter((l) => l.postTotal);
  const quizAvg = post.length
    ? `${Math.round((post.reduce((a, l) => a + (l.postScore ?? 0) / (l.postTotal ?? 1), 0) / post.length) * 100)}`
    : none;

  // 当前课
  const current = (progress.lastLesson && lessonTitleById(progress.lastLesson, index, lang)) || none;

  // 薄弱领域 + 入学等级(定级)
  const pl = loadPlacement();
  const entry = pl ? (zh ? `第 ${pl.entry} 阶段(定级 ${pl.total}/50)` : `phase ${pl.entry} (placed ${pl.total}/50)`) : (zh ? '未定级' : 'not placed');
  let weakAreas = zh ? '未定级' : 'not placed';
  if (pl) {
    const weak = AREAS.filter((a) => (pl.areaScores[a.key] ?? 0) < SOLID_AREA).map(
      (a) => `${zh ? a.zh : a.en}(${pl.areaScores[a.key] ?? 0}/10)`,
    );
    weakAreas = weak.length ? weak.join('、') : (zh ? '无明显薄弱' : 'none');
  }

  // 近期低分课程(post < 60%,取前 3)
  const weakLessons: string[] = [];
  for (const [id, l] of Object.entries(progress.lessons)) {
    if (weakLessons.length >= 3) break;
    if (l.postTotal && (l.postScore ?? 0) / l.postTotal < 0.6) {
      const t = lessonTitleById(id, index, lang);
      if (t) weakLessons.push(t);
    }
  }

  // 徽章
  const got = computeBadges(progress, index).filter((b) => b.unlocked).map((b) => (zh ? b.zh : b.en));
  const badges = got.length ? got.slice(0, 5).join('、') + (got.length > 5 ? (zh ? ' 等' : ' …') : '') : none;

  return [
    `${zh ? '昵称' : 'name'}:${nick}｜${zh ? '入学等级' : 'level'}:${entry}`,
    `${zh ? '进度' : 'progress'}:${doneCount}/${totalLessons}(${donePct}%)｜${zh ? '测验平均' : 'quiz avg'}:${quizAvg}`,
    `${zh ? '当前学习' : 'current'}:${current}`,
    `${zh ? '连续学习' : 'streak'} ${streakDays()} ${zh ? '天｜累计活跃' : 'd｜active'} ${progress.visits.length} ${zh ? '天' : 'd'}`,
    `${zh ? '薄弱领域' : 'weak areas'}:${weakAreas}`,
    `${zh ? '近期低分课程' : 'low-scoring'}:${weakLessons.length ? weakLessons.join('、') : none}`,
    `${zh ? '已获徽章' : 'badges'}:${badges}`,
  ].join('\n');
}

/** 把一轮问答(用户问 + 助教答)存到 Supabase,按用户隔离。best-effort,失败不打扰用户。 */
export async function saveTutorMessages(
  userId: string,
  lessonId: string | null,
  turns: { role: 'user' | 'assistant'; content: string }[],
): Promise<void> {
  if (!cloudEnabled) return;
  try {
    await getSupabase()
      .from('tutor_messages')
      .insert(turns.map((t) => ({ user_id: userId, lesson_id: lessonId, role: t.role, content: t.content })));
  } catch {
    /* 存储是附带功能,失败静默 */
  }
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
  userProfile: string | null,
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
      user_profile: userProfile,
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
      let ev: { type?: string; text?: string; message?: string; truncated?: boolean };
      try {
        ev = JSON.parse(json);
      } catch {
        continue; // 忽略心跳/空行
      }
      if (ev.type === 'delta' && typeof ev.text === 'string') opts.onDelta(ev.text);
      else if (ev.type === 'error') throw new Error(ev.message || '辅导服务出错');
      else if (ev.type === 'done' && ev.truncated) opts.onDelta('\n\n_(回答可能被中断,请重试)_');
    }
  }
}
