import { cloudEnabled, getSupabase } from './supabase';
import type { Lang } from './types';

export type LeaderboardId = 'progress' | 'placement' | 'unit_quiz';

export interface LeaderboardEntry {
  position: 1 | 2 | 3;
  displayName: string;
  score: number;
}

export interface LeaderboardMine {
  eligible: boolean;
  ready: boolean;
  score: number | null;
  attempts: number | null;
  percentile: number | null;
  onPodium: boolean;
  gap: number | null;
}

export interface LeaderboardBoard {
  id: LeaderboardId;
  top: LeaderboardEntry[];
  mine: LeaderboardMine;
}

export interface LeaderboardSummary {
  lessonTotal: number;
  boards: LeaderboardBoard[];
}

export const MIN_UNIT_QUIZ_ATTEMPTS = 10;

const BOARD_IDS = new Set<LeaderboardId>(['progress', 'placement', 'unit_quiz']);

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function boolOrFalse(value: unknown): boolean {
  return value === true;
}

function parseEntry(value: unknown): LeaderboardEntry | null {
  const row = asRecord(value);
  if (!row) return null;
  const position = numberOrNull(row.position);
  const score = numberOrNull(row.score);
  if (position !== 1 && position !== 2 && position !== 3) return null;
  if (score == null) return null;
  return {
    position,
    displayName: typeof row.displayName === 'string' && row.displayName.trim() ? row.displayName : '未命名学习者',
    score,
  };
}

function parseMine(value: unknown): LeaderboardMine {
  const row = asRecord(value);
  return {
    eligible: boolOrFalse(row?.eligible),
    ready: boolOrFalse(row?.ready),
    score: numberOrNull(row?.score),
    attempts: numberOrNull(row?.attempts),
    percentile: numberOrNull(row?.percentile),
    onPodium: boolOrFalse(row?.onPodium),
    gap: numberOrNull(row?.gap),
  };
}

function parseBoard(value: unknown): LeaderboardBoard | null {
  const row = asRecord(value);
  if (!row) return null;
  const id = row?.id;
  if (id !== 'progress' && id !== 'placement' && id !== 'unit_quiz') return null;
  const topRaw = Array.isArray(row.top) ? row.top : [];
  return {
    id,
    top: topRaw.map(parseEntry).filter((entry): entry is LeaderboardEntry => entry != null),
    mine: parseMine(row.mine),
  };
}

function parseSummary(value: unknown): LeaderboardSummary {
  const row = asRecord(value);
  const boardsRaw = Array.isArray(row?.boards) ? row.boards : [];
  const boards = boardsRaw.map(parseBoard).filter((board): board is LeaderboardBoard => board != null);
  return {
    lessonTotal: numberOrNull(row?.lessonTotal) ?? 0,
    boards: boards.filter((board) => BOARD_IDS.has(board.id)),
  };
}

export async function fetchLeaderboardSummary(lessonIds: string[]): Promise<LeaderboardSummary> {
  if (!cloudEnabled) throw new Error('Cloud sync is not configured.');
  const { data, error } = await getSupabase().rpc('get_leaderboard_summary', {
    lesson_ids: lessonIds,
  });
  if (error) throw error;
  return parseSummary(data);
}

export function leaderboardTitle(id: LeaderboardId, lang: Lang): string {
  if (lang === 'zh') {
    if (id === 'progress') return '学习进度榜';
    if (id === 'placement') return '定级测试榜';
    return '单元测试榜';
  }
  if (id === 'progress') return 'Progress';
  if (id === 'placement') return 'Placement';
  return 'Unit tests';
}

export function leaderboardSubtitle(id: LeaderboardId, lang: Lang): string {
  if (lang === 'zh') {
    if (id === 'progress') return '按当前课程目录内的完成课数排序';
    if (id === 'placement') return '按当前定级测试总分排序';
    return `按首次课后测验平均分排序，至少完成 ${MIN_UNIT_QUIZ_ATTEMPTS} 个`;
  }
  if (id === 'progress') return 'Ranked by completed lessons in the current catalog';
  if (id === 'placement') return 'Ranked by current placement score';
  return `Ranked by first-attempt post-quiz average, minimum ${MIN_UNIT_QUIZ_ATTEMPTS}`;
}

export function formatLeaderboardScore(id: LeaderboardId, score: number | null, lessonTotal: number, lang: Lang): string {
  if (score == null) return lang === 'zh' ? '暂无' : 'None yet';
  if (id === 'progress') return `${Math.round(score)}/${lessonTotal}`;
  if (id === 'placement') return `${Math.round(score)}/50`;
  return `${Math.round(score * 10) / 10}%`;
}

export function percentileLabel(mine: LeaderboardMine, lang: Lang): string {
  if (!mine.eligible) return lang === 'zh' ? '尚未参与' : 'Not ranked';
  if (!mine.ready || mine.percentile == null) return lang === 'zh' ? '数据积累中' : 'Collecting data';
  return lang === 'zh' ? `约前 ${mine.percentile}%` : `Top ${mine.percentile}%`;
}

export function statusLabel(board: LeaderboardBoard, lang: Lang): string {
  const mine = board.mine;
  if (!mine.ready) return lang === 'zh' ? '榜单数据积累中' : 'Leaderboard is warming up';
  if (mine.onPodium) return lang === 'zh' ? '你当前在前三名' : 'You are currently on the podium';
  if (!mine.eligible) {
    if (board.id === 'placement') return lang === 'zh' ? '完成定级测试后参与榜单' : 'Take placement to enter';
    if (board.id === 'unit_quiz') {
      const done = mine.attempts ?? 0;
      const gap = Math.max(MIN_UNIT_QUIZ_ATTEMPTS - done, 0);
      return lang === 'zh'
        ? `再完成 ${gap} 个课后测验后参与榜单`
        : `${gap} more post-quizzes to enter`;
    }
    return lang === 'zh' ? '开始学习后参与榜单' : 'Start learning to enter';
  }
  if (mine.gap == null) return lang === 'zh' ? '继续保持' : 'Keep going';
  if (mine.gap <= 0) return lang === 'zh' ? '已达到前三名分数' : 'You have reached the podium score';
  if (board.id === 'progress') {
    return lang === 'zh'
      ? `距离上榜还差 ${Math.ceil(mine.gap)} 节课`
      : `${Math.ceil(mine.gap)} lessons from the podium`;
  }
  if (board.id === 'placement') {
    return lang === 'zh'
      ? `距离上榜还差 ${Math.ceil(mine.gap)} 分`
      : `${Math.ceil(mine.gap)} points from the podium`;
  }
  return lang === 'zh'
    ? `平均分还差 ${Math.round(mine.gap * 10) / 10} 个百分点`
    : `${Math.round(mine.gap * 10) / 10} percentage points from the podium`;
}
