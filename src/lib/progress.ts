import { useSyncExternalStore } from 'react';

export interface LessonProgress {
  done?: boolean;
  preScore?: number;
  preTotal?: number;
  postScore?: number;
  postTotal?: number;
  completedAt?: string;
}

export interface ProgressState {
  v: 1;
  lessons: Record<string, LessonProgress>;
  visits: string[]; // ISO dates (YYYY-MM-DD) with activity, for streaks
  lastLesson?: string; // "phaseSlug/lessonSlug"
}

const KEY = 'aes:progress:v1';
const listeners = new Set<() => void>();

let state: ProgressState = load();

function load(): ProgressState {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as ProgressState;
      if (parsed && parsed.v === 1 && parsed.lessons) return parsed;
    }
  } catch {
    /* corrupted storage falls through to fresh state */
  }
  return { v: 1, lessons: {}, visits: [] };
}

function commit(next: ProgressState) {
  state = next;
  localStorage.setItem(KEY, JSON.stringify(state));
  listeners.forEach((fn) => fn());
}

function today(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

export function getProgress(): ProgressState {
  return state;
}

export function setLessonDone(id: string, done: boolean) {
  const prev = state.lessons[id] ?? {};
  commit({
    ...state,
    lessons: {
      ...state.lessons,
      [id]: { ...prev, done, completedAt: done ? new Date().toISOString() : undefined },
    },
  });
}

export function saveQuizScore(id: string, stage: 'pre' | 'post', score: number, total: number) {
  const prev = state.lessons[id] ?? {};
  const patch =
    stage === 'pre' ? { preScore: score, preTotal: total } : { postScore: score, postTotal: total };
  commit({ ...state, lessons: { ...state.lessons, [id]: { ...prev, ...patch } } });
}

export function recordVisit(lessonId: string) {
  const day = today();
  const visits = state.visits.includes(day) ? state.visits : [...state.visits, day];
  commit({ ...state, visits, lastLesson: lessonId });
}

export function exportProgress(): string {
  return JSON.stringify(state, null, 2);
}

export function importProgress(json: string): boolean {
  try {
    const parsed = JSON.parse(json) as ProgressState;
    if (parsed?.v !== 1 || typeof parsed.lessons !== 'object') return false;
    commit({ v: 1, lessons: parsed.lessons, visits: parsed.visits ?? [], lastLesson: parsed.lastLesson });
    return true;
  } catch {
    return false;
  }
}

export function streakDays(): number {
  const days = new Set(state.visits);
  let count = 0;
  const cursor = new Date();
  // Today counts if visited; otherwise the streak may still be alive from yesterday.
  if (!days.has(today())) cursor.setDate(cursor.getDate() - 1);
  for (;;) {
    const key = `${cursor.getFullYear()}-${String(cursor.getMonth() + 1).padStart(2, '0')}-${String(cursor.getDate()).padStart(2, '0')}`;
    if (!days.has(key)) break;
    count += 1;
    cursor.setDate(cursor.getDate() - 1);
  }
  return count;
}

export function useProgress(): ProgressState {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => state,
  );
}
