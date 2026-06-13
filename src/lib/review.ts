import { useSyncExternalStore } from 'react';

/**
 * Spaced-repetition review of missed quiz questions (Leitner system).
 * A question answered wrong in a post-quiz enters box 0; answering it right in
 * review promotes it (longer interval), wrong demotes it to box 0.
 * Local-first (localStorage); items reference the lesson + post-quiz index so
 * the review page can pull the bilingual question text on demand.
 */

const INTERVALS_DAYS = [1, 2, 4, 8, 16]; // box 0..4

export interface ReviewItem {
  id: string; // `${lessonId}#${postIdx}`
  lessonId: string;
  postIdx: number;
  box: number; // 0..4
  due: string; // YYYY-MM-DD
  addedAt: string;
}

interface ReviewState {
  v: 1;
  items: Record<string, ReviewItem>;
}

const KEY = 'aes:review:v1';
const listeners = new Set<() => void>();

function load(): ReviewState {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as ReviewState;
      if (parsed?.v === 1 && parsed.items) return parsed;
    }
  } catch {
    /* corrupt storage → fresh */
  }
  return { v: 1, items: {} };
}

let state = load();

function commit(next: ReviewState) {
  state = next;
  localStorage.setItem(KEY, JSON.stringify(state));
  listeners.forEach((fn) => fn());
}

function today(): Date {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  return d;
}

function dayKey(d: Date): string {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

function plusDays(days: number): string {
  const d = today();
  d.setDate(d.getDate() + days);
  return dayKey(d);
}

/** Add a missed question. No-op if it's already queued (keeps its SRS state). */
export function addReviewItem(lessonId: string, postIdx: number) {
  const id = `${lessonId}#${postIdx}`;
  if (state.items[id]) return;
  const now = new Date().toISOString();
  commit({
    ...state,
    // Due immediately so a missed question is reviewable the same day; once
    // reviewed, the SRS schedule (gradeReview) pushes it out.
    items: { ...state.items, [id]: { id, lessonId, postIdx, box: 0, due: dayKey(today()), addedAt: now } },
  });
}

/** Grade a review answer: promote on correct, reset to box 0 on wrong. */
export function gradeReview(id: string, correct: boolean) {
  const item = state.items[id];
  if (!item) return;
  const box = correct ? Math.min(item.box + 1, INTERVALS_DAYS.length - 1) : 0;
  commit({ ...state, items: { ...state.items, [id]: { ...item, box, due: plusDays(INTERVALS_DAYS[box]) } } });
}

export function getReviewState(): ReviewState {
  return state;
}

export function dueItems(): ReviewItem[] {
  const t = dayKey(today());
  return Object.values(state.items)
    .filter((it) => it.due <= t)
    .sort((a, b) => a.due.localeCompare(b.due));
}

export function dueCount(): number {
  return dueItems().length;
}

export function useReview(): ReviewState {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => state,
  );
}
