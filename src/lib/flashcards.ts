import { useSyncExternalStore } from 'react';

/**
 * Spaced-repetition flashcards over the glossary terms (Leitner system),
 * reusing the same box/interval scheme as the quiz review queue. New terms are
 * introduced gradually; each session mixes due cards with a few new ones.
 */

const INTERVALS_DAYS = [1, 2, 4, 8, 16];
const NEW_PER_SESSION = 10;
const KEY = 'aes:flashcards:v1';
const listeners = new Set<() => void>();

interface CardState {
  box: number; // 0..4
  due: string; // YYYY-MM-DD
}
interface FlashState {
  v: 1;
  cards: Record<string, CardState>; // keyed by term name
}

function load(): FlashState {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const p = JSON.parse(raw) as FlashState;
      if (p?.v === 1 && p.cards) return p;
    }
  } catch {
    /* fresh */
  }
  return { v: 1, cards: {} };
}

let state = load();

function commit(next: FlashState) {
  state = next;
  localStorage.setItem(KEY, JSON.stringify(state));
  listeners.forEach((fn) => fn());
}

function dayKey(d: Date): string {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}
function plusDays(days: number): string {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  d.setDate(d.getDate() + days);
  return dayKey(d);
}
function today(): string {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  return dayKey(d);
}

export function gradeCard(term: string, remembered: boolean) {
  const prev = state.cards[term];
  const box = remembered ? Math.min((prev?.box ?? 0) + 1, INTERVALS_DAYS.length - 1) : 0;
  commit({ ...state, cards: { ...state.cards, [term]: { box, due: plusDays(INTERVALS_DAYS[box]) } } });
}

/** Build a session: due cards first, then a few never-seen terms. */
export function buildSession(allTerms: string[]): string[] {
  const t = today();
  const due = allTerms.filter((term) => {
    const c = state.cards[term];
    return c && c.due <= t;
  });
  const fresh = allTerms.filter((term) => !state.cards[term]).slice(0, NEW_PER_SESSION);
  return [...due, ...fresh];
}

export function dueAndNewCount(allTerms: string[]): number {
  return buildSession(allTerms).length;
}

export function learnedCount(): number {
  return Object.keys(state.cards).length;
}

export function useFlashcards(): FlashState {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => state,
  );
}
