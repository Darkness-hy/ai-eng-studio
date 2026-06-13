import { useSyncExternalStore } from 'react';

/** Local record of which lessons' coding challenges the learner has passed. */
const KEY = 'aes:challenges:v1';
const listeners = new Set<() => void>();

function load(): Set<string> {
  try {
    const raw = localStorage.getItem(KEY);
    return new Set(raw ? (JSON.parse(raw) as string[]) : []);
  } catch {
    return new Set();
  }
}

let passed = load();

export function markChallengePassed(lessonId: string) {
  if (passed.has(lessonId)) return;
  passed = new Set(passed);
  passed.add(lessonId);
  localStorage.setItem(KEY, JSON.stringify([...passed]));
  listeners.forEach((fn) => fn());
}

export function isChallengePassed(lessonId: string): boolean {
  return passed.has(lessonId);
}

export function useChallenges(): Set<string> {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => passed,
  );
}
