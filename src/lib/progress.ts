import { useSyncExternalStore } from 'react';

export interface LessonProgress {
  done?: boolean;
  preScore?: number;
  preTotal?: number;
  postScore?: number;
  postTotal?: number;
  completedAt?: string;
  updatedAt?: string;
}

export interface ProgressState {
  v: 1;
  lessons: Record<string, LessonProgress>;
  visits: string[]; // ISO dates (YYYY-MM-DD) with activity, for streaks
  lastLesson?: string; // "phaseSlug/lessonSlug"
}

export interface ProgressChange {
  source: 'local' | 'remote';
  lessonIds: string[];
  visitDay?: string;
}

const KEY = 'aes:progress:v1';
const renderListeners = new Set<() => void>();
const changeListeners = new Set<(change: ProgressChange) => void>();

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

function commit(next: ProgressState, change?: ProgressChange) {
  state = next;
  localStorage.setItem(KEY, JSON.stringify(state));
  renderListeners.forEach((fn) => fn());
  if (change) changeListeners.forEach((fn) => fn(change));
}

// Cross-tab sync: when another tab writes our key, reload and re-render. We do
// NOT emit a change event here (no cloud re-push) — the other tab already
// persisted/synced its own write.
if (typeof window !== 'undefined') {
  window.addEventListener('storage', (e) => {
    if (e.key === KEY) {
      state = load();
      renderListeners.forEach((fn) => fn());
    }
  });
}

function today(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

export function getProgress(): ProgressState {
  return state;
}

/** Subscribe to data mutations (used by the cloud sync layer). */
export function subscribeChanges(fn: (change: ProgressChange) => void): () => void {
  changeListeners.add(fn);
  return () => changeListeners.delete(fn);
}

export function setLessonDone(id: string, done: boolean) {
  const prev = state.lessons[id] ?? {};
  commit(
    {
      ...state,
      lessons: {
        ...state.lessons,
        [id]: {
          ...prev,
          done,
          completedAt: done ? new Date().toISOString() : undefined,
          updatedAt: new Date().toISOString(),
        },
      },
    },
    { source: 'local', lessonIds: [id] },
  );
}

/** Mark many lessons done in a single commit (additive — never un-marks).
 *  Used by placement to pre-complete already-mastered phases. */
export function markLessonsDone(ids: string[]) {
  const now = new Date().toISOString();
  const lessons = { ...state.lessons };
  const changed: string[] = [];
  for (const id of ids) {
    if (lessons[id]?.done) continue;
    lessons[id] = { ...lessons[id], done: true, completedAt: now, updatedAt: now };
    changed.push(id);
  }
  if (changed.length === 0) return;
  commit({ ...state, lessons }, { source: 'local', lessonIds: changed });
}

/** Records the FIRST attempt only — retakes are practice and never overwrite. */
export function saveQuizScore(id: string, stage: 'pre' | 'post', score: number, total: number) {
  const prev = state.lessons[id] ?? {};
  const alreadyRecorded = stage === 'pre' ? prev.preTotal != null : prev.postTotal != null;
  if (alreadyRecorded) return;
  const patch =
    stage === 'pre' ? { preScore: score, preTotal: total } : { postScore: score, postTotal: total };
  commit(
    {
      ...state,
      lessons: {
        ...state.lessons,
        [id]: { ...prev, ...patch, updatedAt: new Date().toISOString() },
      },
    },
    { source: 'local', lessonIds: [id] },
  );
}

export function recordVisit(lessonId: string) {
  const day = today();
  const isNewDay = !state.visits.includes(day);
  const visits = isNewDay ? [...state.visits, day] : state.visits;
  commit(
    { ...state, visits, lastLesson: lessonId },
    isNewDay ? { source: 'local', lessonIds: [], visitDay: day } : undefined,
  );
}

/**
 * Apply cloud rows into the local store. Newest updatedAt wins per lesson;
 * visits are unioned. Emits a 'remote' change so the sync layer does not
 * push these rows back.
 */
/** Merge two versions of one lesson field-by-field so a newer-but-sparser side
 *  can't wipe the other's data. done/completedAt/updatedAt follow the newer write
 *  (done can legitimately toggle off); first-attempt scores are write-once, so we
 *  keep whichever side recorded them. */
function mergeLesson(ours: LessonProgress, theirs: LessonProgress): LessonProgress {
  const newer = (ours.updatedAt ?? '') >= (theirs.updatedAt ?? '') ? ours : theirs;
  return {
    done: newer.done,
    completedAt: newer.completedAt,
    updatedAt: newer.updatedAt,
    preScore: ours.preScore ?? theirs.preScore,
    preTotal: ours.preTotal ?? theirs.preTotal,
    postScore: ours.postScore ?? theirs.postScore,
    postTotal: ours.postTotal ?? theirs.postTotal,
  };
}

export function mergeRemote(remote: Record<string, LessonProgress>, remoteVisits: string[]) {
  const lessons = { ...state.lessons };
  for (const [id, theirs] of Object.entries(remote)) {
    const ours = lessons[id];
    lessons[id] = ours ? mergeLesson(ours, theirs) : theirs;
  }
  const visits = [...new Set([...state.visits, ...remoteVisits])].sort();
  commit({ ...state, lessons, visits }, { source: 'remote', lessonIds: Object.keys(remote) });
}

export function exportProgress(): string {
  return JSON.stringify(state, null, 2);
}

export function importProgress(json: string): boolean {
  try {
    const parsed = JSON.parse(json) as ProgressState;
    if (parsed?.v !== 1 || typeof parsed.lessons !== 'object') return false;
    commit(
      { v: 1, lessons: parsed.lessons, visits: parsed.visits ?? [], lastLesson: parsed.lastLesson },
      { source: 'local', lessonIds: Object.keys(parsed.lessons) },
    );
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
      renderListeners.add(cb);
      return () => renderListeners.delete(cb);
    },
    () => state,
  );
}
