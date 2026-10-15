import { syncPlacementCloud } from './placement';
import {
  getProgress,
  replaceLocal,
  subscribeChanges,
  type LessonProgress,
} from './progress';
import { cloudEnabled, getSupabase, type ProgressRow } from './supabase';

/**
 * Local-first cloud sync.
 * - On login: auth switches progress/placement to a user-scoped localStorage key
 *   before this runs, so anonymous or previous-user browser data is not imported.
 * - Pull remote rows and replace the scoped local cache with cloud state. Cloud
 *   is authoritative on login, including admin resets to empty progress.
 * - Afterwards: local mutations are batched and upserted after a 2s debounce.
 */

let stopFns: (() => void)[] = [];
let dirty = new Set<string>();
let dirtyDays = new Set<string>();
let timer: ReturnType<typeof setTimeout> | null = null;
let activeUserId: string | null = null;
let mergeDone = false; // true once initialSync merged remote → uploads are safe

function rowToLocal(row: ProgressRow): LessonProgress {
  return {
    done: row.done || undefined,
    preScore: row.pre_score ?? undefined,
    preTotal: row.pre_total ?? undefined,
    postScore: row.post_score ?? undefined,
    postTotal: row.post_total ?? undefined,
    completedAt: row.completed_at ?? undefined,
    updatedAt: row.updated_at,
  };
}

function localToRow(userId: string, lessonId: string, p: LessonProgress): ProgressRow {
  return {
    user_id: userId,
    lesson_id: lessonId,
    done: Boolean(p.done),
    pre_score: p.preScore ?? null,
    pre_total: p.preTotal ?? null,
    post_score: p.postScore ?? null,
    post_total: p.postTotal ?? null,
    completed_at: p.completedAt ?? null,
    updated_at: p.updatedAt ?? new Date().toISOString(),
  };
}

async function flush() {
  // Gate uploads on a completed initial merge so a failed/partial initialSync
  // can't push un-merged local state over newer remote data.
  if (!activeUserId || !mergeDone || (dirty.size === 0 && dirtyDays.size === 0)) return;
  const supabase = getSupabase();
  const state = getProgress();
  const lessonIds = [...dirty];
  const days = [...dirtyDays];
  dirty = new Set();
  dirtyDays = new Set();
  try {
    if (lessonIds.length > 0) {
      const rows = lessonIds
        .filter((id) => state.lessons[id])
        .map((id) => localToRow(activeUserId!, id, state.lessons[id]));
      if (rows.length > 0) {
        const { error } = await supabase.from('progress').upsert(rows);
        if (error) throw error;
      }
    }
    if (days.length > 0) {
      const { error } = await supabase
        .from('activity')
        .upsert(days.map((day) => ({ user_id: activeUserId!, day })), { ignoreDuplicates: true });
      if (error) throw error;
    }
  } catch (err) {
    // Network hiccup: requeue and retry on the next mutation/flush — unless we
    // logged out meanwhile (don't resurrect a dead session's dirty set).
    if (activeUserId) {
      lessonIds.forEach((id) => dirty.add(id));
      days.forEach((d) => dirtyDays.add(d));
    }
    console.warn('[sync] flush failed, will retry', err);
  }
}

function schedule() {
  if (timer) clearTimeout(timer);
  timer = setTimeout(() => void flush(), 2000);
}

/** Pull remote state into this user's scoped local cache. */
export async function initialSync(userId: string): Promise<void> {
  if (!cloudEnabled) return;
  const supabase = getSupabase();

  const [{ data: rows, error }, { data: acts, error: actError }] = await Promise.all([
    supabase.from('progress').select('*').eq('user_id', userId),
    supabase.from('activity').select('day').eq('user_id', userId),
  ]);
  if (error) throw error;
  if (actError) throw actError;

  const remote: Record<string, LessonProgress> = {};
  for (const row of (rows ?? []) as ProgressRow[]) remote[row.lesson_id] = rowToLocal(row);
  replaceLocal(remote, ((acts ?? []) as { day: string }[]).map((a) => a.day));
  mergeDone = true; // remote loaded into local — future user edits may upload

  // Placement result rides along with the same login sync.
  await syncPlacementCloud(userId).catch(() => undefined);

  activeUserId = userId;
}

function teardownListeners(): void {
  stopFns.forEach((fn) => fn());
  stopFns = [];
  if (timer) {
    clearTimeout(timer);
    timer = null;
  }
}

/** Flush pending writes immediately and wait for them. Call before logout so the
 *  2s-debounce window does not drop the tail. No-op if nothing is pending. */
export async function flushNow(): Promise<void> {
  if (timer) {
    clearTimeout(timer);
    timer = null;
  }
  await flush();
}

/** Start watching local mutations for the logged-in user. Tears down only the
 *  listeners (not session state), so the mergeDone flag set by the preceding
 *  initialSync survives. */
export function startSync(userId: string): void {
  teardownListeners();
  activeUserId = userId;
  const unsub = subscribeChanges((change) => {
    if (change.source !== 'local') return;
    change.lessonIds.forEach((id) => dirty.add(id));
    if (change.visitDay) dirtyDays.add(change.visitDay);
    schedule();
  });
  const onBeforeUnload = () => void flush();
  window.addEventListener('beforeunload', onBeforeUnload);
  stopFns = [unsub, () => window.removeEventListener('beforeunload', onBeforeUnload)];
}

export function stopSync(): void {
  teardownListeners();
  activeUserId = null;
  mergeDone = false;
  dirty = new Set();
  dirtyDays = new Set();
}
