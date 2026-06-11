import {
  getProgress,
  mergeRemote,
  subscribeChanges,
  type LessonProgress,
} from './progress';
import { cloudEnabled, getSupabase, type ProgressRow } from './supabase';

/**
 * Local-first cloud sync.
 * - On login: pull remote rows, merge (newest updatedAt wins), then push the
 *   full local state so a fresh device adopts existing anonymous progress.
 * - Afterwards: local mutations are batched and upserted after a 2s debounce.
 */

let stopFns: (() => void)[] = [];
let dirty = new Set<string>();
let dirtyDays = new Set<string>();
let timer: ReturnType<typeof setTimeout> | null = null;
let activeUserId: string | null = null;

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
  if (!activeUserId || (dirty.size === 0 && dirtyDays.size === 0)) return;
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
    // Network hiccup: requeue and retry on the next mutation or flush.
    lessonIds.forEach((id) => dirty.add(id));
    days.forEach((d) => dirtyDays.add(d));
    console.warn('[sync] flush failed, will retry', err);
  }
}

function schedule() {
  if (timer) clearTimeout(timer);
  timer = setTimeout(() => void flush(), 2000);
}

/** Pull remote state, merge into local, then push everything local. */
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
  mergeRemote(remote, ((acts ?? []) as { day: string }[]).map((a) => a.day));

  // Push the merged state so this account has everything from this device.
  const state = getProgress();
  Object.keys(state.lessons).forEach((id) => dirty.add(id));
  state.visits.forEach((d) => dirtyDays.add(d));
  activeUserId = userId;
  await flush();
}

/** Start watching local mutations for the logged-in user. */
export function startSync(userId: string): void {
  stopSync();
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
  stopFns.forEach((fn) => fn());
  stopFns = [];
  if (timer) clearTimeout(timer);
  timer = null;
  activeUserId = null;
  dirty = new Set();
  dirtyDays = new Set();
}
