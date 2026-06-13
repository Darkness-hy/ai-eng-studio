import { useEffect, useState, type FormEvent } from 'react';
import { createAssignment, deleteAssignment, listAssignments, type Assignment } from '../lib/assignments';
import { phaseTitle } from '../lib/i18n';
import { useProgress } from '../lib/progress';
import type { CourseIndex, Lang } from '../lib/types';
import type { ProgressRow } from '../lib/supabase';

export function AssignmentsPanel({
  classId,
  isOwner,
  index,
  progressRows,
  memberCount,
  lang,
}: {
  classId: string;
  isOwner: boolean;
  index: CourseIndex;
  progressRows: ProgressRow[];
  memberCount: number;
  lang: Lang;
}) {
  const zh = lang === 'zh';
  const [items, setItems] = useState<Assignment[] | null>(null);
  const [title, setTitle] = useState('');
  const [phaseSlug, setPhaseSlug] = useState(index.phases[0]?.slug ?? '');
  const [due, setDue] = useState('');
  const [busy, setBusy] = useState(false);
  const progress = useProgress(); // for the student's own status

  useEffect(() => {
    listAssignments(classId).then(setItems).catch(() => setItems([]));
  }, [classId]);

  const reload = () => listAssignments(classId).then(setItems).catch(() => setItems([]));

  // owner: lessons each member has completed
  const doneByUser = new Map<string, Set<string>>();
  for (const r of progressRows) {
    if (!r.done) continue;
    let s = doneByUser.get(r.user_id);
    if (!s) {
      s = new Set();
      doneByUser.set(r.user_id, s);
    }
    s.add(r.lesson_id);
  }

  const create = async (e: FormEvent) => {
    e.preventDefault();
    const ph = index.phases.find((p) => p.slug === phaseSlug);
    if (!ph || !title.trim()) return;
    setBusy(true);
    try {
      await createAssignment(
        classId,
        title.trim(),
        ph.lessons.map((l) => `${ph.slug}/${l.slug}`),
        due || null,
      );
      setTitle('');
      setDue('');
      reload();
    } catch {
      /* surfaced by empty reload */
    }
    setBusy(false);
  };

  const field =
    'rounded-md border border-hairline bg-paper px-3 py-2 text-[13.5px] outline-none focus:border-faint';

  return (
    <section className="mt-12">
      <div className="mb-3 font-mono text-[10.5px] uppercase tracking-[0.16em] text-faint">
        {zh ? '作业' : 'Assignments'}
      </div>

      {isOwner && (
        <form onSubmit={create} className="mb-4 flex flex-wrap items-center gap-2 rounded-lg border border-hairline bg-paper p-4">
          <input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder={zh ? '作业标题' : 'Title'}
            className={`${field} flex-1`}
            maxLength={60}
          />
          <select value={phaseSlug} onChange={(e) => setPhaseSlug(e.target.value)} className={field}>
            {index.phases.map((p) => (
              <option key={p.slug} value={p.slug}>
                {zh ? `阶段 ${p.num} · ${p.titleZh}` : `Phase ${p.num} · ${p.titleEn}`}
              </option>
            ))}
          </select>
          <input
            type="date"
            value={due}
            onChange={(e) => setDue(e.target.value)}
            className={field}
            title={zh ? '截止日期' : 'Due date'}
          />
          <button type="submit" disabled={busy} className="rounded-md bg-ink px-4 py-2 text-[13px] text-white hover:bg-ink/85 disabled:opacity-50">
            {zh ? '布置' : 'Assign'}
          </button>
        </form>
      )}

      {items && items.length === 0 && (
        <div className="rounded-lg border border-dashed border-hairline px-5 py-6 text-center text-[13px] text-faint">
          {zh ? '还没有作业' : 'No assignments yet'}
        </div>
      )}

      <div className="space-y-2">
        {items?.map((a) => {
          const ph = index.phases.find((p) => `${p.slug}/${p.lessons[0]?.slug}` === a.lesson_ids[0]);
          const N = a.lesson_ids.length;
          const overdue = a.due_date && a.due_date < new Date().toISOString().slice(0, 10);
          return (
            <div key={a.id} className="rounded-lg border border-hairline bg-paper px-5 py-3.5">
              <div className="flex items-baseline justify-between gap-3">
                <div className="min-w-0">
                  <div className="truncate text-[14.5px] font-medium">{a.title}</div>
                  <div className="mt-0.5 font-mono text-[11px] text-faint">
                    {ph ? (zh ? phaseTitle(ph, lang) : ph.titleEn) : ''} · {N} {zh ? '课' : 'lessons'}
                    {a.due_date && (
                      <span className={overdue ? 'text-ink-red' : ''}>
                        {' '}
                        · {zh ? '截止' : 'due'} {a.due_date}
                      </span>
                    )}
                  </div>
                </div>
                {isOwner && (
                  <button
                    type="button"
                    onClick={() => deleteAssignment(a.id).then(reload)}
                    className="shrink-0 font-mono text-[11px] text-faint hover:text-ink-red"
                  >
                    {zh ? '删除' : 'delete'}
                  </button>
                )}
              </div>
              {isOwner ? (
                (() => {
                  let completed = 0;
                  for (const set of doneByUser.values())
                    if (a.lesson_ids.every((l) => set.has(l))) completed += 1;
                  const pct = memberCount ? (completed / memberCount) * 100 : 0;
                  return (
                    <div className="mt-2 flex items-center gap-3">
                      <span className="h-[3px] flex-1 overflow-hidden rounded-full bg-bone">
                        <span className="block h-full rounded-full bg-ink-green" style={{ width: `${pct}%` }} />
                      </span>
                      <span className="font-mono text-[11.5px] text-faint">
                        {zh ? '完成' : 'done'} {completed}/{memberCount}
                      </span>
                    </div>
                  );
                })()
              ) : (
                (() => {
                  const mine = a.lesson_ids.filter((l) => progress.lessons[l]?.done).length;
                  const done = mine === N;
                  return (
                    <div className="mt-2 flex items-center gap-3">
                      <span className="h-[3px] flex-1 overflow-hidden rounded-full bg-bone">
                        <span className="block h-full rounded-full bg-ink-green" style={{ width: `${(mine / N) * 100}%` }} />
                      </span>
                      <span className={`font-mono text-[11.5px] ${done ? 'text-ink-green' : 'text-faint'}`}>
                        {done ? (zh ? '已完成 ✓' : 'done ✓') : `${mine}/${N}`}
                      </span>
                    </div>
                  );
                })()
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
