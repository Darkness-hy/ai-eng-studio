import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { SparkApprovals } from '../components/SparkApprovals';
import { BarChart } from '../components/charts/BarChart';
import { Histogram } from '../components/charts/Histogram';
import { LineChart } from '../components/charts/LineChart';
import { useAuth } from '../lib/auth';
import { fetchIndex } from '../lib/data';
import { useLang } from '../lib/i18n';
import { AREAS, QUESTIONS, QUESTIONS_PER_AREA } from '../lib/placement';
import { getSupabase, type ActivityRow, type ProfileRow, type ProgressRow } from '../lib/supabase';
import type { CourseIndex, Lang } from '../lib/types';

const LETTERS = ['A', 'B', 'C', 'D'];

interface PlacementRow {
  user_id: string;
  answers: number[]; // per-question chosen index, -1 = unanswered
  area_scores: Record<string, number>;
  total: number;
  entry: number;
  taken_at: string;
}

interface AdminData {
  profiles: ProfileRow[];
  progress: ProgressRow[];
  activity: ActivityRow[];
  placement: PlacementRow[];
  index: CourseIndex;
}

interface PhaseDone {
  num: number;
  titleZh: string;
  titleEn: string;
  done: number;
  total: number;
}

interface StudentStat {
  profile: ProfileRow;
  doneCount: number;
  donePct: number;
  perPhase: PhaseDone[];
  quizAvg: number | null; // 0-100
  streak: number;
  lastActive: string | null;
  recentQuizzes: ProgressRow[]; // newest first, max 10
  placement: PlacementRow | null;
}

interface Metrics {
  students: StudentStat[];
  totalLessons: number;
  avgDonePct: number;
  avgPostPct: number | null;
  activeToday: number;
  phaseBars: { label: string; value: number; hint: string }[];
  activityPoints: { label: string; value: number }[];
  histBuckets: { label: string; count: number }[];
  placementCount: number;
  placementQuestionBars: { label: string; value: number; hint: string }[];
}

/** Grade one placement answer against the answer key. */
function gradeAnswer(answers: number[] | undefined, i: number): 'correct' | 'wrong' | 'blank' {
  const a = answers?.[i];
  if (a == null || a === -1) return 'blank';
  return a === QUESTIONS[i].correct ? 'correct' : 'wrong';
}

const pad2 = (n: number): string => String(n).padStart(2, '0');

const fmtPct = (v: number): string => `${Math.round(v * 10) / 10}%`;

/** Mask a learner's email for the admin table (first ~5 chars + domain) so a
 *  screenshot of this page doesn't expose their address. */
function maskEmail(email: string): string {
  const at = email.indexOf('@');
  if (at < 1) return email;
  const local = email.slice(0, at);
  const head = local.slice(0, Math.min(5, Math.max(1, local.length - 1)));
  return `${head}*****${email.slice(at)}`;
}

function dayKey(d: Date): string {
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
}

/** Consecutive active days ending today or yesterday (same logic as local streaks). */
function computeStreak(days: ReadonlySet<string>): number {
  let count = 0;
  const cursor = new Date();
  if (!days.has(dayKey(cursor))) cursor.setDate(cursor.getDate() - 1);
  while (days.has(dayKey(cursor))) {
    count += 1;
    cursor.setDate(cursor.getDate() - 1);
  }
  return count;
}

function buildMetrics(data: AdminData): Metrics {
  const { profiles, progress, activity, placement, index } = data;
  const totalLessons = index.stats.lessons;

  const placementByUser = new Map<string, PlacementRow>();
  for (const row of placement) placementByUser.set(row.user_id, row);

  const doneByUser = new Map<string, Set<string>>();
  const postByUser = new Map<string, ProgressRow[]>();
  for (const row of progress) {
    if (row.done) {
      const set = doneByUser.get(row.user_id) ?? new Set<string>();
      set.add(row.lesson_id);
      doneByUser.set(row.user_id, set);
    }
    if (row.post_total != null && row.post_total > 0) {
      const list = postByUser.get(row.user_id) ?? [];
      list.push(row);
      postByUser.set(row.user_id, list);
    }
  }

  const daysByUser = new Map<string, Set<string>>();
  const usersByDay = new Map<string, Set<string>>();
  for (const row of activity) {
    const days = daysByUser.get(row.user_id) ?? new Set<string>();
    days.add(row.day);
    daysByUser.set(row.user_id, days);
    const users = usersByDay.get(row.day) ?? new Set<string>();
    users.add(row.user_id);
    usersByDay.set(row.day, users);
  }

  const students: StudentStat[] = profiles
    .filter((p) => p.role === 'student')
    .map((p) => {
      const doneSet = doneByUser.get(p.id) ?? new Set<string>();
      const perPhase: PhaseDone[] = index.phases.map((ph) => ({
        num: ph.num,
        titleZh: ph.titleZh,
        titleEn: ph.titleEn,
        done: ph.lessons.filter((l) => doneSet.has(`${ph.slug}/${l.slug}`)).length,
        total: ph.lessons.length,
      }));
      const doneCount = perPhase.reduce((acc, x) => acc + x.done, 0);
      const postRows = postByUser.get(p.id) ?? [];
      const quizAvg = postRows.length
        ? (postRows.reduce((acc, r) => acc + (r.post_score ?? 0) / (r.post_total ?? 1), 0) /
            postRows.length) *
          100
        : null;
      const days = daysByUser.get(p.id) ?? new Set<string>();
      const sortedDays = Array.from(days).sort();
      return {
        profile: p,
        doneCount,
        donePct: totalLessons ? (doneCount / totalLessons) * 100 : 0,
        perPhase,
        quizAvg,
        streak: computeStreak(days),
        lastActive: sortedDays.length ? sortedDays[sortedDays.length - 1] : null,
        recentQuizzes: [...postRows]
          .sort((a, b) => b.updated_at.localeCompare(a.updated_at))
          .slice(0, 10),
        placement: placementByUser.get(p.id) ?? null,
      };
    });
  // 学生明细按注册时间降序(最新注册在前)。
  students.sort((a, b) => b.profile.created_at.localeCompare(a.profile.created_at));

  // Class-wide per-question correct rate: among students who attempted each
  // question (i.e. answered it), the share who got it right.
  const placedStudents = students.filter((s) => s.placement);
  const placementQuestionBars = QUESTIONS.map((q, i) => {
    let attempted = 0;
    let correct = 0;
    for (const s of placedStudents) {
      const a = s.placement!.answers[i];
      if (a != null && a !== -1) {
        attempted += 1;
        if (a === q.correct) correct += 1;
      }
    }
    const area = AREAS[Math.floor(i / QUESTIONS_PER_AREA)];
    return {
      label: String(i + 1),
      value: attempted ? (correct / attempted) * 100 : 0,
      hint: `${area.zh} · Q${i + 1}`,
    };
  });

  const avgDonePct = students.length
    ? students.reduce((acc, s) => acc + s.donePct, 0) / students.length
    : 0;

  const allPost = progress.filter((r) => r.post_total != null && r.post_total > 0);
  const avgPostPct = allPost.length
    ? (allPost.reduce((acc, r) => acc + (r.post_score ?? 0) / (r.post_total ?? 1), 0) /
        allPost.length) *
      100
    : null;

  const activeToday = usersByDay.get(dayKey(new Date()))?.size ?? 0;

  const phaseBars = index.phases.map((ph, j) => ({
    label: pad2(ph.num),
    value: students.length
      ? (students.reduce((acc, s) => {
          const slot = s.perPhase[j];
          return acc + (slot.total ? slot.done / slot.total : 0);
        }, 0) /
          students.length) *
        100
      : 0,
    hint: ph.titleZh,
  }));

  const activityPoints: { label: string; value: number }[] = [];
  for (let i = 29; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    activityPoints.push({
      label: `${d.getMonth() + 1}/${d.getDate()}`,
      value: usersByDay.get(dayKey(d))?.size ?? 0,
    });
  }

  const histBuckets = Array.from({ length: 10 }, (_, i) => ({
    label: `${i * 10}-${(i + 1) * 10}`,
    count: 0,
  }));
  for (const r of allPost) {
    const pct = (r.post_score ?? 0) / (r.post_total ?? 1);
    histBuckets[Math.min(Math.floor(pct * 10), 9)].count += 1;
  }

  return {
    students,
    totalLessons,
    avgDonePct,
    avgPostPct,
    activeToday,
    phaseBars,
    activityPoints,
    histBuckets,
    placementCount: placedStudents.length,
    placementQuestionBars,
  };
}

export function AdminPage() {
  const { enabled, loading, profile } = useAuth();
  const { lang } = useLang();
  const [data, setData] = useState<AdminData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const isAdmin = enabled && profile?.role === 'admin';

  // Only setState after the await, so this is safe to call from an effect.
  const load = useCallback(async () => {
    if (!isAdmin) return;
    try {
      const supabase = getSupabase();
      const [profilesRes, progressRes, activityRes, placementRes, index] = await Promise.all([
        supabase.from('profiles').select('*'),
        supabase.from('progress').select('*'),
        supabase.from('activity').select('*'),
        supabase.from('placement').select('*'),
        fetchIndex(),
      ]);
      const failed =
        profilesRes.error ?? progressRes.error ?? activityRes.error ?? placementRes.error;
      if (failed) {
        setError(failed.message);
        return;
      }
      setError(null);
      setData({
        profiles: (profilesRes.data ?? []) as ProfileRow[],
        progress: (progressRes.data ?? []) as ProgressRow[],
        activity: (activityRes.data ?? []) as ActivityRow[],
        placement: (placementRes.data ?? []) as PlacementRow[],
        index,
      });
      setUpdatedAt(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [isAdmin]);

  // Manual refresh shows a spinner — called from a click, not an effect.
  const refresh = useCallback(() => {
    setRefreshing(true);
    void load().finally(() => setRefreshing(false));
  }, [load]);

  // Keep the dashboard fresh: load on mount, silently refetch when the tab
  // regains focus, and on a slow poll while visible — so the numbers track the
  // live database without needing a page reload.
  useEffect(() => {
    if (!isAdmin) return;
    const onVisible = () => {
      if (document.visibilityState === 'visible') void load();
    };
    const initial = setTimeout(onVisible, 0); // first load, off the effect body
    document.addEventListener('visibilitychange', onVisible);
    window.addEventListener('focus', onVisible);
    const timer = setInterval(onVisible, 30000);
    return () => {
      clearTimeout(initial);
      document.removeEventListener('visibilitychange', onVisible);
      window.removeEventListener('focus', onVisible);
      clearInterval(timer);
    };
  }, [isAdmin, load]);

  if (!enabled) {
    return <Notice text={lang === 'zh' ? '云同步未配置' : 'Cloud sync not configured'} />;
  }
  if (loading) return <Notice text={lang === 'zh' ? '加载中…' : 'Loading…'} />;
  if (profile?.role !== 'admin') {
    return <Notice text={lang === 'zh' ? '无权访问，仅管理员可见' : 'Admins only'} />;
  }
  if (error && !data) {
    return <Notice text={`${lang === 'zh' ? '加载失败' : 'Failed to load'}: ${error}`} />;
  }
  if (!data) return <Notice text={lang === 'zh' ? '加载中…' : 'Loading…'} />;

  return (
    <AdminDashboard
      data={data}
      lang={lang}
      onRefresh={refresh}
      refreshing={refreshing}
      updatedAt={updatedAt}
    />
  );
}

function Notice({ text }: { text: string }) {
  return <div className="py-32 text-center text-[14px] text-faint">{text}</div>;
}

function AdminDashboard({
  data,
  lang,
  onRefresh,
  refreshing,
  updatedAt,
}: {
  data: AdminData;
  lang: Lang;
  onRefresh: () => void;
  refreshing: boolean;
  updatedAt: Date | null;
}) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const zh = lang === 'zh';
  const m = useMemo(() => buildMetrics(data), [data]);

  return (
    <div className="mx-auto max-w-6xl px-5 py-14">
      <header className="flex flex-wrap items-end justify-between gap-4 border-b border-hairline pb-8">
        <div>
          <h1 className="font-serif text-[38px] font-semibold tracking-tight">
            {zh ? '管理后台' : 'Admin Console'}
          </h1>
          <p className="mt-2 font-mono text-[12px] uppercase tracking-[0.14em] text-faint">
            {zh
              ? `学生 ${m.students.length} · 总课程数 ${m.totalLessons}`
              : `${m.students.length} students · ${m.totalLessons} lessons total`}
          </p>
        </div>
        <div className="flex items-center gap-3">
          {updatedAt && (
            <span className="font-mono text-[11px] text-faint">
              {zh ? '更新于 ' : 'updated '}
              {updatedAt.toLocaleTimeString(zh ? 'zh-CN' : 'en-US', {
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
              })}
            </span>
          )}
          <button
            type="button"
            onClick={onRefresh}
            disabled={refreshing}
            className="rounded-md border border-hairline px-3 py-1.5 font-mono text-[11.5px] text-faint transition-colors hover:bg-bone hover:text-ink disabled:opacity-50"
          >
            {refreshing ? (zh ? '刷新中…' : 'Refreshing…') : zh ? '↻ 刷新' : '↻ Refresh'}
          </button>
        </div>
      </header>

      <section className="mt-8 grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatCard label={zh ? '学生数' : 'Students'} value={String(m.students.length)} />
        <StatCard label={zh ? '平均完成度' : 'Avg completion'} value={fmtPct(m.avgDonePct)} />
        <StatCard
          label={zh ? '平均课后测验得分率' : 'Avg post-quiz score'}
          value={m.avgPostPct == null ? '—' : fmtPct(m.avgPostPct)}
        />
        <StatCard label={zh ? '今日活跃' : 'Active today'} value={String(m.activeToday)} />
      </section>

      <section className="mt-8 grid gap-3 md:grid-cols-2">
        <ChartCard label={zh ? '各阶段平均完成度' : 'Avg completion by phase'} wide>
          <BarChart data={m.phaseBars} maxValue={100} format={fmtPct} />
        </ChartCard>
        <ChartCard label={zh ? '近 30 天每日活跃人数' : 'Daily active users · 30 days'}>
          <LineChart points={m.activityPoints} />
        </ChartCard>
        <ChartCard label={zh ? '课后测验得分分布' : 'Post-quiz score distribution'}>
          <Histogram buckets={m.histBuckets} />
        </ChartCard>
        {m.placementCount > 0 && (
          <ChartCard
            label={
              zh
                ? `定级题目正确率（全班 ${m.placementCount} 人，越低越难）`
                : `Placement question accuracy (${m.placementCount} graded)`
            }
            wide
          >
            <BarChart data={m.placementQuestionBars} maxValue={100} format={fmtPct} />
          </ChartCard>
        )}
      </section>

      <section className="mt-8 rounded-lg border border-hairline bg-paper">
        <div className="border-b border-hairline px-6 py-4 font-mono text-[10.5px] uppercase tracking-[0.16em] text-faint">
          {zh ? '学生明细' : 'Students'}
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[13.5px]">
            <thead>
              <tr className="border-b border-hairline text-left font-mono text-[10.5px] uppercase tracking-[0.12em] text-faint">
                <th className="px-6 py-3 font-normal">{zh ? '学生' : 'Student'}</th>
                <th className="px-4 py-3 font-normal">{zh ? '完成进度' : 'Progress'}</th>
                <th className="px-4 py-3 font-normal">{zh ? '测验均分' : 'Quiz avg'}</th>
                <th className="px-4 py-3 font-normal">{zh ? '连续天数' : 'Streak'}</th>
                <th className="px-4 py-3 font-normal">{zh ? '最近活跃' : 'Last active'}</th>
                <th className="px-6 py-3 font-normal">{zh ? '注册时间' : 'Joined'}</th>
              </tr>
            </thead>
            <tbody>
              {m.students.map((s) => (
                <StudentRow
                  key={s.profile.id}
                  stat={s}
                  totalLessons={m.totalLessons}
                  lang={lang}
                  expanded={expanded === s.profile.id}
                  onToggle={() =>
                    setExpanded((prev) => (prev === s.profile.id ? null : s.profile.id))
                  }
                />
              ))}
              {m.students.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-6 py-10 text-center text-faint">
                    {zh ? '暂无学生' : 'No students yet'}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <SparkApprovals />
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-hairline bg-paper p-5">
      <div className="font-mono text-[10.5px] uppercase tracking-[0.16em] text-faint">{label}</div>
      <div className="mt-2 font-serif text-[30px] font-semibold leading-none tracking-tight">
        {value}
      </div>
    </div>
  );
}

function ChartCard({
  label,
  wide = false,
  children,
}: {
  label: string;
  wide?: boolean;
  children: ReactNode;
}) {
  return (
    <div
      className={`rounded-lg border border-hairline bg-paper p-6 ${wide ? 'md:col-span-2' : ''}`}
    >
      <div className="font-mono text-[10.5px] uppercase tracking-[0.16em] text-faint">{label}</div>
      <div className="mt-4">{children}</div>
    </div>
  );
}

function StudentRow({
  stat,
  totalLessons,
  lang,
  expanded,
  onToggle,
}: {
  stat: StudentStat;
  totalLessons: number;
  lang: Lang;
  expanded: boolean;
  onToggle: () => void;
}) {
  const zh = lang === 'zh';
  const p = stat.profile;

  return (
    <>
      <tr
        className="cursor-pointer border-b border-hairline transition-colors hover:bg-bone/50"
        onClick={onToggle}
      >
        <td className="px-6 py-3.5">
          <div className="font-medium">{p.display_name ?? p.email.split('@')[0]}</div>
          <div className="text-[12px] text-faint">{maskEmail(p.email)}</div>
        </td>
        <td className="px-4 py-3.5">
          <div className="flex items-center gap-3">
            <span className="h-[3px] w-24 overflow-hidden rounded-full bg-bone">
              <span
                className="block h-full rounded-full bg-ink-green"
                style={{ width: `${stat.donePct}%` }}
              />
            </span>
            <span className="font-mono text-[11.5px] text-faint">
              {stat.doneCount}/{totalLessons}
            </span>
          </div>
        </td>
        <td className="px-4 py-3.5 font-mono text-[12px]">
          {stat.quizAvg == null ? (
            <span className="text-faint">—</span>
          ) : (
            `${Math.round(stat.quizAvg)}%`
          )}
        </td>
        <td className="px-4 py-3.5 font-mono text-[12px]">{stat.streak}</td>
        <td className="px-4 py-3.5 font-mono text-[12px] text-faint">{stat.lastActive ?? '—'}</td>
        <td className="px-6 py-3.5 font-mono text-[12px] text-faint">
          {p.created_at.slice(0, 10)}
        </td>
      </tr>
      {expanded && (
        <tr className="border-b border-hairline bg-bone/40">
          <td colSpan={6} className="px-6 py-5">
            <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-faint">
              {zh ? '各阶段完成度' : 'Per-phase completion'}
            </div>
            <div className="mt-3 flex flex-wrap items-end gap-[5px]">
              {stat.perPhase.map((ph) => {
                const pct = ph.total ? (ph.done / ph.total) * 100 : 0;
                return (
                  <div
                    key={ph.num}
                    className="flex flex-col items-center gap-1"
                    title={`${zh ? ph.titleZh : ph.titleEn} · ${Math.round(pct)}%`}
                  >
                    <div className="flex h-12 w-3.5 items-end overflow-hidden rounded-sm bg-hairline/50">
                      <div className="w-full bg-ink-blue" style={{ height: `${pct}%` }} />
                    </div>
                    <span className="font-mono text-[9px] text-faint">{pad2(ph.num)}</span>
                  </div>
                );
              })}
            </div>
            <div className="mt-5 font-mono text-[10px] uppercase tracking-[0.16em] text-faint">
              {zh ? '最近 10 次测验' : 'Last 10 quizzes'}
            </div>
            {stat.recentQuizzes.length === 0 ? (
              <div className="mt-2 text-[12.5px] text-faint">
                {zh ? '暂无测验记录' : 'No quiz records'}
              </div>
            ) : (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {stat.recentQuizzes.map((r) => {
                  const pct = (r.post_score ?? 0) / (r.post_total ?? 1);
                  const tail = r.lesson_id.split('/').pop() ?? r.lesson_id;
                  return (
                    <span
                      key={r.lesson_id}
                      className={`rounded-md px-2 py-1 font-mono text-[10.5px] ${
                        pct >= 0.6 ? 'bg-pale-green text-ink-green' : 'bg-pale-red text-ink-red'
                      }`}
                    >
                      {tail} · {r.post_score}/{r.post_total}
                    </span>
                  );
                })}
              </div>
            )}
            {stat.placement && <PlacementDetail placement={stat.placement} zh={zh} />}
          </td>
        </tr>
      )}
    </>
  );
}

function PlacementDetail({ placement: pl, zh }: { placement: PlacementRow; zh: boolean }) {
  const areaScore = (key: string): number => pl.area_scores[key] ?? 0;
  const blanks = pl.answers.filter((a) => a === -1).length;
  const strongest = AREAS.reduce((best, a) => (areaScore(a.key) > areaScore(best.key) ? a : best));
  const weakest = AREAS.reduce((worst, a) => (areaScore(a.key) < areaScore(worst.key) ? a : worst));

  return (
    <div className="mt-5">
      <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-faint">
        {zh ? '定级测试答题情况' : 'Placement quiz answers'}
      </div>
      <div className="mt-2 font-mono text-[11.5px] text-faint">
        {zh ? '总分' : 'Total'} {pl.total}/50 · {zh ? '起点 阶段 ' : 'Entry Phase '}
        {pl.entry} · {zh ? '测于 ' : 'taken '}
        {pl.taken_at.slice(0, 10)}
      </div>
      <div className="mt-3 space-y-1.5">
        {AREAS.map((a, ai) => (
          <div key={a.key} className="flex items-center gap-3">
            <span className="w-36 shrink-0 text-[11.5px]">
              {zh ? a.zh : a.en}{' '}
              <span className="font-mono text-faint">
                {areaScore(a.key)}/{QUESTIONS_PER_AREA}
              </span>
            </span>
            <div className="flex flex-wrap gap-1">
              {Array.from({ length: QUESTIONS_PER_AREA }, (_, k) => {
                const idx = ai * QUESTIONS_PER_AREA + k;
                const g = gradeAnswer(pl.answers, idx);
                const sel = pl.answers[idx];
                const cls =
                  g === 'correct'
                    ? 'bg-pale-green text-ink-green'
                    : g === 'wrong'
                      ? 'bg-pale-red text-ink-red'
                      : 'border border-hairline bg-paper text-faint';
                const selLabel = sel == null || sel === -1 ? (zh ? '未答' : 'blank') : LETTERS[sel];
                const title =
                  `Q${idx + 1} · ${zh ? a.zh : a.en}\n` +
                  `${zh ? '你的选择' : 'chose'}: ${selLabel} · ${zh ? '正确答案' : 'key'}: ${LETTERS[QUESTIONS[idx].correct]}`;
                return (
                  <span
                    key={idx}
                    title={title}
                    className={`flex h-6 w-6 items-center justify-center rounded font-mono text-[9.5px] ${cls}`}
                  >
                    {idx + 1}
                  </span>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-2.5 text-[12px] text-faint">
        {zh
          ? `最强：${strongest.zh}（${areaScore(strongest.key)}/10）· 最弱：${weakest.zh}（${areaScore(weakest.key)}/10）· 未作答 ${blanks} 题`
          : `Strongest: ${strongest.en} (${areaScore(strongest.key)}/10) · Weakest: ${weakest.en} (${areaScore(weakest.key)}/10) · ${blanks} blank`}
      </div>
    </div>
  );
}
