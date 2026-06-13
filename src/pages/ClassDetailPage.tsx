import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { classMembers, getClass, type ClassRow } from '../lib/classes';
import { fetchIndex } from '../lib/data';
import { useLang } from '../lib/i18n';
import { getSupabase, type ProfileRow, type ProgressRow } from '../lib/supabase';
import type { CourseIndex } from '../lib/types';

interface MemberStat {
  profile: ProfileRow;
  doneCount: number;
  donePct: number;
  quizAvg: number | null;
  entry: number | null;
  lastActive: string | null;
}

export function ClassDetailPage() {
  const { classId } = useParams();
  const { lang } = useLang();
  const zh = lang === 'zh';
  const [cls, setCls] = useState<ClassRow | null>(null);
  const [stats, setStats] = useState<MemberStat[] | null>(null);
  const [totalLessons, setTotalLessons] = useState(0);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!classId) return;
    let live = true;
    (async () => {
      const supabase = getSupabase();
      const [c, members, index] = await Promise.all([getClass(classId), classMembers(classId), fetchIndex()]);
      if (!live) return;
      setCls(c);
      const ids = members.map((m) => m.id);
      const lessonsTotal = (index as CourseIndex).stats.lessons;
      setTotalLessons(lessonsTotal);
      let progress: ProgressRow[] = [];
      let placement: { user_id: string; entry: number }[] = [];
      let activity: { user_id: string; day: string }[] = [];
      if (ids.length) {
        const [p, pl, ac] = await Promise.all([
          supabase.from('progress').select('*').in('user_id', ids),
          supabase.from('placement').select('user_id,entry').in('user_id', ids),
          supabase.from('activity').select('user_id,day').in('user_id', ids),
        ]);
        progress = (p.data ?? []) as ProgressRow[];
        placement = (pl.data ?? []) as { user_id: string; entry: number }[];
        activity = (ac.data ?? []) as { user_id: string; day: string }[];
      }
      const idx = index as CourseIndex;
      const built = members.map((m): MemberStat => {
        const mine = progress.filter((r) => r.user_id === m.id);
        const doneSet = new Set(mine.filter((r) => r.done).map((r) => r.lesson_id));
        let doneCount = 0;
        for (const ph of idx.phases)
          doneCount += ph.lessons.filter((l) => doneSet.has(`${ph.slug}/${l.slug}`)).length;
        const post = mine.filter((r) => r.post_total != null && r.post_total > 0);
        const quizAvg = post.length
          ? Math.round((post.reduce((a, r) => a + (r.post_score ?? 0) / (r.post_total ?? 1), 0) / post.length) * 100)
          : null;
        const days = activity.filter((a) => a.user_id === m.id).map((a) => a.day).sort();
        const pl = placement.find((x) => x.user_id === m.id);
        return {
          profile: m,
          doneCount,
          donePct: lessonsTotal ? (doneCount / lessonsTotal) * 100 : 0,
          quizAvg,
          entry: pl ? pl.entry : null,
          lastActive: days.length ? days[days.length - 1] : null,
        };
      });
      built.sort((a, b) => b.doneCount - a.doneCount);
      setStats(built);
    })().catch(() => live && setFailed(true));
    return () => {
      live = false;
    };
  }, [classId]);

  if (failed) return <Notice text={zh ? '加载失败或无权访问' : 'Failed or no access'} />;
  if (!cls || !stats) return <Notice text={zh ? '加载中…' : 'Loading…'} />;

  const avg = stats.length ? Math.round(stats.reduce((a, s) => a + s.donePct, 0) / stats.length) : 0;

  return (
    <div className="mx-auto max-w-4xl px-5 py-14">
      <Link to="/classes" className="font-mono text-[11px] tracking-[0.14em] text-faint hover:text-ink">
        ← {zh ? '返回班级' : 'Back to classes'}
      </Link>
      <header className="mt-5 border-b border-hairline pb-6">
        <h1 className="font-serif text-[34px] font-semibold tracking-tight">{cls.name}</h1>
        <p className="mt-2 font-mono text-[12px] uppercase tracking-[0.12em] text-faint">
          {zh ? '邀请码' : 'CODE'} {cls.invite_code} · {stats.length} {zh ? '名学生' : 'students'} ·{' '}
          {zh ? `平均完成度 ${avg}%` : `${avg}% avg`}
        </p>
      </header>

      {stats.length === 0 ? (
        <Notice text={zh ? '还没有学生加入。把邀请码发给学生即可。' : 'No students yet — share the invite code.'} />
      ) : (
        <table className="mt-4 w-full text-[13.5px]">
          <thead>
            <tr className="border-b border-hairline text-left font-mono text-[10.5px] uppercase tracking-[0.12em] text-faint">
              <th className="py-3">{zh ? '学生' : 'Student'}</th>
              <th className="py-3">{zh ? '完成进度' : 'Progress'}</th>
              <th className="py-3">{zh ? '测验均分' : 'Quiz avg'}</th>
              <th className="py-3">{zh ? '定级起点' : 'Level'}</th>
              <th className="py-3 text-right">{zh ? '最近活跃' : 'Last active'}</th>
            </tr>
          </thead>
          <tbody>
            {stats.map((s) => (
              <tr key={s.profile.id} className="border-b border-hairline">
                <td className="py-3">
                  <div className="font-medium">{s.profile.display_name ?? s.profile.email.split('@')[0]}</div>
                  <div className="text-[12px] text-faint">{s.profile.email}</div>
                </td>
                <td className="py-3">
                  <div className="flex items-center gap-2">
                    <span className="h-[3px] w-20 overflow-hidden rounded-full bg-bone">
                      <span className="block h-full rounded-full bg-ink-green" style={{ width: `${s.donePct}%` }} />
                    </span>
                    <span className="font-mono text-[11.5px] text-faint">{s.doneCount}/{totalLessons}</span>
                  </div>
                </td>
                <td className="py-3 font-mono text-[12px]">{s.quizAvg == null ? <span className="text-faint">—</span> : `${s.quizAvg}%`}</td>
                <td className="py-3 font-mono text-[12px] text-faint">{s.entry == null ? '—' : `${zh ? '阶段' : 'P'} ${s.entry}`}</td>
                <td className="py-3 text-right font-mono text-[12px] text-faint">{s.lastActive ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function Notice({ text }: { text: string }) {
  return <div className="py-24 text-center text-[14px] text-faint">{text}</div>;
}
