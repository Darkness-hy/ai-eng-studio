import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { BadgeWall } from '../components/BadgeWall';
import { Heatmap } from '../components/Heatmap';
import { computeBadges, dailyCompletions } from '../lib/achievements';
import { useAuth } from '../lib/auth';
import { generateCertificate, generateShareCard, downloadBlob } from '../lib/certificate';
import { fetchIndex } from '../lib/data';
import { phaseTitle, useLang } from '../lib/i18n';
import { exportProgress, importProgress, streakDays, useProgress } from '../lib/progress';
import type { CourseIndex } from '../lib/types';

export function ProgressPage() {
  const { lang, t } = useLang();
  const { profile } = useAuth();
  const [index, setIndex] = useState<CourseIndex | null>(null);
  const progress = useProgress();
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchIndex().then(setIndex);
  }, []);

  if (!index) return <div className="py-32 text-center text-faint">{t('loading')}</div>;

  // Index-based count: ignore stale lesson ids left in localStorage.
  const doneTotal = index.phases.reduce(
    (acc, p) => acc + p.lessons.filter((l) => progress.lessons[`${p.slug}/${l.slug}`]?.done).length,
    0,
  );
  const pct = Math.round((doneTotal / index.stats.lessons) * 100);

  const quizEntries = Object.values(progress.lessons).filter((l) => l.postTotal);
  const quizAvg =
    quizEntries.length > 0
      ? Math.round(
          (quizEntries.reduce((acc, l) => acc + (l.postScore ?? 0) / (l.postTotal ?? 1), 0) /
            quizEntries.length) *
            100,
        )
      : null;

  const zh = lang === 'zh';
  const certName =
    profile?.display_name ?? profile?.email?.split('@')[0] ?? (zh ? '学习者' : 'Learner');
  const certDate = new Date().toLocaleDateString(zh ? 'zh-CN' : 'en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
  const completedPhases = index.phases.filter(
    (p) => p.lessons.length > 0 && p.lessons.every((l) => progress.lessons[`${p.slug}/${l.slug}`]?.done),
  );
  const dlCert = async (achievement: string, file: string) => {
    try {
      downloadBlob(await generateCertificate({ name: certName, achievement, date: certDate, zh }), file);
    } catch {
      alert(zh ? '证书生成失败,请重试或换个浏览器' : 'Certificate generation failed — try again or another browser');
    }
  };
  const dlShare = async () => {
    try {
      const badges = computeBadges(progress, index).filter((b) => b.unlocked).length;
      downloadBlob(
        await generateShareCard({
          name: certName,
          doneCount: doneTotal,
          totalLessons: index.stats.lessons,
          streak: streakDays(),
          badges,
          zh,
        }),
        'ai-eng-share.png',
      );
    } catch {
      alert(zh ? '分享卡生成失败,请重试' : 'Share card generation failed — try again');
    }
  };

  const download = () => {
    const blob = new Blob([exportProgress()], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ai-eng-studio-progress.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const upload = (file: File) => {
    file.text().then((text) => {
      if (!importProgress(text)) alert(lang === 'zh' ? '导入失败：文件格式不正确' : 'Import failed: bad format');
    });
  };

  return (
    <div className="mx-auto max-w-4xl px-5 py-14">
      <header className="flex flex-wrap items-end justify-between gap-6 border-b border-hairline pb-8">
        <div>
          <h1 className="font-serif text-[38px] font-semibold tracking-tight">{t('progress_title')}</h1>
          <div className="mt-4 flex items-baseline gap-8">
            <Big n={`${pct}%`} label={t('progress_overall')} />
            <Big n={`${doneTotal}/${index.stats.lessons}`} label={t('lessons')} />
            <Big n={`${streakDays()}`} label={`${t('streak')} (${t('days')})`} />
            {quizAvg != null && <Big n={`${quizAvg}`} label={t('quiz_avg')} />}
          </div>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={download}
            className="rounded-md border border-hairline bg-paper px-3 py-1.5 text-[13px] text-faint transition-colors hover:bg-bone hover:text-ink"
          >
            {t('export')}
          </button>
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            className="rounded-md border border-hairline bg-paper px-3 py-1.5 text-[13px] text-faint transition-colors hover:bg-bone hover:text-ink"
          >
            {t('import')}
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="application/json"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
          />
        </div>
      </header>

      {/* activity heatmap */}
      <section className="mt-10">
        <div className="mb-4 font-mono text-[10.5px] uppercase tracking-[0.16em] text-faint">
          {lang === 'zh' ? '学习日历 · 近半年' : 'Activity · last 26 weeks'}
        </div>
        <Heatmap counts={dailyCompletions(progress)} lang={lang} />
      </section>

      {/* achievement badges */}
      <section className="mt-12">
        <BadgeWall badges={computeBadges(progress, index)} lang={lang} />
      </section>

      {/* certificates & sharing */}
      <section className="mt-12">
        <div className="mb-3 font-mono text-[10.5px] uppercase tracking-[0.16em] text-faint">
          {zh ? '证书与分享' : 'Certificates & sharing'}
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={dlShare}
            className="rounded-md border border-hairline bg-paper px-3 py-1.5 text-[13px] text-faint transition-colors hover:bg-bone hover:text-ink"
          >
            {zh ? '下载学习成就卡' : 'Achievement card'}
          </button>
          {doneTotal === index.stats.lessons && (
            <button
              type="button"
              onClick={() => dlCert(zh ? '完成全部 499 节课程' : 'completed all 499 lessons', 'certificate-full.png')}
              className="rounded-md border border-ink-green/30 bg-pale-green px-3 py-1.5 text-[13px] text-ink-green transition-colors hover:bg-pale-green/70"
            >
              {zh ? '结业证书' : 'Full certificate'}
            </button>
          )}
          {completedPhases.map((p) => (
            <button
              key={p.slug}
              type="button"
              onClick={() =>
                dlCert(
                  zh
                    ? `完成「${p.titleZh}」阶段（${p.lessons.length} 课）`
                    : `completed ${p.titleEn} (${p.lessons.length} lessons)`,
                  `certificate-phase-${p.num}.png`,
                )
              }
              className="rounded-md border border-hairline bg-paper px-3 py-1.5 text-[13px] text-faint transition-colors hover:bg-bone hover:text-ink"
            >
              {zh ? `阶段 ${p.num} 证书` : `Phase ${p.num} cert`}
            </button>
          ))}
        </div>
        {completedPhases.length === 0 && doneTotal < index.stats.lessons && (
          <p className="mt-2 text-[12.5px] text-faint">
            {zh ? '完成一个阶段的全部课程即可领取该阶段结业证书。' : 'Complete a phase to earn its certificate.'}
          </p>
        )}
      </section>

      <div className="mt-12 mb-4 font-mono text-[10.5px] uppercase tracking-[0.16em] text-faint">
        {lang === 'zh' ? '各阶段进度' : 'Per-phase progress'}
      </div>
      <ol>
        {index.phases.map((phase) => {
          const done = phase.lessons.filter(
            (l) => progress.lessons[`${phase.slug}/${l.slug}`]?.done,
          ).length;
          const phasePct = phase.lessons.length ? (done / phase.lessons.length) * 100 : 0;
          return (
            <li key={phase.slug} className="border-b border-hairline">
              <Link
                to={`/phase/${phase.slug}`}
                className="group grid grid-cols-[44px_1fr_140px_64px] items-center gap-4 py-3.5"
              >
                <span className="font-mono text-[12px] text-faint">
                  {String(phase.num).padStart(2, '0')}
                </span>
                <span className="truncate text-[14.5px] font-medium group-hover:underline group-hover:decoration-hairline group-hover:underline-offset-4">
                  {phaseTitle(phase, lang)}
                </span>
                <span className="h-[3px] overflow-hidden rounded-full bg-bone">
                  <span
                    className="block h-full rounded-full bg-ink-green"
                    style={{ width: `${phasePct}%` }}
                  />
                </span>
                <span className="text-right font-mono text-[11.5px] text-faint">
                  {done}/{phase.lessons.length}
                </span>
              </Link>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function Big({ n, label }: { n: string; label: string }) {
  return (
    <div>
      <div className="font-serif text-[30px] font-semibold leading-none">{n}</div>
      <div className="mt-1 text-[12px] text-faint">{label}</div>
    </div>
  );
}
