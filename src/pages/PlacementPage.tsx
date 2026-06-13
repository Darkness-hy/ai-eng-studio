import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../lib/auth';
import { fetchIndex } from '../lib/data';
import { phaseTitle, useLang } from '../lib/i18n';
import {
  AREAS,
  QUESTIONS,
  QUESTIONS_PER_AREA,
  clearPlacement,
  deletePlacementCloud,
  entryPhase,
  phaseStatus,
  pushPlacementCloud,
  savePlacement,
  usePlacement,
  type PlacementResult,
} from '../lib/placement';
import type { CourseIndex } from '../lib/types';

const LETTERS = ['A', 'B', 'C', 'D'];

export function PlacementPage() {
  const { lang } = useLang();
  const zh = lang === 'zh';
  const { profile } = useAuth();
  const [index, setIndex] = useState<CourseIndex | null>(null);
  const [current, setCurrent] = useState(0); // 0..49
  const [confirming, setConfirming] = useState(false);
  const [answers, setAnswers] = useState<(number | null)[]>(() => QUESTIONS.map(() => null));
  const result = usePlacement();

  useEffect(() => {
    fetchIndex().then(setIndex);
  }, []);

  const finish = (finalAnswers: (number | null)[]) => {
    const areaScores: Record<string, number> = {};
    QUESTIONS.forEach((q, i) => {
      areaScores[q.area] = (areaScores[q.area] ?? 0) + (finalAnswers[i] === q.correct ? 1 : 0);
    });
    const total = Object.values(areaScores).reduce((a, b) => a + b, 0);
    const res: PlacementResult = {
      v: 1,
      answers: finalAnswers.map((a) => a ?? -1),
      areaScores,
      total,
      entry: entryPhase(total),
      date: new Date().toISOString(),
    };
    savePlacement(res);
    if (profile) void pushPlacementCloud(profile.id);
  };

  if (result && index) return <ResultView result={result} index={index} onRetake={() => {
    clearPlacement();
    if (profile) void deletePlacementCloud(profile.id);
    setCurrent(0);
    setConfirming(false);
    setAnswers(QUESTIONS.map(() => null));
  }} />;

  const total = QUESTIONS.length;
  const answeredCount = answers.filter((a) => a != null).length;
  const q = QUESTIONS[current];
  const chosen = answers[current];
  const curArea = AREAS[Math.floor(current / QUESTIONS_PER_AREA)];
  const go = (i: number) => setCurrent(Math.max(0, Math.min(total - 1, i)));
  const setAnswer = (oi: number | null) =>
    setAnswers((prev) => prev.map((a, i) => (i === current ? oi : a)));
  const submit = () => (answeredCount === total ? finish(answers) : setConfirming(true));

  return (
    <div className="mx-auto max-w-2xl px-5 py-14">
      <p className="font-mono text-[11px] tracking-[0.18em] text-faint">FIND YOUR LEVEL</p>
      <h1 className="mt-2 font-serif text-[34px] font-semibold tracking-tight">
        {zh ? '找到你的起点' : 'Find your level'}
      </h1>
      <p className="mt-2 text-[14.5px] leading-relaxed text-faint">
        {zh
          ? '50 道题、5 个知识领域。逐题作答，可随时跳过或在答题卡上跳转，最后点「提交交卷」。未作答的题计为错误。'
          : 'Fifty questions across five areas. Answer one at a time, skip freely or jump via the answer sheet, then submit. Unanswered questions count as wrong.'}
      </p>

      {/* answer sheet */}
      <div className="mt-8 rounded-lg border border-hairline bg-paper px-5 py-4">
        <div className="mb-3 flex items-baseline justify-between">
          <span className="font-mono text-[10.5px] tracking-[0.14em] text-faint">
            {zh ? '答题卡' : 'ANSWER SHEET'}
          </span>
          <span className="font-mono text-[11.5px] text-faint">
            {zh ? `已答 ${answeredCount} / ${total}` : `${answeredCount} / ${total} answered`}
          </span>
        </div>
        <div className="space-y-2">
          {AREAS.map((a, ai) => (
            <div key={a.key} className="flex items-center gap-3">
              <span className="w-28 shrink-0 text-[11.5px] text-faint">{zh ? a.zh : a.en}</span>
              <div className="flex flex-wrap gap-1">
                {Array.from({ length: QUESTIONS_PER_AREA }, (_, k) => {
                  const idx = ai * QUESTIONS_PER_AREA + k;
                  const answered = answers[idx] != null;
                  const isCur = idx === current;
                  return (
                    <button
                      key={idx}
                      type="button"
                      onClick={() => go(idx)}
                      className={`h-6 w-6 rounded font-mono text-[10px] transition-colors ${
                        isCur
                          ? 'bg-ink text-white'
                          : answered
                            ? 'bg-pale-green text-ink-green hover:bg-pale-green/70'
                            : 'border border-hairline bg-paper text-faint hover:bg-bone'
                      }`}
                    >
                      {idx + 1}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* current question */}
      <section className="mt-6 rounded-lg border border-hairline bg-paper px-6 py-5">
        <div className="flex items-baseline justify-between">
          <span className="font-mono text-[12px] text-faint">
            Q{current + 1} / {total} · {zh ? curArea.zh : curArea.en}
          </span>
          {chosen != null && (
            <button
              type="button"
              onClick={() => setAnswer(null)}
              className="font-mono text-[11px] text-faint underline decoration-hairline underline-offset-2 hover:text-ink"
            >
              {zh ? '清除本题' : 'Clear'}
            </button>
          )}
        </div>
        <p className="mt-2 font-medium leading-relaxed">{zh ? q.zh : q.en}</p>
        <div className="mt-3 space-y-1.5">
          {(zh ? q.optionsZh : q.optionsEn).map((opt, oi) => (
            <button
              key={oi}
              type="button"
              onClick={() => setAnswer(oi)}
              className={`flex w-full items-start gap-3 rounded-md border px-4 py-2.5 text-left text-[14.5px] leading-relaxed transition-colors ${
                chosen === oi ? 'border-ink bg-bone' : 'border-hairline bg-paper hover:bg-bone'
              }`}
            >
              <span className="mt-0.5 font-mono text-[12px] text-faint">{LETTERS[oi]}</span>
              <span className="flex-1">{opt}</span>
            </button>
          ))}
        </div>
      </section>

      {/* navigation */}
      <div className="mt-4 flex items-center justify-between">
        <button
          type="button"
          disabled={current === 0}
          onClick={() => go(current - 1)}
          className="rounded-md border border-hairline px-4 py-2 text-[13px] text-faint transition-colors hover:bg-bone hover:text-ink disabled:opacity-40"
        >
          ← {zh ? '上一题' : 'Prev'}
        </button>
        <button
          type="button"
          disabled={current === total - 1}
          onClick={() => go(current + 1)}
          className="rounded-md border border-hairline px-4 py-2 text-[13px] text-faint transition-colors hover:bg-bone hover:text-ink disabled:opacity-40"
        >
          {chosen != null ? (zh ? '下一题' : 'Next') : zh ? '跳过' : 'Skip'} →
        </button>
      </div>

      {/* submit */}
      <div className="mt-6 border-t border-hairline pt-6">
        {confirming ? (
          <div className="rounded-lg border border-ink-yellow/30 bg-pale-yellow px-5 py-4">
            <p className="text-[14px] leading-relaxed text-ink-yellow">
              {zh
                ? `还有 ${total - answeredCount} 题未作答，未作答将计为错误。确认提交？`
                : `${total - answeredCount} question(s) unanswered will be marked wrong. Submit anyway?`}
            </p>
            <div className="mt-3 flex gap-2">
              <button
                type="button"
                onClick={() => finish(answers)}
                className="rounded-md bg-ink px-4 py-2 text-[13px] text-white transition-colors hover:bg-ink/85"
              >
                {zh ? '确认提交' : 'Submit anyway'}
              </button>
              <button
                type="button"
                onClick={() => setConfirming(false)}
                className="rounded-md border border-hairline px-4 py-2 text-[13px] text-faint transition-colors hover:bg-bone hover:text-ink"
              >
                {zh ? '继续作答' : 'Keep going'}
              </button>
            </div>
          </div>
        ) : (
          <button
            type="button"
            onClick={submit}
            className="w-full rounded-md bg-ink px-5 py-3 text-[15px] text-white transition-colors hover:bg-ink/85"
          >
            {zh ? `提交交卷（已答 ${answeredCount} / ${total}）` : `Submit (${answeredCount} / ${total} answered)`}
          </button>
        )}
      </div>
    </div>
  );
}

function ResultView({
  result,
  index,
  onRetake,
}: {
  result: PlacementResult;
  index: CourseIndex;
  onRetake: () => void;
}) {
  const { lang } = useLang();
  const zh = lang === 'zh';
  const entry = index.phases.find((p) => p.num === result.entry);

  const rows = index.phases.map((p) => {
    const status = phaseStatus(p.num, result.entry, result.areaScores);
    return { phase: p, status, hours: status === 'skip' ? null : p.hours };
  });
  const totalHours = rows.reduce((acc, r) => acc + (r.hours ?? 0), 0);
  const doCount = rows.filter((r) => r.status !== 'skip').length;
  const weakest = AREAS.reduce((min, a) =>
    (result.areaScores[a.key] ?? 0) < (result.areaScores[min.key] ?? 0) ? a : min,
  );

  const statusLabel = { skip: zh ? '跳过' : 'Skip', review: zh ? '复习' : 'Review', do: zh ? '学习' : 'Do' };
  const statusCls = {
    skip: 'text-faint',
    review: 'bg-pale-yellow text-ink-yellow',
    do: 'bg-pale-green text-ink-green',
  };

  return (
    <div className="mx-auto max-w-2xl px-5 py-14">
      <p className="font-mono text-[11px] tracking-[0.18em] text-faint">FIND YOUR LEVEL</p>
      <h1 className="mt-2 font-serif text-[34px] font-semibold tracking-tight">
        {zh ? '你的定级结果' : 'Your placement'}
      </h1>

      {/* breakdown */}
      <section className="mt-8 rounded-lg border border-hairline bg-paper px-6 py-5">
        <div className="space-y-2.5">
          {AREAS.map((a) => {
            const s = result.areaScores[a.key] ?? 0;
            return (
              <div key={a.key} className="grid grid-cols-[160px_1fr_40px] items-center gap-3">
                <span className="text-[13.5px]">{zh ? a.zh : a.en}</span>
                <div className="h-[5px] overflow-hidden rounded-full bg-bone">
                  <div
                    className={`h-full rounded-full ${s >= 8 ? 'bg-ink-green' : s >= 4 ? 'bg-ink-yellow' : 'bg-ink-red/60'}`}
                    style={{ width: `${(s / QUESTIONS_PER_AREA) * 100}%` }}
                  />
                </div>
                <span className="text-right font-mono text-[12px] text-faint">{s}/{QUESTIONS_PER_AREA}</span>
              </div>
            );
          })}
        </div>
        <div className="mt-4 flex items-baseline justify-between border-t border-hairline pt-4">
          <span className="text-[13.5px] text-faint">{zh ? '总分' : 'Total'}</span>
          <span className="font-serif text-[28px] font-semibold">
            {result.total}/{AREAS.length * QUESTIONS_PER_AREA}
          </span>
        </div>
      </section>

      {/* entry point */}
      {entry && (
        <Link
          to={`/phase/${entry.slug}`}
          className="mt-4 block rounded-lg border border-ink-green/30 bg-pale-green px-6 py-5 transition-all hover:-translate-y-0.5 hover:shadow-lift"
        >
          <div className="font-mono text-[10.5px] tracking-[0.16em] text-ink-green">
            {zh ? '建议起点' : 'START HERE'}
          </div>
          <div className="mt-1 font-serif text-[24px] font-semibold text-ink-green">
            {zh ? `阶段 ${entry.num} · ${entry.titleZh}` : `Phase ${entry.num} · ${entry.titleEn}`}
          </div>
          <div className="mt-1 text-[13.5px] text-ink-green/80">
            {zh
              ? `个性化路径：${doCount} 个阶段，约 ${totalHours} 小时。优先补强最弱领域「${weakest.zh}」。`
              : `Your path: ~${totalHours} hours across ${doCount} phases. Focus first on ${weakest.en}.`}
          </div>
        </Link>
      )}

      {/* path table */}
      <section className="mt-4 overflow-hidden rounded-lg border border-hairline bg-paper">
        <table className="w-full text-[13.5px]">
          <thead>
            <tr className="border-b border-hairline font-mono text-[10.5px] uppercase tracking-[0.1em] text-faint">
              <th className="px-4 py-2.5 text-left font-medium">{zh ? '阶段' : 'Phase'}</th>
              <th className="px-4 py-2.5 text-left font-medium">{zh ? '名称' : 'Name'}</th>
              <th className="px-4 py-2.5 text-left font-medium">{zh ? '状态' : 'Status'}</th>
              <th className="px-4 py-2.5 text-right font-medium">{zh ? '预计小时' : 'Est. hours'}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ phase, status, hours }) => (
              <tr key={phase.slug} className={`border-b border-hairline last:border-0 ${status === 'skip' ? 'opacity-50' : ''}`}>
                <td className="px-4 py-2 font-mono text-[12px] text-faint">
                  {String(phase.num).padStart(2, '0')}
                </td>
                <td className="px-4 py-2">
                  <Link to={`/phase/${phase.slug}`} className="hover:underline">
                    {phaseTitle(phase, lang)}
                  </Link>
                </td>
                <td className="px-4 py-2">
                  <span className={`rounded-full px-2 py-0.5 text-[11px] ${statusCls[status]}`}>
                    {statusLabel[status]}
                  </span>
                </td>
                <td className="px-4 py-2 text-right font-mono text-[12px] text-faint">
                  {hours ?? '--'}
                </td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr className="bg-bone/50 font-medium">
              <td colSpan={3} className="px-4 py-2.5 text-[13px]">
                {zh ? '个性化路径合计' : 'Personalized total'}
              </td>
              <td className="px-4 py-2.5 text-right font-mono text-[13px]">~{totalHours}h</td>
            </tr>
          </tfoot>
        </table>
      </section>

      <div className="mt-6 text-center">
        <button
          type="button"
          onClick={onRetake}
          className="font-mono text-[12px] text-faint underline decoration-hairline underline-offset-2 hover:text-ink"
        >
          {zh ? '重新测试' : 'Retake the quiz'}
        </button>
      </div>
    </div>
  );
}
