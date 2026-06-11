import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { fetchIndex } from '../lib/data';
import { phaseTitle, useLang } from '../lib/i18n';
import {
  AREAS,
  QUESTIONS,
  clearPlacement,
  entryPhase,
  loadPlacement,
  phaseStatus,
  savePlacement,
  type PlacementResult,
} from '../lib/placement';
import type { CourseIndex } from '../lib/types';

const LETTERS = ['A', 'B', 'C', 'D'];

export function PlacementPage() {
  const { lang } = useLang();
  const zh = lang === 'zh';
  const [index, setIndex] = useState<CourseIndex | null>(null);
  const [round, setRound] = useState(0); // 0..4
  const [answers, setAnswers] = useState<(number | null)[]>(() => QUESTIONS.map(() => null));
  const [result, setResult] = useState<PlacementResult | null>(() => loadPlacement());

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
    setResult(res);
  };

  if (result && index) return <ResultView result={result} index={index} onRetake={() => {
    clearPlacement();
    setResult(null);
    setRound(0);
    setAnswers(QUESTIONS.map(() => null));
  }} />;

  const qa = round * 2;
  const roundQuestions = [qa, qa + 1];
  const roundDone = roundQuestions.every((i) => answers[i] != null);
  const roundScore = roundQuestions.filter((i) => answers[i] === QUESTIONS[i].correct).length;
  const area = AREAS[round];

  return (
    <div className="mx-auto max-w-2xl px-5 py-14">
      <p className="font-mono text-[11px] tracking-[0.18em] text-faint">FIND YOUR LEVEL</p>
      <h1 className="mt-2 font-serif text-[34px] font-semibold tracking-tight">
        {zh ? '找到你的起点' : 'Find your level'}
      </h1>
      <p className="mt-2 text-[14.5px] leading-relaxed text-faint">
        {zh
          ? '10 道题、5 个知识领域。根据得分把你映射到合适的起始阶段，并生成带工时估算的个性化学习路径。答完才揭晓对错。'
          : 'Ten questions across five areas. Your score maps to a starting phase and a personalized path with hour estimates.'}
      </p>

      {/* round progress */}
      <div className="mt-8 flex items-center gap-2">
        {AREAS.map((a, i) => (
          <div key={a.key} className="flex items-center gap-2">
            <span
              className={`rounded-full px-2.5 py-0.5 text-[11px] ${
                i < round
                  ? 'bg-pale-green text-ink-green'
                  : i === round
                    ? 'bg-ink text-white'
                    : 'border border-hairline bg-paper text-faint'
              }`}
            >
              {zh ? a.zh : a.en}
            </span>
            {i < AREAS.length - 1 && <span className="h-px w-3 bg-hairline" />}
          </div>
        ))}
      </div>

      <div className="mt-6 space-y-6">
        {roundQuestions.map((qi) => {
          const q = QUESTIONS[qi];
          const chosen = answers[qi];
          return (
            <section key={qi} className="rounded-lg border border-hairline bg-paper px-6 py-5">
              <p className="font-medium leading-relaxed">
                <span className="mr-2 font-mono text-[12px] text-faint">Q{qi + 1}.</span>
                {zh ? q.zh : q.en}
              </p>
              <div className="mt-3 space-y-1.5">
                {(zh ? q.optionsZh : q.optionsEn).map((opt, oi) => (
                  <button
                    key={oi}
                    type="button"
                    onClick={() => setAnswers((prev) => prev.map((a, i) => (i === qi ? oi : a)))}
                    className={`flex w-full items-start gap-3 rounded-md border px-4 py-2.5 text-left text-[14.5px] leading-relaxed transition-colors ${
                      chosen === oi
                        ? 'border-ink bg-bone'
                        : 'border-hairline bg-paper hover:bg-bone'
                    }`}
                  >
                    <span className="mt-0.5 font-mono text-[12px] text-faint">{LETTERS[oi]}</span>
                    <span className="flex-1">{opt}</span>
                  </button>
                ))}
              </div>
            </section>
          );
        })}
      </div>

      <div className="mt-6 flex items-center justify-between">
        <span className="font-mono text-[12px] text-faint">
          {zh ? `第 ${round + 1} / 5 轮` : `Round ${round + 1} / 5`}
          {roundDone &&
            ` · ${zh ? area.zh : area.en}: ${roundScore}/2`}
        </span>
        <button
          type="button"
          disabled={!roundDone}
          onClick={() => (round < 4 ? setRound(round + 1) : finish(answers))}
          className="rounded-md bg-ink px-5 py-2 text-[14px] text-white transition-colors hover:bg-ink/85 disabled:opacity-40"
        >
          {round < 4 ? (zh ? '下一轮' : 'Next round') : zh ? '查看结果' : 'See results'}
        </button>
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
                    className={`h-full rounded-full ${s === 2 ? 'bg-ink-green' : s === 1 ? 'bg-ink-yellow' : 'bg-ink-red/60'}`}
                    style={{ width: `${(s / 2) * 100}%` }}
                  />
                </div>
                <span className="text-right font-mono text-[12px] text-faint">{s}/2</span>
              </div>
            );
          })}
        </div>
        <div className="mt-4 flex items-baseline justify-between border-t border-hairline pt-4">
          <span className="text-[13.5px] text-faint">{zh ? '总分' : 'Total'}</span>
          <span className="font-serif text-[28px] font-semibold">{result.total}/10</span>
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
