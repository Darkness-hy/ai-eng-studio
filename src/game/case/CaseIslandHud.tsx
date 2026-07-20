import { ClipboardList, X } from 'lucide-react';
import { Link } from 'react-router-dom';
import { caseStatusLabel } from './caseStore';
import {
  completenessLabel,
  contradictionOpenLabel,
  isContradictionOpen,
  nextCaseHint,
} from './copy';
import type { ResearchCase } from './types';
import { useLang } from '../../lib/i18n';

/** Compact case status chip for the 3D/2D island HUD. */
export function CaseIslandHud({
  caseFile,
  onOpenMentor,
  onOpenDesk,
}: {
  caseFile: ResearchCase | null;
  onOpenMentor?: () => void;
  /** Prefer in-world overlay over hard navigation (avoids white full-page jump). */
  onOpenDesk?: () => void;
}) {
  const { lang } = useLang();
  const zh = lang === 'zh';
  const hint = nextCaseHint(caseFile, zh);
  const deskBtnClass =
    'mt-2 inline-flex min-h-10 items-center rounded border border-white/20 bg-white/5 px-3 text-[14px] text-white/90 hover:bg-white/10';

  if (!caseFile) {
    return (
      <div className="pointer-events-auto absolute bottom-20 left-3 z-20 max-w-[min(92vw,360px)] rounded-md border border-white/18 bg-[#0f1a1e]/92 px-3.5 py-3 text-white shadow-[0_12px_40px_rgba(0,0,0,0.35)] backdrop-blur-md sm:bottom-6 sm:left-4">
        <p className="font-mono text-[11px] tracking-[0.1em] text-white/55">
          {zh ? '卷宗' : 'Dossier'}
        </p>
        <p className="mt-1 text-[15px] leading-snug">{hint}</p>
        {onOpenDesk ? (
          <button type="button" className={deskBtnClass} onClick={onOpenDesk}>
            {zh ? '打开卷宗台立案' : 'Open case desk'}
          </button>
        ) : (
          <Link to="/research-game/case" className={deskBtnClass}>
            {zh ? '打开卷宗台立案' : 'Open case desk'}
          </Link>
        )}
      </div>
    );
  }

  const pct = Math.round(caseFile.completeness * 100);
  const nEv = caseFile.evidence.length;
  const open = isContradictionOpen(caseFile);

  const btn =
    'inline-flex min-h-10 items-center rounded border border-white/18 bg-white/5 px-3 text-[13px] text-white/90 hover:bg-white/10';
  return (
    <div className="pointer-events-auto absolute bottom-20 left-3 z-20 max-w-[min(92vw,380px)] rounded-md border border-white/16 bg-[#0f1a1e]/92 px-3.5 py-3 text-white shadow-[0_12px_40px_rgba(0,0,0,0.35)] backdrop-blur-md sm:bottom-6 sm:left-4">
      <div className="flex items-start gap-2.5">
        <ClipboardList size={18} className="mt-0.5 shrink-0 text-white/70" aria-hidden />
        <div className="min-w-0 flex-1">
          <p className="font-mono text-[11px] tracking-[0.1em] text-white/55">
            {zh ? '卷宗' : 'Dossier'} · {caseStatusLabel(caseFile.status, zh)} · {pct}%
          </p>
          <p className="mt-0.5 truncate font-serif text-[17px] font-semibold sm:text-[18px]">
            {zh ? caseFile.title.zh : caseFile.title.en}
          </p>
          <p className="mt-1 text-[14px] text-white/70">
            {zh ? `证据 ${nEv}` : `Evidence ${nEv}`}
            {' · '}
            <span className={open ? 'text-[#e0c98a]' : 'text-[#9fd4a8]'}>
              {contradictionOpenLabel(zh, open)}
            </span>
          </p>
          <p className="mt-1.5 text-[14px] leading-snug text-white/80">{hint}</p>
          <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/15" aria-label={completenessLabel(zh)}>
            <div className="h-full bg-[#9fd0ff]" style={{ width: `${pct}%` }} />
          </div>
          <div className="mt-2.5 flex flex-wrap gap-2">
            {onOpenDesk ? (
              <button type="button" className={btn} onClick={onOpenDesk}>
                {zh ? '打开卷宗' : 'Open dossier'}
              </button>
            ) : (
              <Link to="/research-game/case" className={btn}>
                {zh ? '打开卷宗' : 'Open dossier'}
              </Link>
            )}
            {onOpenMentor && caseFile.status !== 'closed' && (
              <button type="button" className={btn} onClick={onOpenMentor}>
                {zh ? '找导师' : 'Mentor'}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export function CaseEvidenceOverlay({
  title,
  body,
  ok,
  reason,
  onClose,
}: {
  title: string;
  body: string;
  ok: boolean;
  reason?: string;
  onClose: () => void;
}) {
  const { lang } = useLang();
  const zh = lang === 'zh';
  const reasonText =
    reason === 'soft_gate_flag'
      ? zh
        ? '信息门：先向导师说清海报与复现的矛盾。'
        : 'Soft gate: name the poster-vs-repro contradiction to the mentor.'
      : reason === 'soft_gate_evidence'
        ? zh
          ? '信息门：还缺前置证据。'
          : 'Soft gate: missing prerequisite evidence.'
        : reason === 'already_inspected'
          ? zh
            ? '这里已经取过证了。'
            : 'Already inspected.'
          : reason
            ? zh
              ? `无法检视（${reason}）`
              : `Cannot inspect (${reason})`
            : '';

  return (
    <div
      className="absolute inset-0 z-40 flex items-end justify-center bg-[#0b1418]/40 p-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] sm:items-center"
      role="dialog"
    >
      <section className="w-full max-w-lg rounded-lg border border-white/20 bg-[#17252a]/96 p-4 text-white shadow-lift backdrop-blur-md">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="font-mono text-[10px] tracking-[0.12em] text-white/55">
              {ok ? (zh ? '现场线索' : 'Field clue') : zh ? '信息门' : 'Soft gate'}
            </p>
            <h2 className="mt-1 font-serif text-[18px] font-semibold sm:text-[20px]">{title}</h2>
          </div>
          <button
            type="button"
            className="min-h-9 min-w-9 rounded p-1 text-white/70 hover:bg-white/10"
            onClick={onClose}
            aria-label={zh ? '关闭' : 'Close'}
          >
            <X size={18} />
          </button>
        </div>
        <p className="mt-3 text-[14px] leading-relaxed text-white/85">{ok ? body : reasonText || body}</p>
        <button
          type="button"
          className="mt-4 min-h-10 w-full rounded-md bg-white px-3 py-2 text-[14px] text-ink hover:bg-white/90 sm:w-auto"
          onClick={onClose}
        >
          {zh ? '继续探案' : 'Keep investigating'}
        </button>
      </section>
    </div>
  );
}

export function CaseMentorOverlay({
  flags,
  mentorStage,
  onVague,
  onClear,
  message,
  onClose,
}: {
  flags: string[];
  mentorStage: string;
  message: string | null;
  onVague: () => void;
  onClear: () => void;
  onClose: () => void;
}) {
  const { lang } = useLang();
  const zh = lang === 'zh';
  const taught = mentorStage === 'taught_controls' || mentorStage === 'done' || flags.includes('knows_leak');
  return (
    <div
      className="absolute inset-0 z-40 flex items-end justify-center bg-[#0b1418]/40 p-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] sm:items-center"
      role="dialog"
    >
      <section className="w-full max-w-lg rounded-lg border border-white/20 bg-[#17252a]/96 p-4 text-white shadow-lift">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="font-mono text-[10px] tracking-[0.12em] text-white/55">
              {zh ? '方法导师' : 'Methods mentor'} · {mentorStage}
            </p>
            <h2 className="mt-1 font-serif text-[18px] font-semibold sm:text-[20px]">
              {zh ? '当面请教' : 'Office hours'}
            </h2>
          </div>
          <button
            type="button"
            className="min-h-9 min-w-9 rounded p-1 text-white/70 hover:bg-white/10"
            onClick={onClose}
            aria-label={zh ? '关闭' : 'Close'}
          >
            <X size={18} />
          </button>
        </div>
        <p className="mt-3 text-[14px] leading-relaxed text-white/80">
          {taught
            ? zh
              ? '你已经拿到方法线索。去档案柜与机房把证据钉上；卷宗写清楚再来对表述。'
              : 'You have the methodological lead. Pin archive + bench evidence, then clean the dossier.'
            : zh
              ? '带着「海报宣称领先 vs 复现失败」的矛盾来。说不清不给关键线索。'
              : 'Bring the poster-lead vs repro-fail contradiction. No key lead until you name it.'}
        </p>
        {flags.includes('knows_leak') && (
          <p className="mt-2 font-mono text-[11px] text-[#a9efbb]">
            {zh ? '已开锁：档案柜切分草稿' : 'Unlocked: archive split draft'}
          </p>
        )}
        {message && <p className="mt-3 border-l-2 border-white/25 pl-3 text-[13.5px] leading-relaxed">{message}</p>}
        <div className="mt-4 flex flex-col gap-2 sm:flex-row sm:flex-wrap">
          {!taught && (
            <>
              <button
                type="button"
                className="min-h-10 rounded-md border border-white/25 px-3 py-2 text-[13px]"
                onClick={onVague}
              >
                {zh ? '含糊请教' : 'Vague ask'}
              </button>
              <button
                type="button"
                className="min-h-10 rounded-md bg-white px-3 py-2 text-[13px] text-ink"
                onClick={onClear}
              >
                {zh ? '说清矛盾' : 'State contradiction'}
              </button>
            </>
          )}
          {taught && (
            <button
              type="button"
              className="min-h-10 rounded-md bg-white px-3 py-2 text-[13px] text-ink"
              onClick={onClear}
            >
              {zh ? '再听一遍要点' : 'Recap the tip'}
            </button>
          )}
        </div>
      </section>
    </div>
  );
}

export function CaseExperimentOverlay({
  canRun,
  busy,
  lastDetail,
  onRun,
  onClose,
}: {
  canRun: boolean;
  busy: boolean;
  lastDetail: string | null;
  onRun: () => void;
  onClose: () => void;
}) {
  const { lang } = useLang();
  const zh = lang === 'zh';
  return (
    <div
      className="absolute inset-0 z-40 flex items-end justify-center bg-[#0b1418]/40 p-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] sm:items-center"
      role="dialog"
    >
      <section className="w-full max-w-lg rounded-lg border border-white/20 bg-[#17252a]/96 p-4 text-white shadow-lift">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="font-mono text-[10px] tracking-[0.12em] text-white/55">
              {zh ? '机房对照工位' : 'Control bench'}
            </p>
            <h2 className="mt-1 font-serif text-[18px] font-semibold sm:text-[20px]">
              {zh ? '清洁切分对照' : 'Clean-split control'}
            </h2>
          </div>
          <button
            type="button"
            className="min-h-9 min-w-9 rounded p-1 text-white/70 hover:bg-white/10"
            onClick={onClose}
            aria-label={zh ? '关闭' : 'Close'}
          >
            <X size={18} />
          </button>
        </div>
        <p className="mt-3 text-[14px] leading-relaxed text-white/80">
          {zh
            ? '在白名单模板上重跑对照。若已配置 Research API 会尝试创建真实 run；否则自动回退本地模拟，语义同构。'
            : 'Rerun the whitelist control. Uses a live ResearchRun when configured; otherwise a local simulation with the same case semantics.'}
        </p>
        {!canRun && (
          <p className="mt-2 text-[13px] text-[#f0d78d]">
            {zh
              ? '信息门：先取得「可疑切分」证据（找导师 → 档案柜）。'
              : 'Soft gate: get the suspicious-split evidence first (mentor → archive).'}
          </p>
        )}
        {lastDetail && <p className="mt-2 font-mono text-[11px] text-white/60">{lastDetail}</p>}
        <div className="mt-4 flex flex-col gap-2 sm:flex-row">
          <button
            type="button"
            disabled={!canRun || busy}
            className="min-h-10 rounded-md bg-white px-3 py-2 text-[13px] text-ink disabled:opacity-40"
            onClick={onRun}
          >
            {busy ? (zh ? '运行中…' : 'Running…') : zh ? '跑对照实验' : 'Run control'}
          </button>
          <button
            type="button"
            className="min-h-10 rounded-md border border-white/25 px-3 py-2 text-[13px]"
            onClick={onClose}
          >
            {zh ? '关闭' : 'Close'}
          </button>
        </div>
      </section>
    </div>
  );
}
