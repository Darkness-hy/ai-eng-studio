import { ClipboardList, X } from 'lucide-react';
import { Link } from 'react-router-dom';
import type { ResearchCase } from './types';

/** Compact case status chip for the 3D/2D island HUD. */
export function CaseIslandHud({
  caseFile,
  onOpenMentor,
}: {
  caseFile: ResearchCase | null;
  onOpenMentor?: () => void;
}) {
  if (!caseFile) {
    return (
      <div className="pointer-events-auto absolute bottom-20 left-3 z-20 max-w-[min(88vw,320px)] rounded-md border border-white/25 bg-[#17252a]/88 px-3 py-2 text-white shadow-lift backdrop-blur-sm sm:bottom-6 sm:left-4">
        <p className="font-mono text-[10px] tracking-[0.14em] text-white/55">CASE</p>
        <p className="mt-1 text-[13px]">尚无进行中的案件</p>
        <Link
          to="/research-game/case"
          className="mt-2 inline-flex text-[12.5px] text-[#cfe9ff] underline decoration-white/30 underline-offset-2"
        >
          打开案件台立案
        </Link>
      </div>
    );
  }

  const pct = Math.round(caseFile.completeness * 100);
  const nEv = caseFile.evidence.length;

  return (
    <div className="pointer-events-auto absolute bottom-20 left-3 z-20 max-w-[min(88vw,340px)] rounded-md border border-white/25 bg-[#17252a]/88 px-3 py-2.5 text-white shadow-lift backdrop-blur-sm sm:bottom-6 sm:left-4">
      <div className="flex items-start gap-2">
        <ClipboardList size={16} className="mt-0.5 shrink-0 text-white/70" aria-hidden />
        <div className="min-w-0 flex-1">
          <p className="font-mono text-[10px] tracking-[0.14em] text-white/55">
            CASE · {caseFile.status} · {pct}%
          </p>
          <p className="mt-0.5 truncate font-serif text-[15px] font-semibold">{caseFile.title.zh}</p>
          <p className="mt-1 text-[12px] text-white/70">
            证据 {nEv} · 审稿 {caseFile.reviewRounds.filter((r) => r.status === 'passed').length > 0 ? '已过' : '未过'}
          </p>
          <div className="mt-2 h-1 overflow-hidden rounded-full bg-white/15">
            <div className="h-full bg-[#9fd0ff]" style={{ width: `${pct}%` }} />
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            <Link
              to="/research-game/case"
              className="rounded border border-white/25 px-2 py-1 text-[11.5px] text-white/90 hover:bg-white/10"
            >
              打开案件台
            </Link>
            {onOpenMentor && caseFile.status !== 'closed' && (
              <button
                type="button"
                className="rounded border border-white/25 px-2 py-1 text-[11.5px] text-white/90 hover:bg-white/10"
                onClick={onOpenMentor}
              >
                导师门控
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
  const reasonZh =
    reason === 'soft_gate_flag'
      ? '信息门：先向导师说清海报与复现的矛盾。'
      : reason === 'soft_gate_evidence'
        ? '信息门：还缺前置证据。'
        : reason === 'already_inspected'
          ? '这里已经取过证了。'
          : reason
            ? `无法检视（${reason}）`
            : '';

  return (
    <div className="absolute inset-0 z-40 flex items-end justify-center bg-[#0b1418]/35 p-3 sm:items-center" role="dialog">
      <section className="w-full max-w-lg rounded-lg border border-white/20 bg-[#17252a]/95 p-4 text-white shadow-lift backdrop-blur-md">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="font-mono text-[10px] tracking-[0.16em] text-white/55">
              {ok ? 'CASE EVIDENCE' : 'CASE GATE'}
            </p>
            <h2 className="mt-1 font-serif text-[20px] font-semibold">{title}</h2>
          </div>
          <button type="button" className="rounded p-1 text-white/70 hover:bg-white/10" onClick={onClose} aria-label="关闭">
            <X size={18} />
          </button>
        </div>
        <p className="mt-3 text-[14px] leading-relaxed text-white/85">{ok ? body : reasonZh || body}</p>
        <button
          type="button"
          className="mt-4 rounded-md bg-white px-3 py-1.5 text-[13px] text-ink hover:bg-white/90"
          onClick={onClose}
        >
          继续探案
        </button>
      </section>
    </div>
  );
}

export function CaseMentorOverlay({
  flags,
  onVague,
  onClear,
  message,
  onClose,
}: {
  flags: string[];
  message: string | null;
  onVague: () => void;
  onClear: () => void;
  onClose: () => void;
}) {
  return (
    <div className="absolute inset-0 z-40 flex items-end justify-center bg-[#0b1418]/35 p-3 sm:items-center" role="dialog">
      <section className="w-full max-w-lg rounded-lg border border-white/20 bg-[#17252a]/95 p-4 text-white shadow-lift">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="font-mono text-[10px] tracking-[0.16em] text-white/55">MENTOR · CASE GATE</p>
            <h2 className="mt-1 font-serif text-[20px] font-semibold">方法导师</h2>
          </div>
          <button type="button" className="rounded p-1 text-white/70 hover:bg-white/10" onClick={onClose} aria-label="关闭">
            <X size={18} />
          </button>
        </div>
        <p className="mt-3 text-[14px] leading-relaxed text-white/80">
          有效互动：带着「海报宣称领先 vs 复现失败」的矛盾来。说不清不给关键线索。
        </p>
        {flags.includes('knows_leak') && (
          <p className="mt-2 font-mono text-[11px] text-[#a9efbb]">flag: knows_leak · 档案柜已可检视</p>
        )}
        {message && <p className="mt-3 border-l-2 border-white/25 pl-3 text-[13.5px] leading-relaxed">{message}</p>}
        <div className="mt-4 flex flex-wrap gap-2">
          <button type="button" className="rounded-md border border-white/25 px-3 py-1.5 text-[13px]" onClick={onVague}>
            含糊请教
          </button>
          <button type="button" className="rounded-md bg-white px-3 py-1.5 text-[13px] text-ink" onClick={onClear}>
            说清矛盾
          </button>
        </div>
      </section>
    </div>
  );
}
