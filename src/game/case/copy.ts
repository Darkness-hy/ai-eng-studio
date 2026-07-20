import type { ResearchCase } from './types';
import { caseChecklist } from './checklist';

function passedReview(c: ResearchCase): boolean {
  return c.reviewRounds.some((r) => r.status === 'passed');
}

/** Next investigative step in case language (not task-list language). */
export function nextCaseHint(c: ResearchCase | null, zh: boolean): string {
  if (!c) {
    return zh ? '尚未立案：先打开案件台，接下这起异常。' : 'No active case — open the desk and accept the bulletin.';
  }
  if (c.status === 'closed') {
    return zh ? '本案已结案。可重置再侦，或导出稿归档。' : 'Case closed. Reset to investigate again, or keep the L1 report.';
  }
  if (passedReview(c)) {
    return zh ? '审稿已过：导出 L1 结案卷宗即可收工。' : 'Review passed — export the L1 dossier to close.';
  }
  const items = caseChecklist(c);
  const first = items.find((i) => !i.ok);
  if (!first) {
    return zh ? '卷宗已齐：送审，看审稿人挑哪条洞。' : 'Dossier is ready — submit for review.';
  }
  const map: Record<string, [string, string]> = {
    question: ['矛盾还没写成问题——用一句话钉住你在查什么。', 'Name the contradiction as a research question.'],
    evidence: ['现场证据不够：海报与复现日志至少都要钉上。', 'Not enough field evidence — pin poster and repro log at least.'],
    hypothesis: ['假设还太飘：写成可操作、可被实验打脸的句子。', 'Hypothesis is still vague — make it operational and falsifiable.'],
    experiment: ['还缺对照验证：去机房跑清洁切分（或模拟）。', 'Still need a control run at the compute bench.'],
    results: ['实验跑了，结果摘要还没写进卷宗。', 'Experiment exists, but results summary is empty.'],
    limitations: ['还没写下局限——审稿人最爱抓这个。', 'Add at least one limitation before review.'],
    claims: ['有主张悬空：补 supports 边，或撤回站不住的 claim。', 'Dangling claims — add supports or retract.'],
    review: ['预检未过：按清单补洞后再送审。', 'Preview still fails — fix checklist items, then submit.'],
  };
  const pair = map[first.id] ?? [first.zh, first.en];
  return zh ? pair[0] : pair[1];
}

export function completenessLabel(zh: boolean): string {
  return zh ? '卷宗完整度' : 'Dossier completeness';
}

export function contradictionOpenLabel(zh: boolean, open: boolean): string {
  if (open) return zh ? '矛盾未解' : 'Contradiction open';
  return zh ? '矛盾已收敛' : 'Contradiction resolved';
}

export function isContradictionOpen(c: ResearchCase): boolean {
  if (c.status === 'closed' || passedReview(c)) return false;
  return caseChecklist(c).some((i) => !i.ok);
}
