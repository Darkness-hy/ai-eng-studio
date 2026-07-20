import { danglingClaimIds, hasExperiment, relatedWorkCount } from './completeness';
import { evaluateReview } from './reviewRules';
import type { ResearchCase } from './types';

export type ChecklistItem = {
  id: string;
  ok: boolean;
  zh: string;
  en: string;
};

/** Player-facing readiness checklist (mirrors review + completeness). */
export function caseChecklist(c: ResearchCase): ChecklistItem[] {
  return [
    {
      id: 'question',
      ok: c.researchQuestion.trim().length >= 20,
      zh: '研究问题写清（≥20 字）',
      en: 'Research question clear (≥20 chars)',
    },
    {
      id: 'evidence',
      ok: relatedWorkCount(c) >= 2,
      zh: '对立证据 ≥ 2（海报/日志/观察等）',
      en: '≥2 opposing/related evidence items',
    },
    {
      id: 'hypothesis',
      ok: Boolean(c.hypothesis.text.trim()) && c.hypothesis.maturity !== 'wild_guess',
      zh: '假设已写且非「猜测」',
      en: 'Hypothesis set and not a wild guess',
    },
    {
      id: 'experiment',
      ok: hasExperiment(c),
      zh: '已挂对照实验/验证',
      en: 'Experiment or verification linked',
    },
    {
      id: 'results',
      ok: Boolean(c.resultsSummary.trim()),
      zh: '结果摘要非空',
      en: 'Results summary present',
    },
    {
      id: 'limitations',
      ok: c.limitations.some((x) => x.trim()),
      zh: '至少一条局限',
      en: 'At least one limitation',
    },
    {
      id: 'claims',
      ok: danglingClaimIds(c).length === 0 && c.claims.some((cl) => cl.state === 'active'),
      zh: '无悬空主张（active claim 有 supports）',
      en: 'No dangling active claims',
    },
    {
      id: 'review',
      ok: evaluateReview(c).length === 0,
      zh: '预检：可过审稿规则',
      en: 'Preview: would pass review rules',
    },
  ];
}
