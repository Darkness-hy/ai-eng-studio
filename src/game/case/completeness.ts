import type { ResearchCase } from './types';

export function relatedWorkCount(c: ResearchCase): number {
  return c.evidence.filter((e) => e.kind === 'citation' || e.kind === 'testimony' || e.kind === 'observation' || e.kind === 'log')
    .length;
}

export function hasExperiment(c: ResearchCase): boolean {
  return (
    c.experimentRuns.length > 0 ||
    c.evidence.some((e) => e.kind === 'experiment_result')
  );
}

export function danglingClaimIds(c: ResearchCase): string[] {
  const active = c.claims.filter((cl) => cl.state === 'active');
  return active
    .filter((cl) => !c.edges.some((e) => e.type === 'supports' && e.to === cl.id))
    .map((cl) => cl.id);
}

/** 0–1 completeness per v0 design doc. */
export function computeCompleteness(c: ResearchCase): number {
  const q = c.researchQuestion.trim().length >= 20 ? 1 : c.researchQuestion.trim() ? 0.4 : 0;
  const rw = Math.min(1, relatedWorkCount(c) / 2);
  const hyp =
    c.hypothesis.maturity === 'operationalized'
      ? 1
      : c.hypothesis.maturity === 'plausible'
        ? 0.7
        : c.hypothesis.text.trim()
          ? 0.35
          : 0;
  const exp = hasExperiment(c) ? 1 : 0;
  const res = c.resultsSummary.trim() ? 1 : 0;
  const lim = c.limitations.some((x) => x.trim()) ? 1 : 0;
  const dang = danglingClaimIds(c).length === 0 && c.claims.some((cl) => cl.state === 'active') ? 1 : 0;

  return (
    0.15 * q +
    0.15 * rw +
    0.15 * hyp +
    0.2 * exp +
    0.15 * res +
    0.1 * lim +
    0.1 * dang
  );
}

export function isReadyForReview(c: ResearchCase): boolean {
  return computeCompleteness(c) >= 0.85;
}
