/**
 * Pure logic smoke for Case Detective v0 (no browser).
 * Run: npx --yes tsx scripts/check-case-detective-v0.ts
 */
import { computeCompleteness, danglingClaimIds, hasExperiment } from '../src/game/case/completeness';
import { createDemoCaseShell, DEMO_EVIDENCE_CATALOG } from '../src/game/case/demoCase';
import { evaluateReview } from '../src/game/case/reviewRules';
import { buildL1Markdown } from '../src/game/case/L1Export';
import type { ResearchCase } from '../src/game/case/types';

function fail(msg: string): never {
  console.error('FAIL:', msg);
  process.exit(1);
}

function main() {
  const now = new Date().toISOString();
  let c: ResearchCase = createDemoCaseShell(now);

  // Empty dossier must fail review
  let findings = evaluateReview(c);
  if (findings.length === 0) fail('empty case should not pass review');

  // Discover core evidence
  for (const id of ['ev_poster_a', 'ev_repro_fail', 'ev_split_leak', 'ev_exp_01'] as const) {
    const cat = DEMO_EVIDENCE_CATALOG[id];
    c.evidence.push({ ...cat, discoveredAt: now });
  }
  c.experimentRuns.push({
    id: 'sim_test',
    label: 'sim',
    summary: 'gap vanishes',
    linkedAt: now,
    simulated: true,
  });
  c.researchQuestion =
    '在公开海报宣称方法甲显著优于方法乙、而独立复现失败的前提下，该差距更可能来自模型本身，还是来自有偏的评估协议？';
  c.hypothesis = {
    text: '严格隔离测试后优势消失，说明协议有偏。',
    maturity: 'operationalized',
  };
  c.resultsSummary = '清洁切分下甲乙接近。';
  c.limitations = ['模板与 seed 有限'];
  c.edges = [
    { id: 'e1', from: 'ev_poster_a', to: 'cl_method_a_magic', type: 'supports' },
    { id: 'e2', from: 'ev_repro_fail', to: 'cl_method_a_magic', type: 'contradicts' },
    { id: 'e3', from: 'ev_split_leak', to: 'cl_protocol_bias', type: 'supports' },
    { id: 'e4', from: 'ev_exp_01', to: 'cl_protocol_bias', type: 'supports' },
    { id: 'e5', from: 'ev_exp_01', to: 'cl_method_a_magic', type: 'contradicts' },
  ];
  // Overclaim: magic still active
  findings = evaluateReview(c);
  if (!findings.some((f) => f.code === 'overclaim')) fail('expected overclaim finding');

  c.claims = c.claims.map((cl) =>
    cl.id === 'cl_method_a_magic' ? { ...cl, state: 'retracted' } : cl,
  );
  findings = evaluateReview(c);
  if (findings.length !== 0) {
    fail(`expected pass after retract, got ${findings.map((f) => f.code).join(',')}`);
  }
  if (!hasExperiment(c)) fail('hasExperiment');
  if (danglingClaimIds(c).length !== 0) fail('dangling');
  const comp = computeCompleteness(c);
  if (comp < 0.85) fail(`completeness ${comp} < 0.85`);

  const md = buildL1Markdown(c, 'zh');
  if (!md.includes('## 1. 问题') || !md.includes(c.researchQuestion.slice(0, 12))) {
    fail('L1 markdown missing sections');
  }

  console.log('OK case-detective-v0 smoke');
  console.log(JSON.stringify({ completeness: comp, reviewFindings: findings.length, mdChars: md.length }));
}

main();
