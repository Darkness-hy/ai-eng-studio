import { useSyncExternalStore } from 'react';
import { computeCompleteness, isReadyForReview } from './completeness';
import { createDemoCaseShell, DEMO_CASE_ID, DEMO_EVIDENCE_CATALOG, DEMO_POIS } from './demoCase';
import { evaluateReview } from './reviewRules';
import type {
  CaseEdge,
  CasePoi,
  EdgeType,
  EvidenceNode,
  ExperimentLink,
  HypothesisMaturity,
  ResearchCase,
} from './types';

const STORAGE_KEY = 'aes:case:v0:anon';

type Listener = () => void;

let state: ResearchCase | null = load();
const listeners = new Set<Listener>();

function load(): ResearchCase | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as ResearchCase;
    if (!parsed?.id) return null;
    parsed.completeness = computeCompleteness(parsed);
    return parsed;
  } catch {
    return null;
  }
}

function persist(next: ResearchCase | null) {
  state = next;
  try {
    if (next) localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* ignore quota */
  }
  listeners.forEach((l) => l());
}

function touch(c: ResearchCase): ResearchCase {
  const updated = {
    ...c,
    updatedAt: new Date().toISOString(),
    completeness: computeCompleteness(c),
  };
  if (updated.status === 'closed') return updated;
  // After a passed review, freeze status transitions until export closes the case.
  if (!updated.reviewRounds.some((r) => r.status === 'passed')) {
    if (isReadyForReview(updated) && updated.status !== 'revision') {
      updated.status = 'ready_for_review';
    } else if (updated.status === 'briefed' && updated.evidence.length > 0) {
      updated.status = 'investigating';
    } else if (updated.status === 'ready_for_review' && !isReadyForReview(updated)) {
      updated.status = 'investigating';
    }
  }
  updated.completeness = computeCompleteness(updated);
  return updated;
}

function requireCase(): ResearchCase {
  if (!state) throw new Error('No active case');
  return state;
}

function mutate(fn: (c: ResearchCase) => ResearchCase) {
  const cur = requireCase();
  persist(touch(fn(structuredClone(cur))));
}

export function getCase(): ResearchCase | null {
  return state;
}

export function getDemoPois(): CasePoi[] {
  return DEMO_POIS;
}

export function startDemoCase(): ResearchCase {
  const c = createDemoCaseShell();
  persist(touch(c));
  return c;
}

export function resetCase(): void {
  persist(null);
}

export function discoverEvidence(evidenceId: string): EvidenceNode | null {
  const catalog = DEMO_EVIDENCE_CATALOG[evidenceId];
  if (!catalog) return null;
  let created: EvidenceNode | null = null;
  mutate((c) => {
    if (c.evidence.some((e) => e.id === evidenceId)) return c;
    created = { ...catalog, discoveredAt: new Date().toISOString() };
    c.evidence.push(created);
    return c;
  });
  return created;
}

export function canVisitPoi(poi: CasePoi, c: ResearchCase): { ok: boolean; reason?: string } {
  if (poi.requires?.flags?.length) {
    const missing = poi.requires.flags.filter((f) => !c.flags.includes(f));
    if (missing.length) return { ok: false, reason: 'soft_gate_flag' };
  }
  if (poi.requires?.evidenceIds?.length) {
    const have = new Set(c.evidence.map((e) => e.id));
    const missing = poi.requires.evidenceIds.filter((id) => !have.has(id));
    if (missing.length) return { ok: false, reason: 'soft_gate_evidence' };
  }
  if (poi.inspectOnce && poi.grantsEvidenceIds.every((id) => c.evidence.some((e) => e.id === id))) {
    if (poi.grantsEvidenceIds.length > 0) return { ok: false, reason: 'already_inspected' };
  }
  return { ok: true };
}

/** Desk-mode visit: grants evidence if allowed. Mentor handled separately. */
export function visitPoi(poiId: string): { ok: boolean; reason?: string; gained: string[] } {
  const c = requireCase();
  const poi = DEMO_POIS.find((p) => p.id === poiId);
  if (!poi) return { ok: false, reason: 'unknown_poi', gained: [] };
  const gate = canVisitPoi(poi, c);
  if (!gate.ok) return { ok: false, reason: gate.reason, gained: [] };

  const gained: string[] = [];
  for (const id of poi.grantsEvidenceIds) {
    const node = discoverEvidence(id);
    if (node) gained.push(id);
  }
  return { ok: true, gained };
}

export function talkToMentor(statedContradiction: boolean): {
  ok: boolean;
  message: { zh: string; en: string };
} {
  requireCase();
  if (!statedContradiction) {
    return {
      ok: false,
      message: {
        zh: '先说清：海报与复现之间到底哪里对不上？说不清我没法往下教。',
        en: 'First name where the poster and the reproduction disagree. I will not continue until you do.',
      },
    };
  }
  mutate((cur) => {
    if (!cur.flags.includes('knows_leak')) cur.flags.push('knows_leak');
    if (!cur.evidence.some((e) => e.id === 'ev_mentor_note')) {
      const note = DEMO_EVIDENCE_CATALOG.ev_mentor_note;
      cur.evidence.push({ ...note, discoveredAt: new Date().toISOString() });
    }
    return cur;
  });
  return {
    ok: true,
    message: {
      zh: '好。先别急着造新模型——去查评估协议与切分。档案柜里那份草稿值得一看。',
      en: 'Good. Do not invent a new model yet — inspect the evaluation protocol and split. That archive draft matters.',
    },
  };
}

export function runSimulatedExperiment(): ExperimentLink {
  const link: ExperimentLink = {
    id: `sim_${Date.now()}`,
    label: '清洁切分对照（模拟）',
    summary: '严格隔离测试集后，甲乙主指标差距落入噪声；支持协议有偏。',
    linkedAt: new Date().toISOString(),
    simulated: true,
  };
  mutate((c) => {
    c.experimentRuns.push(link);
    if (!c.evidence.some((e) => e.id === 'ev_exp_01')) {
      const exp = DEMO_EVIDENCE_CATALOG.ev_exp_01;
      c.evidence.push({ ...exp, discoveredAt: new Date().toISOString() });
    }
    // auto-support protocol bias claim
    if (!c.edges.some((e) => e.from === 'ev_exp_01' && e.to === 'cl_protocol_bias')) {
      c.edges.push({
        id: `edge_${Date.now()}`,
        from: 'ev_exp_01',
        to: 'cl_protocol_bias',
        type: 'supports',
      });
    }
    if (!c.edges.some((e) => e.from === 'ev_exp_01' && e.to === 'cl_method_a_magic' && e.type === 'contradicts')) {
      c.edges.push({
        id: `edge_${Date.now()}_c`,
        from: 'ev_exp_01',
        to: 'cl_method_a_magic',
        type: 'contradicts',
      });
    }
    return c;
  });
  return link;
}

export function setResearchQuestion(text: string) {
  mutate((c) => {
    c.researchQuestion = text;
    return c;
  });
}

export function setHypothesis(text: string, maturity: HypothesisMaturity) {
  mutate((c) => {
    c.hypothesis = { text, maturity };
    return c;
  });
}

export function setResultsSummary(text: string) {
  mutate((c) => {
    c.resultsSummary = text;
    return c;
  });
}

export function setLimitations(lines: string[]) {
  mutate((c) => {
    c.limitations = lines;
    return c;
  });
}

export function setClaimState(claimId: string, claimState: ResearchCase['claims'][0]['state']) {
  mutate((c) => {
    const cl = c.claims.find((x) => x.id === claimId);
    if (cl) cl.state = claimState;
    return c;
  });
}

export function addEdge(from: string, to: string, type: EdgeType) {
  mutate((c) => {
    const exists = c.edges.some((e) => e.from === from && e.to === to && e.type === type);
    if (exists) return c;
    const edge: CaseEdge = { id: `edge_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`, from, to, type };
    c.edges.push(edge);
    return c;
  });
}

export function removeEdge(edgeId: string) {
  mutate((c) => {
    c.edges = c.edges.filter((e) => e.id !== edgeId);
    return c;
  });
}

export function submitReview(): { passed: boolean; findingsCount: number } {
  const c = requireCase();
  const findings = evaluateReview(c);
  const passed = findings.length === 0;
  mutate((cur) => {
    const round = {
      id: `rev_${Date.now()}`,
      index: cur.reviewRounds.length + 1,
      status: (passed ? 'passed' : 'failed') as 'passed' | 'failed',
      findings,
      submittedAt: new Date().toISOString(),
      resolvedAt: new Date().toISOString(),
    };
    cur.reviewRounds.push(round);
    cur.status = passed ? 'ready_for_review' : 'revision';
    return cur;
  });
  return { passed, findingsCount: findings.length };
}

export function markClosed() {
  mutate((c) => {
    c.status = 'closed';
    return c;
  });
}

export function useCase(): ResearchCase | null {
  return useSyncExternalStore(
    (onStoreChange) => {
      listeners.add(onStoreChange);
      return () => listeners.delete(onStoreChange);
    },
    () => state,
    () => null,
  );
}

export function fillDemoScaffoldTexts() {
  mutate((c) => {
    if (!c.researchQuestion.trim()) {
      c.researchQuestion =
        '在公开海报宣称方法甲显著优于方法乙、而独立复现失败的前提下，该差距更可能来自模型本身，还是来自有偏的评估协议？';
    }
    if (!c.hypothesis.text.trim()) {
      c.hypothesis = {
        text: '若严格隔离测试数据并固定对照，方法甲相对乙的优势将显著缩小或消失——说明海报数字受到评估协议偏差影响。',
        maturity: 'operationalized',
      };
    }
    if (!c.resultsSummary.trim()) {
      c.resultsSummary =
        '清洁切分对照下甲乙主指标接近；海报中的大幅领先无法在可复现设置中站稳，更支持协议有偏。';
    }
    if (!c.limitations.some((x) => x.trim())) {
      c.limitations = [
        '仅使用白名单固定模板与有限 seed，不能外推到全部任务与模型族。',
        '未获取海报作者原始训练代码，协议还原可能不完整。',
      ];
    }
    // helpful default edges if evidence present
    if (c.evidence.some((e) => e.id === 'ev_poster_a')) {
      if (!c.edges.some((e) => e.from === 'ev_poster_a' && e.to === 'cl_method_a_magic')) {
        c.edges.push({ id: `edge_auto_pa`, from: 'ev_poster_a', to: 'cl_method_a_magic', type: 'supports' });
      }
    }
    if (c.evidence.some((e) => e.id === 'ev_repro_fail')) {
      if (!c.edges.some((e) => e.from === 'ev_repro_fail' && e.to === 'cl_method_a_magic')) {
        c.edges.push({ id: `edge_auto_rf`, from: 'ev_repro_fail', to: 'cl_method_a_magic', type: 'contradicts' });
      }
    }
    if (c.evidence.some((e) => e.id === 'ev_split_leak')) {
      if (!c.edges.some((e) => e.from === 'ev_split_leak' && e.to === 'cl_protocol_bias')) {
        c.edges.push({ id: `edge_auto_sl`, from: 'ev_split_leak', to: 'cl_protocol_bias', type: 'supports' });
      }
    }
    return c;
  });
}

export { DEMO_CASE_ID };
