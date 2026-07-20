import type { CasePoi, EvidenceNode, Localized, ResearchCase } from './types';

const L = (zh: string, en: string): Localized => ({ zh, en });

/** Demo case content: exaggerated accuracy / protocol bias. */
export const DEMO_CASE_ID = 'case_demo_01';

export const DEMO_EVIDENCE_CATALOG: Record<string, Omit<EvidenceNode, 'discoveredAt'>> = {
  ev_poster_a: {
    id: 'ev_poster_a',
    kind: 'citation',
    title: L('海报：方法甲 92%', 'Poster: Method A at 92%'),
    body: L(
      '学术广场海报宣称方法甲在公开基准上准确率 92%，显著高于方法乙的 78%。海报未写清训练/测试切分与对照设置。',
      'A plaza poster claims Method A hits 92% on a public benchmark, far above Method B at 78%. Split and controls are underspecified.',
    ),
    sourcePoiId: 'poi_poster_a',
    sourceNpcId: null,
    tags: ['poster', 'method_a', 'metric'],
  },
  ev_repro_fail: {
    id: 'ev_repro_fail',
    kind: 'log',
    title: L('复现日志：持平甚至更差', 'Repro log: flat or worse'),
    body: L(
      '机房残骸上的复现日志显示：按海报描述实现后，方法甲与乙主指标几乎持平，个别 seed 更差。作者“显著领先”无法复现。',
      'A repro log on the compute wreck shows Method A roughly ties B; some seeds are worse. The claimed lead does not reproduce.',
    ),
    sourcePoiId: 'poi_repro_log',
    sourceNpcId: null,
    tags: ['repro', 'conflict'],
  },
  ev_split_leak: {
    id: 'ev_split_leak',
    kind: 'observation',
    title: L('数据切分说明可疑', 'Suspicious data split notes'),
    body: L(
      '档案柜中的协议草稿写着：测试子集曾用于早停与调参。若属实，海报数字可能含泄漏，不能当作泛化成绩。',
      'A protocol draft says the test subset was used for early stopping and tuning. If true, the poster number may leak and is not a clean generalization score.',
    ),
    sourcePoiId: 'poi_split_doc',
    sourceNpcId: null,
    tags: ['leak', 'protocol', 'baseline'],
  },
  ev_exp_01: {
    id: 'ev_exp_01',
    kind: 'experiment_result',
    title: L('对照实验：清洁切分后差距消失', 'Experiment: gap vanishes under clean split'),
    body: L(
      '在白名单模板上按「训练/验证/测试严格隔离」重跑后，方法甲与乙主指标差距落在噪声范围内。支持“评估协议有偏”而非“甲有魔法”。',
      'Rerunning the whitelist template with a strict train/val/test isolation, A vs B falls within noise. Supports biased evaluation protocol over magical Method A.',
    ),
    sourcePoiId: 'poi_bench',
    sourceNpcId: null,
    tags: ['experiment', 'baseline', 'leak'],
  },
  ev_mentor_note: {
    id: 'ev_mentor_note',
    kind: 'note',
    title: L('导师笔记：先找矛盾再下结论', 'Mentor note: name the contradiction first'),
    body: L(
      '导师要求你先口头说清「海报 vs 复现」的矛盾，然后才提示：优先检查评估协议与数据泄漏，而不是急着发明新模型。',
      'The mentor insists you state the poster-vs-repro contradiction before any tip: check evaluation protocol and leakage before inventing a new model.',
    ),
    sourceNpcId: 'mentor',
    tags: ['method', 'mentor'],
  },
};

export const DEMO_POIS: CasePoi[] = [
  {
    id: 'poi_poster_a',
    caseId: DEMO_CASE_ID,
    sceneId: 'plaza',
    label: L('广场海报墙', 'Plaza poster wall'),
    attractor: 'poster',
    grantsEvidenceIds: ['ev_poster_a'],
    inspectOnce: true,
    visitHint: L('一张被翻得起毛的海报还贴着 92% 的箭头。', 'A worn poster still points proudly at 92%.'),
  },
  {
    id: 'poi_repro_log',
    caseId: DEMO_CASE_ID,
    sceneId: 'compute',
    label: L('复现失败残骸', 'Failed repro wreck'),
    attractor: 'wreck',
    grantsEvidenceIds: ['ev_repro_fail'],
    inspectOnce: true,
    visitHint: L('终端还亮着红字 seed 表。', 'A terminal still glows with red seed tables.'),
  },
  {
    id: 'poi_split_doc',
    caseId: DEMO_CASE_ID,
    sceneId: 'research',
    label: L('档案柜·协议草稿', 'Archive · protocol draft'),
    attractor: 'glow',
    grantsEvidenceIds: ['ev_split_leak'],
    requires: { flags: ['knows_leak'] },
    inspectOnce: true,
    visitHint: L('抽屉里夹着一页被划掉又改写的切分说明。', 'A reworked split note is stuck in the drawer.'),
  },
  {
    id: 'poi_mentor',
    caseId: DEMO_CASE_ID,
    sceneId: 'academy',
    label: L('学院·方法导师', 'Academy · methods mentor'),
    attractor: 'npc',
    grantsEvidenceIds: [],
    inspectOnce: false,
    visitHint: L('导师只在你说清矛盾之后才肯往下讲。', 'The mentor only continues after you name the contradiction.'),
  },
  {
    id: 'poi_bench',
    caseId: DEMO_CASE_ID,
    sceneId: 'compute',
    label: L('机房对照工位', 'Compute control bench'),
    attractor: 'terminal',
    grantsEvidenceIds: ['ev_exp_01'],
    requires: { evidenceIds: ['ev_split_leak'] },
    inspectOnce: true,
    visitHint: L('白名单实验模板已加载，可跑一次清洁切分对照。', 'A whitelist template is loaded for one clean-split control run.'),
  },
];

export function createDemoCaseShell(now = new Date().toISOString()): ResearchCase {
  return {
    id: DEMO_CASE_ID,
    title: L('被夸大的准确率', 'The Inflated Accuracy'),
    bulletin: L(
      '公开海报称方法甲在基准上远超方法乙；独立复现却得到持平甚至更差。谁在说谎——还是协议在说谎？',
      'A public poster says Method A crushes Method B; independent reproduction is flat or worse. Who is lying — or is the protocol?',
    ),
    status: 'briefed',
    researchQuestion: '',
    hypothesis: { text: '', maturity: 'wild_guess' },
    resultsSummary: '',
    limitations: [],
    claims: [
      {
        id: 'cl_method_a_magic',
        text: '方法甲本身显著优于方法乙（海报叙事）',
        state: 'active',
      },
      {
        id: 'cl_protocol_bias',
        text: '评估协议/数据切分有偏，导致海报数字不可比',
        state: 'active',
      },
    ],
    evidence: [],
    edges: [],
    reviewRounds: [],
    experimentRuns: [],
    flags: [],
    completeness: 0,
    createdAt: now,
    updatedAt: now,
  };
}
