import { useEffect, useState } from 'react';

export type ZoneId = 'observatory' | 'lab' | 'compute' | 'paper' | 'plaza';
export type TutorId = 'daily' | 'navigator' | 'poster';
export type ZoneStatus = 'locked' | 'ready' | 'active' | 'complete';
export type ExperimentChoice = 'baseline' | 'ablation' | 'stress';

export type Zone = {
  id: ZoneId;
  name: string;
  shortName: string;
  subtitle: string;
  asset: string;
  hotspot: {
    left: string;
    top: string;
    width: string;
    height: string;
    radius: string;
  };
  accent: string;
  task: string;
  output: string;
  startAction: string;
  completeAction: string;
  lockedHint: string;
  tutor: TutorId;
  tutorLine: string;
  noteLabel: string;
  notePlaceholder: string;
  stats: Array<{ label: string; value: string }>;
};

export type TutorOutfit = {
  id: TutorId;
  name: string;
  asset: string;
  role: string;
};

export type Artifact = {
  id: ZoneId;
  zoneId: ZoneId;
  name: string;
  kind: string;
  detail: string;
};

export type GameState = {
  started: ZoneId[];
  completed: ZoneId[];
  activeZoneId: ZoneId;
  tutorId: TutorId;
  experimentChoice: ExperimentChoice;
  experimentRan: boolean;
  zoneNotes: Partial<Record<ZoneId, string>>;
};

export const ASSET_ROOT = `${import.meta.env.BASE_URL}game/research-archipelago`;
export const STORAGE_KEY = 'research-game:p1-state';
export const zoneOrder: ZoneId[] = ['observatory', 'lab', 'compute', 'paper', 'plaza'];

export const zones: Zone[] = [
  {
    id: 'observatory',
    name: '模型与系统实验室 · 项目与数据终端',
    shortName: '项目终端',
    subtitle: '审计 CodeSearchNet 真实代码、人工相关性标注与 BM25 失败案例',
    asset: `${ASSET_ROOT}/landmarks/model-system-observatory.webp`,
    hotspot: { left: '7%', top: '2%', width: '36%', height: '39%', radius: '44%' },
    accent: 'border-ink-blue bg-pale-blue text-ink-blue',
    task: '审计冻结的 CodeSearchNet Python 人工评测切片，定位 body-only BM25 的检索盲区。',
    output: '基准审计报告',
    startAction: '开始基准审计',
    completeAction: '封存基准审计',
    lockedHint: '从这里开始。',
    tutor: 'navigator',
    tutorLine: '先检查真实源码、人工相关性标注和 BM25 失败查询，再判断 AST 字段、混合检索和重排序是否值得加入。',
    noteLabel: '基准审计记录',
    notePlaceholder: '记录数据来源、仓库隔离、人工相关性标注、源码排除项和典型失败案例。',
    stats: [
      { label: '真实代码', value: '938' },
      { label: '测试查询', value: '85' },
      { label: '仓库/提交', value: '738' },
    ],
  },
  {
    id: 'lab',
    name: '模型与系统实验室 · 研究假设白板',
    shortName: '假设白板',
    subtitle: '冻结检索假设、变量、指标与实验比较关系',
    asset: `${ASSET_ROOT}/interiors/personal-lab.webp`,
    hotspot: { left: '3%', top: '36%', width: '37%', height: '44%', radius: '38%' },
    accent: 'border-[#2f7f80] bg-[#e4f3f1] text-[#226364]',
    task: '提出完整 Code RAG 检索流水线相对 body-only BM25 提升 nDCG@10 的可检验假设。',
    output: '检索实验契约',
    startAction: '起草检索假设',
    completeAction: '冻结实验契约',
    lockedHint: '先在同一实验室的项目与数据终端完成基准审计。',
    tutor: 'daily',
    tutorLine: '每次只解释一个增量：真实 AST 字段、local LSA、RRF 和 reranker 都必须留下独立证据。',
    noteLabel: '检索假设',
    notePlaceholder: '相较 body-only BM25，完整检索流水线将提高 nDCG@10，因为……',
    stats: [
      { label: '自变量', value: '流水线' },
      { label: '主指标', value: 'nDCG@10' },
      { label: '方向', value: '提高' },
    ],
  },
  {
    id: 'compute',
    name: '算力工坊',
    shortName: '算力工坊',
    subtitle: '构建索引、运行四种检索配置并比较消融结果',
    asset: `${ASSET_ROOT}/landmarks/compute-workshop.webp`,
    hotspot: { left: '49%', top: '3%', width: '45%', height: '40%', radius: '34%' },
    accent: 'border-ink-green bg-pale-green text-ink-green',
    task: '依次运行 BM25 Body、BM25 AST、Hybrid 与 Hybrid + Reranker 固定基准。',
    output: '代码检索基准结果',
    startAction: '配置检索基准',
    completeAction: '封存检索结果',
    lockedHint: '先在模型与系统实验室完成实验设计。',
    tutor: 'navigator',
    tutorLine: '四个配置使用同一批 85 条测试查询和人工等级标注。不要把数据变化误当成检索算法收益。',
    noteLabel: '基准运行备注',
    notePlaceholder: '记录索引配置、查询失败、延迟和各阶段 nDCG@10 变化。',
    stats: [
      { label: '配置', value: '4' },
      { label: 'Bootstrap', value: '1,000' },
      { label: '网络', value: '禁用' },
    ],
  },
  {
    id: 'paper',
    name: '论文与评审工坊',
    shortName: '评审工坊',
    subtitle: '分析 nDCG@10 与置信区间、解释负向消融并完成复现验证',
    asset: `${ASSET_ROOT}/landmarks/paper-review-workshop.webp`,
    hotspot: { left: '30%', top: '48%', width: '39%', height: '50%', radius: '38%' },
    accent: 'border-ink-yellow bg-pale-yellow text-ink-yellow',
    task: '将四组检索结果转化为可审稿的 claim，明确最佳配置、延迟代价和基准限制。',
    output: 'Code RAG 论文草稿',
    startAction: '搭建论文叙事',
    completeAction: '生成论文草稿',
    lockedHint: '先完成算力工坊的实验结果。',
    tutor: 'poster',
    tutorLine: '如果 reranker 让结果变差，就把它写成负向消融。论文的任务是解释证据，而不是粉饰组件。',
    noteLabel: '核心 Claim',
    notePlaceholder: '写下主张、对应证据、适用范围，以及目前仍无法支持的部分。',
    stats: [
      { label: '段落', value: '8' },
      { label: '证据', value: '5' },
      { label: '缺口', value: '2' },
    ],
  },
  {
    id: 'plaza',
    name: '学术交流广场',
    shortName: '交流广场',
    subtitle: '发布验证后的检索报告、复现包与代码搜索展示',
    asset: `${ASSET_ROOT}/landmarks/academic-exchange-plaza.webp`,
    hotspot: { left: '66%', top: '45%', width: '30%', height: '46%', radius: '50%' },
    accent: 'border-ink-red bg-pale-red text-ink-red',
    task: '公开经过复现验证的 Code RAG 研究卡，同时保持原始运行日志私有。',
    output: 'Code RAG 研究卡',
    startAction: '准备展示',
    completeAction: '生成展示卡片',
    lockedHint: '先完成论文与评审工坊。',
    tutor: 'poster',
    tutorLine: '公开结果必须同时说明数据来源、人工评测范围、本地 LSA 的能力边界和 reranker 的负向结果。',
    noteLabel: '90 秒展示摘要',
    notePlaceholder: '问题是什么、你做了什么、发现了什么、最需要哪一种反馈？',
    stats: [
      { label: '反馈', value: '4' },
      { label: '匹配', value: '2' },
      { label: '热度', value: '新' },
    ],
  },
];

export const tutorOutfits: TutorOutfit[] = [
  {
    id: 'daily',
    name: '研究方法导师',
    asset: `${ASSET_ROOT}/characters/tutor-daily-research.webp`,
    role: '梳理问题、校准假设与研究路线',
  },
  {
    id: 'navigator',
    name: '科研岛向导',
    asset: `${ASSET_ROOT}/characters/tutor-campus-navigator.webp`,
    role: '定位任务、解释区域与实验路径',
  },
  {
    id: 'poster',
    name: '学术交流导师',
    asset: `${ASSET_ROOT}/characters/tutor-poster-session.webp`,
    role: '组织证据、评审反馈与成果展示',
  },
];

export const experimentOptions: Array<{
  id: ExperimentChoice;
  name: string;
  detail: string;
  result: string;
  score: string;
}> = [
  {
    id: 'baseline',
    name: 'BM25 Body',
    detail: '静态演示：只索引代码正文，建立不使用 AST 字段的稀疏检索基线。',
    result: '真实 runner 的冻结结果中，body-only BM25 是最低的 nDCG@10 配置。',
    score: 'nDCG .7940',
  },
  {
    id: 'ablation',
    name: 'Hybrid',
    detail: '静态演示：融合真实 AST 字段 BM25 与本地字符 n-gram LSA 排名。',
    result: '真实 runner 的冻结结果中，Hybrid 获得最高 nDCG@10。',
    score: 'nDCG .8353',
  },
  {
    id: 'stress',
    name: '+ Reranker',
    detail: '静态演示：在 Hybrid 候选集上加入确定性的代码特征重排序。',
    result: '真实 runner 的冻结结果中，轻量 reranker 略有退化，应作为负向消融报告。',
    score: 'nDCG .8302',
  },
];

const artifactByZone: Record<ZoneId, Artifact> = {
  observatory: {
    id: 'observatory',
    zoneId: 'observatory',
    name: '观测报告',
    kind: 'Observation',
    detail: '记录语料、查询类别和 BM25 排名失败案例。',
  },
  lab: {
    id: 'lab',
    zoneId: 'lab',
    name: '研究假设',
    kind: 'Hypothesis',
    detail: '冻结检索流水线、nDCG@10 与预期方向。',
  },
  compute: {
    id: 'compute',
    zoneId: 'compute',
    name: '实验结果',
    kind: 'Result',
    detail: '包含四个固定检索配置的真实基准指标。',
  },
  paper: {
    id: 'paper',
    zoneId: 'paper',
    name: '论文草稿',
    kind: 'Draft',
    detail: '把 nDCG@10、置信区间、延迟和负向消融组织成可审稿结论。',
  },
  plaza: {
    id: 'plaza',
    zoneId: 'plaza',
    name: '展示卡片',
    kind: 'Poster',
    detail: '面向同伴反馈的 Code RAG 检索研究摘要。',
  },
};

const defaultState: GameState = {
  started: [],
  completed: [],
  activeZoneId: 'observatory',
  tutorId: 'navigator',
  experimentChoice: 'baseline',
  experimentRan: false,
  zoneNotes: {},
};

export const researcherAsset = `${ASSET_ROOT}/characters/researcher-base-avatar.webp`;
export const worldAsset = `${ASSET_ROOT}/world/integrated-five-zone-campus.webp`;

export function getZone(zoneId: string | undefined): Zone | undefined {
  return zones.find((zone) => zone.id === zoneId);
}

export function isZoneId(value: unknown): value is ZoneId {
  return typeof value === 'string' && zoneOrder.includes(value as ZoneId);
}

export function getZoneStatus(zoneId: ZoneId, game: GameState): ZoneStatus {
  if (game.completed.includes(zoneId)) return 'complete';
  const index = zoneOrder.indexOf(zoneId);
  const previous = zoneOrder[index - 1];
  const unlocked = index === 0 || game.completed.includes(previous);
  if (!unlocked) return 'locked';
  if (game.started.includes(zoneId)) return 'active';
  return 'ready';
}

export function statusLabel(status: ZoneStatus): string {
  if (status === 'complete') return '已完成';
  if (status === 'active') return '进行中';
  if (status === 'ready') return '可进入';
  return '未解锁';
}

export function nextZone(zoneId: ZoneId): Zone | undefined {
  const nextId = zoneOrder[zoneOrder.indexOf(zoneId) + 1];
  return zones.find((zone) => zone.id === nextId);
}

export function useResearchGame() {
  const [game, setGame] = useState<GameState>(loadGameState);

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(game));
  }, [game]);

  const startZone = (zoneId: ZoneId) => {
    const zone = getZone(zoneId);
    setGame((current) => {
      if (getZoneStatus(zoneId, current) !== 'ready') return current;
      return {
        ...current,
        activeZoneId: zoneId,
        tutorId: zone?.tutor ?? current.tutorId,
        started: addUnique(current.started, zoneId),
      };
    });
  };

  const completeZone = (zoneId: ZoneId) => {
    setGame((current) => {
      if (getZoneStatus(zoneId, current) !== 'active') return current;
      if (zoneId === 'compute' && !current.experimentRan) return current;
      const completed = addUnique(current.completed, zoneId);
      const nextZoneId = nextAvailableZone(zoneId, completed);
      const next = nextZoneId ? getZone(nextZoneId) : undefined;
      return {
        ...current,
        completed,
        started: addUnique(current.started, zoneId),
        activeZoneId: nextZoneId ?? zoneId,
        tutorId: next?.tutor ?? current.tutorId,
      };
    });
  };

  const setTutor = (tutorId: TutorId) => {
    setGame((current) => ({ ...current, tutorId }));
  };

  const setZoneNote = (zoneId: ZoneId, note: string) => {
    setGame((current) => ({
      ...current,
      activeZoneId: zoneId,
      zoneNotes: { ...current.zoneNotes, [zoneId]: note },
    }));
  };

  const selectExperiment = (experimentChoice: ExperimentChoice) => {
    setGame((current) => ({ ...current, experimentChoice, activeZoneId: 'compute' }));
  };

  const runExperiment = () => {
    setGame((current) => ({
      ...current,
      activeZoneId: 'compute',
      tutorId: 'navigator',
      started: addUnique(current.started, 'compute'),
      experimentRan: true,
    }));
  };

  const resetProgress = () => {
    setGame({ ...defaultState, started: [], completed: [], zoneNotes: {} });
  };

  const artifacts = zones
    .filter((zone) => game.completed.includes(zone.id))
    .map((zone) => artifactByZone[zone.id]);

  return {
    game,
    artifacts,
    startZone,
    completeZone,
    setTutor,
    setZoneNote,
    selectExperiment,
    runExperiment,
    resetProgress,
  };
}

function loadGameState(): GameState {
  if (typeof window === 'undefined') return defaultState;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaultState;
    const parsed = JSON.parse(raw) as Partial<GameState>;
    return {
      started: normalizeZones(parsed.started),
      completed: normalizeZones(parsed.completed),
      activeZoneId: isZoneId(parsed.activeZoneId) ? parsed.activeZoneId : defaultState.activeZoneId,
      tutorId: isTutorId(parsed.tutorId) ? parsed.tutorId : defaultState.tutorId,
      experimentChoice: isExperimentChoice(parsed.experimentChoice)
        ? parsed.experimentChoice
        : defaultState.experimentChoice,
      experimentRan: !!parsed.experimentRan,
      zoneNotes: normalizeZoneNotes(parsed.zoneNotes),
    };
  } catch {
    return defaultState;
  }
}

function normalizeZones(value: unknown): ZoneId[] {
  if (!Array.isArray(value)) return [];
  return zoneOrder.filter((zoneId) => value.includes(zoneId));
}

function normalizeZoneNotes(value: unknown): Partial<Record<ZoneId, string>> {
  if (!value || typeof value !== 'object') return {};
  const source = value as Record<string, unknown>;
  return Object.fromEntries(
    zoneOrder
      .filter((zoneId) => typeof source[zoneId] === 'string')
      .map((zoneId) => [zoneId, source[zoneId] as string]),
  );
}

function isTutorId(value: unknown): value is TutorId {
  return value === 'daily' || value === 'navigator' || value === 'poster';
}

function isExperimentChoice(value: unknown): value is ExperimentChoice {
  return value === 'baseline' || value === 'ablation' || value === 'stress';
}

function addUnique<T>(items: T[], item: T): T[] {
  return items.includes(item) ? items : [...items, item];
}

function nextAvailableZone(current: ZoneId, completed: ZoneId[]): ZoneId | null {
  const currentIndex = zoneOrder.indexOf(current);
  for (let index = currentIndex + 1; index < zoneOrder.length; index += 1) {
    const zoneId = zoneOrder[index];
    if (!completed.includes(zoneId)) return zoneId;
  }
  return null;
}
