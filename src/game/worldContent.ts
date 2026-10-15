import type {
  AgentDefinition,
  AgentId,
  AvatarPreset,
  AvatarPresetId,
  GameSceneDefinition,
  GameSceneId,
} from './types';

const root = `${import.meta.env.BASE_URL}game/research-archipelago`;
const roleRoot = `${root}/characters/roles`;

export const avatarPresets: AvatarPreset[] = [
  {
    id: 'theory-blue',
    name: '理论探索者',
    field: '理论与算法',
    asset: `${roleRoot}/theory-blue.webp`,
    color: 0x1976d2,
    cssColor: '#1976d2',
  },
  {
    id: 'systems-teal',
    name: '系统建造者',
    field: '系统与基础设施',
    asset: `${roleRoot}/systems-teal.webp`,
    color: 0x00897b,
    cssColor: '#00897b',
  },
  {
    id: 'vision-green',
    name: '感知研究员',
    field: '视觉与多模态',
    asset: `${roleRoot}/vision-green.webp`,
    color: 0x43a047,
    cssColor: '#43a047',
  },
  {
    id: 'robotics-gold',
    name: '具身实践者',
    field: '机器人与强化学习',
    asset: `${roleRoot}/robotics-gold.webp`,
    color: 0xf9a825,
    cssColor: '#b97700',
  },
  {
    id: 'agents-red',
    name: '智能体设计师',
    field: 'Agent 与自动化科研',
    asset: `${roleRoot}/agents-red.webp`,
    color: 0xe53935,
    cssColor: '#d32f2f',
  },
  {
    id: 'safety-violet',
    name: '安全审稿人',
    field: '安全、伦理与对齐',
    asset: `${roleRoot}/safety-violet.webp`,
    color: 0x7e57c2,
    cssColor: '#6d4c9c',
  },
];

export const avatarPresetById = Object.fromEntries(
  avatarPresets.map((preset) => [preset.id, preset]),
) as Record<AvatarPresetId, AvatarPreset>;

export const agents: Record<AgentId, AgentDefinition> = {
  'campus-guide': {
    id: 'campus-guide',
    name: '林澈',
    role: '科研岛向导 Agent',
    greeting: '先不用急着做大课题。我们从你现在最不确定的一个问题开始。',
    focus: '识别当前研究阶段，安排场景路线，并解释每个区域能解决什么问题。',
    asset: `${root}/characters/tutor-campus-navigator.webp`,
  },
  'research-mentor': {
    id: 'research-mentor',
    name: '苏老师',
    role: '科研训练导师 Agent',
    greeting: '科研不是灵感考试，而是一套可以练习的动作。告诉我你卡在哪一步。',
    focus: '基于 503 节课程诊断基础，教授选题、文献、实验、分析、复现和写作。',
    asset: `${root}/characters/tutor-daily-research.webp`,
  },
  'methods-mentor': {
    id: 'methods-mentor',
    name: '周岚',
    role: '研究方法 Agent',
    greeting: '把课题、证据和可反驳条件放在同一张白板上，我们再决定实验。',
    focus: '合并数据审计、研究问题、假设、变量和主指标，并保持人工确认。',
    asset: `${root}/characters/tutor-daily-research.webp`,
  },
  'experiment-engineer': {
    id: 'experiment-engineer',
    name: '程野',
    role: '实验工程 Agent',
    greeting: '配置先冻结，结果再说话。失败日志也属于实验产物。',
    focus: '解释固定实验配置、队列、资源限制、失败恢复和零 API 成本边界。',
    asset: `${root}/characters/tutor-campus-navigator.webp`,
    workspaceZone: 'compute',
  },
  reviewer: {
    id: 'reviewer',
    name: '顾言',
    role: '审稿人 Agent',
    greeting: '我会先找最可能推翻结论的地方，再帮你组织论文。',
    focus: '检查 claim、置信区间、负向消融、复现结果和仍需外部验证的边界。',
    asset: `${root}/characters/tutor-poster-session.webp`,
    workspaceZone: 'paper',
  },
  'community-host': {
    id: 'community-host',
    name: '许禾',
    role: '学术社区 Agent',
    greeting: '先说你最希望别人挑战哪条结论，这比只展示最好结果更有用。',
    focus: '帮助发布经过验证的研究卡、组织反馈和保持原始日志私有。',
    asset: `${root}/characters/tutor-poster-session.webp`,
    workspaceZone: 'plaza',
  },
};

export const gameScenes: Record<GameSceneId, GameSceneDefinition> = {
  campus: {
    id: 'campus',
    name: '科研公社大厅',
    subtitle: '自由行走、遇见研究伙伴，并从五个入口开始科研',
    background: `${root}/world/integrated-five-zone-campus.webp`,
    spawn: { x: 800, y: 570 },
    tint: 0xe9f3f1,
    interactions: [
      { id: 'guide', kind: 'agent', label: '科研岛向导', position: { x: 790, y: 430 }, agentId: 'campus-guide' },
      { id: 'academy-door', kind: 'portal', label: '进入科研训练馆', position: { x: 390, y: 250 }, targetScene: 'academy' },
      { id: 'research-door', kind: 'portal', label: '进入模型与系统实验室', position: { x: 310, y: 650 }, targetScene: 'research' },
      { id: 'compute-door', kind: 'portal', label: '进入算力工坊', position: { x: 1180, y: 270 }, targetScene: 'compute' },
      { id: 'paper-door', kind: 'portal', label: '进入论文与评审工坊', position: { x: 790, y: 720 }, targetScene: 'paper' },
      { id: 'plaza-door', kind: 'portal', label: '进入学术交流广场', position: { x: 1320, y: 650 }, targetScene: 'plaza' },
    ],
  },
  academy: {
    id: 'academy',
    name: '科研训练馆',
    subtitle: '面向没有系统科研训练的学习者，连接 503 节课程与研究方法',
    background: `${root}/landmarks/model-system-observatory.webp`,
    spawn: { x: 800, y: 720 },
    tint: 0xe8f2fa,
    interactions: [
      { id: 'academy-mentor', kind: 'agent', label: '和苏老师讨论科研方法', position: { x: 800, y: 450 }, agentId: 'research-mentor' },
      { id: 'academy-exit', kind: 'portal', label: '返回科研公社大厅', position: { x: 170, y: 760 }, targetScene: 'campus' },
    ],
  },
  research: {
    id: 'research',
    name: '模型与系统实验室',
    subtitle: '在同一场景完成项目选择、数据审计、研究问题和假设冻结',
    background: `${root}/interiors/personal-lab.webp`,
    spawn: { x: 800, y: 720 },
    tint: 0xe8f3ef,
    interactions: [
      { id: 'methods-mentor', kind: 'agent', label: '和研究方法导师讨论', position: { x: 790, y: 430 }, agentId: 'methods-mentor' },
      { id: 'project-terminal', kind: 'workspace', label: '使用项目与数据终端', position: { x: 1080, y: 520 }, workspaceZone: 'observatory' },
      { id: 'hypothesis-board', kind: 'workspace', label: '使用假设白板', position: { x: 500, y: 520 }, workspaceZone: 'lab' },
      { id: 'research-exit', kind: 'portal', label: '返回科研公社大厅', position: { x: 170, y: 760 }, targetScene: 'campus' },
    ],
  },
  compute: {
    id: 'compute',
    name: '算力工坊',
    subtitle: '启动固定实验、查看逐 case 指标并处理失败恢复',
    background: `${root}/landmarks/compute-workshop.webp`,
    spawn: { x: 800, y: 720 },
    tint: 0xeaf4ed,
    interactions: [
      { id: 'compute-agent', kind: 'agent', label: '和实验工程师讨论', position: { x: 650, y: 460 }, agentId: 'experiment-engineer' },
      { id: 'cluster-console', kind: 'workspace', label: '操作离线实验集群', position: { x: 1020, y: 500 }, workspaceZone: 'compute' },
      { id: 'compute-exit', kind: 'portal', label: '返回科研公社大厅', position: { x: 170, y: 760 }, targetScene: 'campus' },
    ],
  },
  paper: {
    id: 'paper',
    name: '论文与评审工坊',
    subtitle: '让审稿人先挑战结论，再组织分析、复现与论文证据',
    background: `${root}/landmarks/paper-review-workshop.webp`,
    spawn: { x: 800, y: 720 },
    tint: 0xf8f2e6,
    interactions: [
      { id: 'reviewer', kind: 'agent', label: '接受审稿人质询', position: { x: 660, y: 450 }, agentId: 'reviewer' },
      { id: 'manuscript-desk', kind: 'workspace', label: '打开分析与复现稿件', position: { x: 1020, y: 520 }, workspaceZone: 'paper' },
      { id: 'paper-exit', kind: 'portal', label: '返回科研公社大厅', position: { x: 170, y: 760 }, targetScene: 'campus' },
    ],
  },
  plaza: {
    id: 'plaza',
    name: '学术交流广场',
    subtitle: '遇见其他研究者，展示经过验证且可以撤回的研究成果',
    background: `${root}/landmarks/academic-exchange-plaza.webp`,
    spawn: { x: 800, y: 720 },
    tint: 0xf4eceb,
    interactions: [
      { id: 'community-host', kind: 'agent', label: '和学术社区导师讨论', position: { x: 650, y: 450 }, agentId: 'community-host' },
      { id: 'poster-board', kind: 'workspace', label: '查看研究海报板', position: { x: 1020, y: 520 }, workspaceZone: 'plaza' },
      { id: 'plaza-exit', kind: 'portal', label: '返回科研公社大厅', position: { x: 170, y: 760 }, targetScene: 'campus' },
    ],
  },
};

export const gameSceneIds = Object.keys(gameScenes) as GameSceneId[];

export function isGameSceneId(value: unknown): value is GameSceneId {
  return typeof value === 'string' && gameSceneIds.includes(value as GameSceneId);
}
