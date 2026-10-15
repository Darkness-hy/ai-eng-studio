import type { RunSnapshot, RunStatus } from './researchApi';
import type { ZoneId, ZoneStatus } from './researchGame';

const postExperimentStatuses: RunStatus[] = [
  'analysis_ready',
  'analyzing',
  'verification_required',
  'verifying',
  'completed',
];

export function getResearchRunZoneStatus(zoneId: ZoneId, snapshot: RunSnapshot | null): ZoneStatus {
  if (!snapshot) return zoneId === 'observatory' ? 'ready' : 'locked';

  const { run, hypothesis, artifacts } = snapshot;
  const hasResearchCard = artifacts.some((artifact) => artifact.kind === 'research_card');
  const cardPublished = artifacts.some(
    (artifact) => artifact.kind === 'research_card' && artifact.visibility === 'public',
  );

  if (zoneId === 'observatory') return 'complete';
  if (zoneId === 'lab') return hypothesis?.status === 'accepted' ? 'complete' : 'active';

  if (zoneId === 'compute') {
    if (!hypothesis) return 'locked';
    if (postExperimentStatuses.includes(run.status)) return 'complete';
    if (run.status === 'hypothesis_ready') return 'ready';
    return 'active';
  }

  if (zoneId === 'paper') {
    if (run.status === 'completed') return 'complete';
    if (postExperimentStatuses.includes(run.status)) return 'active';
    if (run.status === 'failed' && run.currentStage === 'verification') return 'active';
    return 'locked';
  }

  if (!hasResearchCard) return 'locked';
  return cardPublished ? 'complete' : 'active';
}

export function completedResearchRunZones(snapshot: RunSnapshot | null): number {
  const zoneIds: ZoneId[] = ['observatory', 'lab', 'compute', 'paper', 'plaza'];
  return zoneIds.filter((zoneId) => getResearchRunZoneStatus(zoneId, snapshot) === 'complete').length;
}

export function runStatusLabel(status: RunStatus): string {
  const labels: Record<RunStatus, string> = {
    draft: '等待研究假设',
    hypothesis_ready: '实验设计就绪',
    queued: '实验排队中',
    running: '实验运行中',
    analysis_ready: '结果待分析',
    analyzing: '正在分析',
    verification_required: '等待复现验证',
    verifying: '正在复现验证',
    completed: '研究流程已验证',
    failed: '运行失败',
    cancelled: '运行已取消',
  };
  return labels[status];
}

export function artifactZoneCount(snapshot: RunSnapshot | null): number {
  if (!snapshot) return 0;
  const kinds = new Set(snapshot.artifacts.map((artifact) => artifact.kind));
  return [
    kinds.has('topic_brief'),
    kinds.has('hypothesis'),
    kinds.has('raw_metrics'),
    kinds.has('analysis') || kinds.has('verification_bundle'),
    kinds.has('research_card'),
  ].filter(Boolean).length;
}
