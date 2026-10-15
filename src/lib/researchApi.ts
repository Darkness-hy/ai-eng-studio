import { cloudEnabled, getSupabase } from './supabase';

const configuredBaseUrl = (import.meta.env.VITE_RESEARCH_API_URL as string | undefined)?.trim();
const devAccessToken = import.meta.env.DEV
  ? (import.meta.env.VITE_RESEARCH_DEV_TOKEN as string | undefined)?.trim()
  : undefined;

export const researchApiConfigured = Boolean(configuredBaseUrl);

export type RunStatus =
  | 'draft'
  | 'hypothesis_ready'
  | 'queued'
  | 'running'
  | 'analysis_ready'
  | 'analyzing'
  | 'verification_required'
  | 'verifying'
  | 'completed'
  | 'failed'
  | 'cancelled';

export type ResearchStage =
  | 'topic'
  | 'hypothesis'
  | 'experiment'
  | 'analysis'
  | 'verification'
  | 'publication';

export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'timed_out';
export type ExpectedDirection = 'increase' | 'decrease' | 'no_change';
export type VerificationStatus = 'unverified' | 'passed' | 'failed';
export type ArtifactVisibility = 'private' | 'public';
export type ArtifactKind =
  | 'topic_brief'
  | 'hypothesis'
  | 'experiment_spec'
  | 'source_bundle'
  | 'experiment_log'
  | 'raw_metrics'
  | 'analysis'
  | 'verification_bundle'
  | 'research_card';

export type ResearchVariant = {
  id: string;
  label: string;
  pipeline: string[];
};

export type ResearchTemplate = {
  id: string;
  capstoneNumber: number;
  version: string;
  name: string;
  nameZh: string;
  researchQuestion: string;
  dataset: string;
  benchmarkKind: 'public_snapshot' | 'packaged_dataset' | 'deterministic_fixture';
  provenance: string[];
  independentVariable: string;
  primaryMetric: string;
  primaryMetricLabel: string;
  expectedDirection: ExpectedDirection;
  effectThreshold: number;
  variants: ResearchVariant[];
  bootstrapSeed: number;
  bootstrapReplicates: number;
  secondaryMetrics: string[];
  limitations: string[];
  externalValidation: string[];
  zeroApiCost: true;
  runtimeMode: 'offline';
  budget: ResearchRun['budget'];
};

export type RunFailure = {
  code: string;
  message: string;
};

export type ResearchRun = {
  id: string;
  topic: string;
  researchQuestion: string;
  engine: 'minimal';
  templateId: string;
  templateVersion: string;
  status: RunStatus;
  currentStage: ResearchStage;
  version: number;
  budget: {
    cpuSecondsPerJob: number;
    memoryMbPerJob: number;
    maxConcurrentJobs: number;
    maxAttemptsPerVariant: number;
  };
  retryCount: number;
  failure: RunFailure | null;
  createdAt: string;
  updatedAt: string;
  completedAt: string | null;
  cancelledAt: string | null;
};

export type ResearchHypothesis = {
  id: string;
  runId: string;
  statement: string;
  independentVariable: string;
  dependentMetric: string;
  expectedDirection: ExpectedDirection;
  rationale: string;
  status: 'draft' | 'submitted' | 'accepted' | 'rejected';
  createdAt: string;
  updatedAt: string;
  acceptedAt: string | null;
};

export type AggregateMetric = {
  primaryScoreMean: number;
  primaryScoreSampleStd: number;
  secondaryMeans: Record<string, number>;
  caseLatencyMsMean: number;
  caseLatencyMsP95: number;
  caseCount: number;
};

export type ExperimentMetrics = {
  caseResults: Array<{
    caseId: string;
    primaryScore: number;
    secondary: Record<string, number>;
    latencyMs: number;
    evidence: string[];
  }>;
  bootstrapResults: Array<{
    replicate: number;
    primaryScore: number;
    sampleSha256: string;
  }>;
  aggregate: AggregateMetric;
  resultSha256: string;
};

export type ExperimentJob = {
  id: string;
  runId: string;
  hypothesisId: string;
  suiteId: string;
  variant: string;
  templateId: string;
  templateVersion: string;
  attempt: number;
  status: JobStatus;
  bootstrapSeed: number;
  bootstrapReplicates: number;
  config: Record<string, unknown>;
  resourceLimits: {
    cpuCount: number;
    memoryMb: number;
    timeoutSeconds: number;
    network: boolean;
  };
  metrics: ExperimentMetrics | null;
  error: RunFailure | null;
  queuedAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  durationMs: number | null;
};

export type ResearchArtifact = {
  id: string;
  runId: string;
  jobId: string | null;
  stage: ResearchStage;
  kind: ArtifactKind;
  version: number;
  content: Record<string, unknown> | null;
  filename: string | null;
  mimeType: string | null;
  byteSize: number | null;
  sha256: string | null;
  verificationStatus: VerificationStatus;
  visibility: ArtifactVisibility;
  downloadUrl: string | null;
  downloadUrlExpiresAt: string | null;
  createdAt: string;
  publishedAt: string | null;
};

export type ResearchEvent = {
  id: string;
  runId: string;
  actor: 'user' | 'runner' | 'verifier' | 'system';
  eventType: string;
  stage: ResearchStage;
  payload: Record<string, unknown>;
  createdAt: string;
};

export type RunSnapshot = {
  run: ResearchRun;
  hypothesis: ResearchHypothesis | null;
  jobs: ExperimentJob[];
  artifacts: ResearchArtifact[];
  lastEventId: string | null;
};

export type HypothesisInput = {
  statement: string;
  expectedDirection: ExpectedDirection;
  rationale: string;
};

type ListRunsResponse = { items: ResearchRun[]; nextCursor: string | null };
type TemplateListResponse = { items: ResearchTemplate[] };
type CommandAccepted = { run: ResearchRun; eventId: string };
type ArtifactCommandResult = { artifact: ResearchArtifact; runVersion: number; eventId: string };

type ErrorEnvelope = {
  error?: {
    code?: string;
    message?: string;
    requestId?: string;
    details?: Record<string, unknown>;
  };
  detail?: string | Array<{ msg?: string }>;
};

export class ResearchApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId?: string;
  readonly details?: Record<string, unknown>;

  constructor(
    message: string,
    options: { status?: number; code?: string; requestId?: string; details?: Record<string, unknown> } = {},
  ) {
    super(message);
    this.name = 'ResearchApiError';
    this.status = options.status ?? 0;
    this.code = options.code ?? 'network_error';
    this.requestId = options.requestId;
    this.details = options.details;
  }
}

export function formatResearchApiError(error: unknown): string {
  if (error instanceof ResearchApiError) {
    const request = error.requestId ? `（请求 ${error.requestId}）` : '';
    return `${error.message}${request}`;
  }
  if (error instanceof Error) return error.message;
  return typeof error === 'string' ? error : '科研服务发生未知错误';
}

async function accessToken(): Promise<string> {
  if (devAccessToken) return devAccessToken;
  if (cloudEnabled) {
    const { data, error } = await getSupabase().auth.getSession();
    if (error) throw new ResearchApiError(`无法读取登录状态：${error.message}`, { code: 'authentication_required' });
    if (data.session?.access_token) return data.session.access_token;
  }
  throw new ResearchApiError('请先登录，再进入联网科研流程。', { code: 'authentication_required', status: 401 });
}

async function requestJson<T>(
  path: string,
  options: {
    method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
    body?: unknown;
    idempotent?: boolean;
    signal?: AbortSignal;
  } = {},
): Promise<T> {
  if (!configuredBaseUrl) {
    throw new ResearchApiError('科研服务未配置（缺少 VITE_RESEARCH_API_URL）', { code: 'not_configured' });
  }
  const token = await accessToken();
  let response: Response;
  try {
    response = await fetch(`${configuredBaseUrl.replace(/\/$/, '')}${path}`, {
      method: options.method ?? 'GET',
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
      signal: options.signal,
      headers: {
        Accept: 'application/json',
        Authorization: `Bearer ${token}`,
        ...(options.body === undefined ? {} : { 'Content-Type': 'application/json' }),
        ...(options.idempotent ? { 'Idempotency-Key': crypto.randomUUID() } : {}),
      },
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error;
    throw new ResearchApiError(`无法连接科研服务：${error instanceof Error ? error.message : '网络错误'}`);
  }

  if (!response.ok) throw await responseError(response);
  return (await response.json()) as T;
}

async function responseError(response: Response): Promise<ResearchApiError> {
  let envelope: ErrorEnvelope = {};
  try {
    envelope = (await response.json()) as ErrorEnvelope;
  } catch {
    // Keep the HTTP fallback below when an upstream proxy returns an empty or non-JSON body.
  }
  const detail = Array.isArray(envelope.detail)
    ? envelope.detail.map((item) => item.msg).filter(Boolean).join('；')
    : envelope.detail;
  return new ResearchApiError(
    envelope.error?.message || detail || `科研服务返回错误 (${response.status})`,
    {
      status: response.status,
      code: envelope.error?.code ?? 'http_error',
      requestId: envelope.error?.requestId,
      details: envelope.error?.details,
    },
  );
}

export const researchApi = {
  listTemplates(signal?: AbortSignal) {
    return requestJson<TemplateListResponse>('/v1/research-templates', { signal });
  },

  listRuns(signal?: AbortSignal) {
    return requestJson<ListRunsResponse>('/v1/research-runs?limit=20', { signal });
  },

  getRun(runId: string, signal?: AbortSignal) {
    return requestJson<RunSnapshot>(`/v1/research-runs/${runId}`, { signal });
  },

  createRun(topic: string, templateId: string) {
    return requestJson<RunSnapshot>('/v1/research-runs', {
      method: 'POST',
      body: { topic, templateId },
      idempotent: true,
    });
  },

  putHypothesis(
    runId: string,
    expectedVersion: number,
    input: HypothesisInput,
    contract: Pick<ResearchTemplate, 'independentVariable' | 'primaryMetric'>,
  ) {
    return requestJson<RunSnapshot>(`/v1/research-runs/${runId}/hypothesis`, {
      method: 'PUT',
      body: {
        expectedVersion,
        statement: input.statement,
        independentVariable: contract.independentVariable,
        dependentMetric: contract.primaryMetric,
        expectedDirection: input.expectedDirection,
        rationale: input.rationale,
      },
    });
  },

  enqueueExperiments(runId: string, expectedVersion: number) {
    return requestJson<{ run: ResearchRun; suiteId: string; jobs: ExperimentJob[]; eventId: string }>(
      `/v1/research-runs/${runId}/experiment-jobs`,
      {
        method: 'POST',
        body: { expectedVersion, benchmark: { includeAllVariants: true } },
        idempotent: true,
      },
    );
  },

  command(runId: string, command: 'analysis' | 'verify' | 'retry' | 'cancel', expectedVersion: number) {
    return requestJson<CommandAccepted>(`/v1/research-runs/${runId}/${command}`, {
      method: 'POST',
      body: { expectedVersion },
      idempotent: true,
    });
  },

  publish(artifactId: string, expectedVersion: number) {
    return requestJson<ArtifactCommandResult>(`/v1/research-artifacts/${artifactId}/publish`, {
      method: 'POST',
      body: { expectedVersion },
      idempotent: true,
    });
  },

  unpublish(artifactId: string, expectedVersion: number) {
    return requestJson<ArtifactCommandResult>(`/v1/research-artifacts/${artifactId}/publication`, {
      method: 'DELETE',
      body: { expectedVersion },
      idempotent: true,
    });
  },
};

export async function streamResearchEvents(
  runId: string,
  after: string | null,
  onEvent: (event: ResearchEvent) => void,
  signal: AbortSignal,
  onOpen?: () => void,
): Promise<void> {
  if (!configuredBaseUrl) return;
  const token = await accessToken();
  const query = after ? `?after=${encodeURIComponent(after)}` : '';
  let response: Response;
  try {
    response = await fetch(
      `${configuredBaseUrl.replace(/\/$/, '')}/v1/research-runs/${runId}/events${query}`,
      {
        signal,
        headers: { Accept: 'text/event-stream', Authorization: `Bearer ${token}` },
      },
    );
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') return;
    throw new ResearchApiError(`科研事件流连接失败：${error instanceof Error ? error.message : '网络错误'}`);
  }
  if (!response.ok) throw await responseError(response);
  if (!response.body) throw new ResearchApiError('科研事件流没有返回内容');
  onOpen?.();

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (!signal.aborted) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let boundary = buffer.indexOf('\n\n');
      while (boundary >= 0) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const data = block
          .split('\n')
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).trimStart())
          .join('\n');
        if (data) onEvent(JSON.parse(data) as ResearchEvent);
        boundary = buffer.indexOf('\n\n');
      }
    }
  } finally {
    reader.releaseLock();
  }
}
