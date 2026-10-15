import { useCallback, useEffect, useMemo, useState } from 'react';
import { useAuth } from './auth';
import {
  formatResearchApiError,
  researchApi,
  ResearchApiError,
  streamResearchEvents,
  type ExpectedDirection,
  type ResearchTemplate,
  type RunSnapshot,
} from './researchApi';

const ACTIVE_RUN_STORAGE = 'research-game:active-run';

export type ResearchCommand =
  | 'create'
  | 'hypothesis'
  | 'experiment'
  | 'analysis'
  | 'verification'
  | 'retry'
  | 'cancel'
  | 'publish'
  | 'unpublish';

type HypothesisDraft = {
  statement: string;
  expectedDirection: ExpectedDirection;
  rationale: string;
};

export function useResearchRun() {
  const auth = useAuth();
  const storageKey = useMemo(
    () => `${ACTIVE_RUN_STORAGE}:${auth.profile?.id ?? (auth.enabled ? 'cloud' : 'dev')}`,
    [auth.enabled, auth.profile?.id],
  );
  const [snapshot, setSnapshot] = useState<RunSnapshot | null>(null);
  const [templates, setTemplates] = useState<ResearchTemplate[]>([]);
  const [loading, setLoading] = useState(true);
  const [pending, setPending] = useState<ResearchCommand | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [streamState, setStreamState] = useState<'idle' | 'connecting' | 'live' | 'retrying'>('idle');

  const adoptSnapshot = useCallback(
    (next: RunSnapshot | null) => {
      setSnapshot(next);
      if (next) window.localStorage.setItem(storageKey, next.run.id);
      else window.localStorage.removeItem(storageKey);
    },
    [storageKey],
  );

  const refresh = useCallback(
    async (silent = true): Promise<RunSnapshot | null> => {
      const runId = window.localStorage.getItem(storageKey);
      if (!runId) return null;
      if (!silent) setLoading(true);
      try {
        const next = await researchApi.getRun(runId);
        adoptSnapshot(next);
        setError(null);
        return next;
      } catch (caught) {
        if (caught instanceof ResearchApiError && caught.status === 404) adoptSnapshot(null);
        setError(formatResearchApiError(caught));
        return null;
      } finally {
        if (!silent) setLoading(false);
      }
    },
    [adoptSnapshot, storageKey],
  );

  useEffect(() => {
    if (auth.loading) return;
    const controller = new AbortController();

    const initialize = async () => {
      setLoading(true);
      setError(null);
      try {
        const catalog = await researchApi.listTemplates(controller.signal);
        if (controller.signal.aborted) return;
        setTemplates(catalog.items);
        const storedRunId = window.localStorage.getItem(storageKey);
        if (storedRunId) {
          try {
            const stored = await researchApi.getRun(storedRunId, controller.signal);
            if (!controller.signal.aborted) adoptSnapshot(stored);
            return;
          } catch (caught) {
            if (!(caught instanceof ResearchApiError) || caught.status !== 404) throw caught;
            window.localStorage.removeItem(storageKey);
          }
        }

        const runs = await researchApi.listRuns(controller.signal);
        if (controller.signal.aborted) return;
        if (!runs.items[0]) {
          adoptSnapshot(null);
          return;
        }
        const latest = await researchApi.getRun(runs.items[0].id, controller.signal);
        if (!controller.signal.aborted) adoptSnapshot(latest);
      } catch (caught) {
        if (!(caught instanceof DOMException && caught.name === 'AbortError')) {
          setError(formatResearchApiError(caught));
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    };

    void initialize();
    return () => controller.abort();
  }, [adoptSnapshot, auth.loading, storageKey]);

  const runId = snapshot?.run.id ?? null;
  const lastEventId = snapshot?.lastEventId ?? null;
  const activeTemplate =
    templates.find((template) => template.id === snapshot?.run.templateId) ?? null;

  useEffect(() => {
    if (!runId) {
      return;
    }
    const controller = new AbortController();
    let cursor = lastEventId;
    let retryTimer: number | undefined;
    let refreshTimer: number | undefined;

    const reconnectDelay = () =>
      new Promise<void>((resolve) => {
        retryTimer = window.setTimeout(resolve, 2000);
      });

    const listen = async () => {
      setStreamState('connecting');
      while (!controller.signal.aborted) {
        try {
          await streamResearchEvents(
            runId,
            cursor,
            (event) => {
              cursor = event.id;
              setStreamState('live');
              window.clearTimeout(refreshTimer);
              refreshTimer = window.setTimeout(() => void refresh(), 60);
            },
            controller.signal,
            () => setStreamState('live'),
          );
          if (!controller.signal.aborted) {
            setStreamState('retrying');
            await reconnectDelay();
          }
        } catch (caught) {
          if (controller.signal.aborted) return;
          console.warn('[research] event stream disconnected', caught);
          setStreamState('retrying');
          await reconnectDelay();
        }
      }
    };

    void listen();
    return () => {
      controller.abort();
      window.clearTimeout(retryTimer);
      window.clearTimeout(refreshTimer);
    };
  }, [lastEventId, refresh, runId]);

  const runMutation = useCallback(
    async <T,>(command: ResearchCommand, mutation: () => Promise<T>): Promise<T | null> => {
      setPending(command);
      setError(null);
      try {
        const result = await mutation();
        await refresh();
        return result;
      } catch (caught) {
        if (caught instanceof ResearchApiError && caught.code === 'version_conflict') await refresh();
        setError(formatResearchApiError(caught));
        return null;
      } finally {
        setPending(null);
      }
    },
    [refresh],
  );

  const createRun = useCallback(
    async (topic: string, templateId: string): Promise<RunSnapshot | null> => {
      setPending('create');
      setError(null);
      try {
        const created = await researchApi.createRun(topic.trim(), templateId);
        adoptSnapshot(created);
        return created;
      } catch (caught) {
        setError(formatResearchApiError(caught));
        return null;
      } finally {
        setPending(null);
        setLoading(false);
      }
    },
    [adoptSnapshot],
  );

  const submitHypothesis = useCallback(
    (draft: HypothesisDraft) => {
      if (!snapshot) return Promise.resolve(null);
      if (!activeTemplate) {
        setError('当前 ResearchRun 的实验模板不在服务目录中。');
        return Promise.resolve(null);
      }
      return runMutation('hypothesis', () =>
        researchApi.putHypothesis(snapshot.run.id, snapshot.run.version, draft, activeTemplate),
      );
    },
    [activeTemplate, runMutation, snapshot],
  );

  const enqueueExperiments = useCallback(() => {
    if (!snapshot) return Promise.resolve(null);
    return runMutation('experiment', () =>
      researchApi.enqueueExperiments(snapshot.run.id, snapshot.run.version),
    );
  }, [runMutation, snapshot]);

  const executeRunCommand = useCallback(
    (command: 'analysis' | 'verify' | 'retry' | 'cancel', pendingCommand: ResearchCommand) => {
      if (!snapshot) return Promise.resolve(null);
      return runMutation(pendingCommand, () =>
        researchApi.command(snapshot.run.id, command, snapshot.run.version),
      );
    },
    [runMutation, snapshot],
  );

  const publishArtifact = useCallback(
    (artifactId: string, publish: boolean) => {
      if (!snapshot) return Promise.resolve(null);
      return runMutation(publish ? 'publish' : 'unpublish', () =>
        publish
          ? researchApi.publish(artifactId, snapshot.run.version)
          : researchApi.unpublish(artifactId, snapshot.run.version),
      );
    },
    [runMutation, snapshot],
  );

  return {
    snapshot,
    templates,
    activeTemplate,
    loading: loading || auth.loading,
    pending,
    error,
    streamState,
    clearError: () => setError(null),
    refresh: () => refresh(false),
    createRun,
    submitHypothesis,
    enqueueExperiments,
    analyze: () => executeRunCommand('analysis', 'analysis'),
    verify: () => executeRunCommand('verify', 'verification'),
    retry: () => executeRunCommand('retry', 'retry'),
    cancel: () => executeRunCommand('cancel', 'cancel'),
    publishArtifact,
  };
}
