import { useState, type FormEvent } from 'react';
import { Link, Navigate, useParams } from 'react-router-dom';
import {
  type ExpectedDirection,
  type HypothesisInput,
  type ResearchArtifact,
  type ResearchTemplate,
  type RunSnapshot,
} from '../lib/researchApi';
import {
  getZone,
  nextZone,
  researcherAsset,
  statusLabel,
  zones,
  type Zone,
  type ZoneId,
  type ZoneStatus,
} from '../lib/researchGame';
import {
  artifactZoneCount,
  getResearchRunZoneStatus,
  runStatusLabel,
} from '../lib/researchRunGame';
import { useResearchRun, type ResearchCommand } from '../lib/useResearchRun';

export function ConnectedResearchZonePage() {
  const { zoneId } = useParams<{ zoneId: string }>();
  const zone = getZone(zoneId);
  const research = useResearchRun();

  if (!zone) return <Navigate to="/research-game" replace />;

  const status = getResearchRunZoneStatus(zone.id, research.snapshot);
  const followingZone = nextZone(zone.id);
  const metrics = zoneMetrics(zone.id, research.snapshot, research.activeTemplate);

  return (
    <div className="research-game-ui bg-canvas">
      <div className="mx-auto max-w-6xl px-4 py-6 sm:px-5 lg:py-8">
        <header className="border-b border-hairline pb-5">
          <Link to="/research-game" className="text-[13px] text-faint transition-colors hover:text-ink">
            ← 返回科研岛
          </Link>
          <div className="mt-5 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <p className="font-mono text-[10px] tracking-[0.18em] text-faint">
                {research.activeTemplate
                  ? `CAPSTONE ${String(research.activeTemplate.capstoneNumber).padStart(2, '0')} · ZERO API · RESEARCHRUN`
                  : '17 ZERO-API CAPSTONES · RESEARCHRUN'}
              </p>
              <h1 className="mt-2 font-serif text-[29px] font-semibold sm:text-[38px]">{zone.name}</h1>
              <p className="mt-2 text-[13.5px] leading-relaxed text-faint">
                {connectedZoneSubtitle(zone.id, research.activeTemplate)}
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              {research.snapshot && (
                <span className="font-mono text-[9.5px] tracking-[0.1em] text-faint">
                  RUN {research.snapshot.run.id.slice(0, 8)} · V{research.snapshot.run.version}
                </span>
              )}
              <StatusPill status={status} />
            </div>
          </div>
        </header>

        <nav className="flex gap-1 overflow-x-auto border-b border-hairline py-3" aria-label="科研岛区域">
          {zones.map((item, index) => {
            const itemStatus = getResearchRunZoneStatus(item.id, research.snapshot);
            return (
              <Link
                key={item.id}
                to={`/research-game/${item.id}`}
                className={`flex min-h-9 shrink-0 items-center gap-2 rounded-md px-3 text-[12.5px] transition-colors ${
                  item.id === zone.id ? 'bg-ink text-paper' : 'text-faint hover:bg-bone hover:text-ink'
                }`}
              >
                <span className="font-mono text-[9.5px]">{index + 1}</span>
                {item.shortName}
                <span className="font-mono text-[8.5px] opacity-70">{statusLabel(itemStatus)}</span>
              </Link>
            );
          })}
        </nav>

        {research.error && (
          <div className="mt-5 flex flex-col gap-3 border-l-2 border-ink-red bg-pale-red px-4 py-3 text-[13px] sm:flex-row sm:items-center sm:justify-between">
            <p className="leading-relaxed text-ink-red">操作失败：{research.error}</p>
            <div className="flex shrink-0 gap-4">
              <button
                type="button"
                onClick={() => void research.refresh()}
                className="font-medium text-ink-red underline underline-offset-4"
              >
                重新读取
              </button>
              <button
                type="button"
                onClick={research.clearError}
                className="text-faint underline underline-offset-4"
              >
                关闭
              </button>
            </div>
          </div>
        )}

        <section className="grid border-b border-hairline lg:grid-cols-[minmax(0,1.05fr)_minmax(320px,0.95fr)]">
          <div className="flex min-h-[280px] items-center justify-center bg-bone px-5 py-8 sm:min-h-[360px]">
            <img src={zone.asset} alt="" className="max-h-[350px] max-w-full object-contain" draggable={false} />
          </div>
          <div className="flex flex-col justify-center border-t border-hairline px-5 py-8 lg:border-l lg:border-t-0 lg:px-8">
            <div className={`w-fit rounded-full border px-2.5 py-1 font-mono text-[9.5px] tracking-[0.14em] ${zone.accent}`}>
              {statusLabel(status)}
            </div>
            <h2 className="mt-5 font-serif text-[23px] font-semibold">当前任务</h2>
            <p className="mt-2 text-[14px] leading-relaxed">
              {zoneMission(zone.id, research.snapshot, research.activeTemplate)}
            </p>
            <div className="mt-5 grid grid-cols-3 gap-2">
              {metrics.map((metric) => (
                <MiniMetric key={metric.label} label={metric.label} value={metric.value} />
              ))}
            </div>
          </div>
        </section>

        <main className="border-b border-hairline py-8">
          {research.loading && !research.snapshot ? (
            <div className="min-h-48 border-l-2 border-ink-blue pl-4">
              <div className="font-mono text-[10px] tracking-[0.14em] text-faint">LOADING RESEARCHRUN</div>
              <p className="mt-3 text-[14px]">正在读取当前研究运行……</p>
            </div>
          ) : (
            <ZoneWorkspace
              zoneId={zone.id}
              snapshot={research.snapshot}
              templates={research.templates}
              activeTemplate={research.activeTemplate}
              pending={research.pending}
              onCreate={async (topic, templateId) =>
                (await research.createRun(topic, templateId)) !== null
              }
              onHypothesis={(draft) => void research.submitHypothesis(draft)}
              onExperiment={() => void research.enqueueExperiments()}
              onAnalyze={() => void research.analyze()}
              onVerify={() => void research.verify()}
              onRetry={() => void research.retry()}
              onCancel={() => void research.cancel()}
              onPublication={(artifactId, publish) => void research.publishArtifact(artifactId, publish)}
            />
          )}
        </main>

        <footer className="flex flex-col gap-4 py-6 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-4">
            <div className="h-14 w-11 overflow-hidden">
              <img src={researcherAsset} alt="" className="h-full w-full object-contain object-top" />
            </div>
            <div>
              <div className="font-mono text-[9.5px] tracking-[0.14em] text-faint">VERIFIABLE RECORD</div>
              <div className="mt-1 text-[13px]">
                已获得 {artifactZoneCount(research.snapshot)} / 5 类研究产物
              </div>
            </div>
          </div>
          {followingZone ? (
            <Link
              to={`/research-game/${followingZone.id}`}
              className="text-[13px] font-medium text-ink-blue transition-colors hover:text-ink"
            >
              下一地区：{followingZone.name} →
            </Link>
          ) : (
            <Link to="/research-game" className="text-[13px] font-medium text-ink-blue transition-colors hover:text-ink">
              返回科研岛总览 →
            </Link>
          )}
        </footer>
      </div>
    </div>
  );
}

/** ResearchRun controls embedded in an in-world terminal or whiteboard. */
export function EmbeddedConnectedResearchWorkspace({ zoneId }: { zoneId: ZoneId }) {
  const research = useResearchRun();
  const zone = getZone(zoneId);

  if (!zone) return null;

  const status = getResearchRunZoneStatus(zoneId, research.snapshot);
  return (
    <div className="research-game-ui">
      <div className="flex flex-col gap-3 border-b border-hairline pb-5 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="font-mono text-[9.5px] tracking-[0.16em] text-faint">
            LIVE RESEARCHRUN · HUMAN CONFIRMATION REQUIRED
          </p>
          <h3 className="mt-2 font-serif text-[24px] font-semibold">{embeddedWorkspaceName(zoneId)}</h3>
          <p className="mt-2 max-w-3xl text-[12.5px] leading-relaxed text-faint">
            {connectedZoneSubtitle(zoneId, research.activeTemplate)}
          </p>
        </div>
        <StatusPill status={status} />
      </div>

      {research.error && (
        <div className="mt-4 flex flex-col gap-3 border-l-2 border-ink-red bg-pale-red px-4 py-3 text-[12.5px] sm:flex-row sm:items-center sm:justify-between">
          <p className="leading-relaxed text-ink-red">操作失败：{research.error}</p>
          <div className="flex shrink-0 gap-4">
            <button
              type="button"
              onClick={() => void research.refresh()}
              className="font-medium text-ink-red underline underline-offset-4"
            >
              重新读取
            </button>
            <button
              type="button"
              onClick={research.clearError}
              className="text-faint underline underline-offset-4"
            >
              关闭
            </button>
          </div>
        </div>
      )}

      <div className="py-6">
        {research.loading && !research.snapshot ? (
          <div className="min-h-40 border-l-2 border-ink-blue pl-4">
            <div className="font-mono text-[10px] tracking-[0.14em] text-faint">
              LOADING RESEARCHRUN
            </div>
            <p className="mt-3 text-[14px]">正在读取当前研究运行……</p>
          </div>
        ) : (
          <ZoneWorkspace
            zoneId={zoneId}
            snapshot={research.snapshot}
            templates={research.templates}
            activeTemplate={research.activeTemplate}
            pending={research.pending}
            onCreate={async (topic, templateId) =>
              (await research.createRun(topic, templateId)) !== null
            }
            onHypothesis={(draft) => void research.submitHypothesis(draft)}
            onExperiment={() => void research.enqueueExperiments()}
            onAnalyze={() => void research.analyze()}
            onVerify={() => void research.verify()}
            onRetry={() => void research.retry()}
            onCancel={() => void research.cancel()}
            onPublication={(artifactId, publish) =>
              void research.publishArtifact(artifactId, publish)
            }
          />
        )}
      </div>
    </div>
  );
}

function ZoneWorkspace({
  zoneId,
  snapshot,
  templates,
  activeTemplate,
  pending,
  onCreate,
  onHypothesis,
  onExperiment,
  onAnalyze,
  onVerify,
  onRetry,
  onCancel,
  onPublication,
}: {
  zoneId: ZoneId;
  snapshot: RunSnapshot | null;
  templates: ResearchTemplate[];
  activeTemplate: ResearchTemplate | null;
  pending: ResearchCommand | null;
  onCreate: (topic: string, templateId: string) => Promise<boolean>;
  onHypothesis: (draft: HypothesisInput) => void;
  onExperiment: () => void;
  onAnalyze: () => void;
  onVerify: () => void;
  onRetry: () => void;
  onCancel: () => void;
  onPublication: (artifactId: string, publish: boolean) => void;
}) {
  if (zoneId === 'observatory') {
    return (
      <ObservatoryWorkspace
        snapshot={snapshot}
        templates={templates}
        activeTemplate={activeTemplate}
        pending={pending}
        onCreate={onCreate}
      />
    );
  }
  if (zoneId === 'lab') {
    return (
      <HypothesisWorkspace
        snapshot={snapshot}
        template={activeTemplate}
        pending={pending}
        onSubmit={onHypothesis}
      />
    );
  }
  if (zoneId === 'compute') {
    return (
      <ComputeWorkspace
        snapshot={snapshot}
        template={activeTemplate}
        pending={pending}
        onExperiment={onExperiment}
        onRetry={onRetry}
        onCancel={onCancel}
      />
    );
  }
  if (zoneId === 'paper') {
    return (
      <PaperWorkspace
        snapshot={snapshot}
        template={activeTemplate}
        pending={pending}
        onAnalyze={onAnalyze}
        onVerify={onVerify}
        onRetry={onRetry}
        onCancel={onCancel}
      />
    );
  }
  return (
    <PlazaWorkspace
      snapshot={snapshot}
      template={activeTemplate}
      pending={pending}
      onPublication={onPublication}
    />
  );
}

function ObservatoryWorkspace({
  snapshot,
  templates,
  activeTemplate,
  pending,
  onCreate,
}: {
  snapshot: RunSnapshot | null;
  templates: ResearchTemplate[];
  activeTemplate: ResearchTemplate | null;
  pending: ResearchCommand | null;
  onCreate: (topic: string, templateId: string) => Promise<boolean>;
}) {
  const initialTemplate = activeTemplate ?? templates[0] ?? null;
  const [templateId, setTemplateId] = useState(initialTemplate?.id ?? '');
  const [topic, setTopic] = useState(initialTemplate?.nameZh ?? '零 API 科研实验');
  const [creatingAnother, setCreatingAnother] = useState(false);
  const showForm = !snapshot || creatingAnother;
  const selectedTemplate = templates.find((template) => template.id === templateId) ?? templates[0] ?? null;

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (selectedTemplate && (await onCreate(topic, selectedTemplate.id))) setCreatingAnother(false);
  };

  if (!showForm && snapshot) {
    return (
      <div className="grid gap-8 lg:grid-cols-[minmax(0,1.2fr)_minmax(280px,0.8fr)]">
        <section>
          <SectionEyebrow>TOPIC BRIEF</SectionEyebrow>
          <h2 className="mt-2 font-serif text-[24px] font-semibold">{snapshot.run.topic}</h2>
          <p className="mt-4 text-[13px] leading-relaxed text-faint">固定研究问题</p>
          <p className="mt-1 border-l-2 border-ink-blue pl-3 text-[14px] leading-relaxed">
            {snapshot.run.researchQuestion}
          </p>
        </section>
        <aside className="border-t border-hairline pt-6 lg:border-l lg:border-t-0 lg:pl-8 lg:pt-0">
          <SectionEyebrow>RUN CONTROL</SectionEyebrow>
          <p className="mt-3 text-[13px] leading-relaxed text-faint">
            {activeTemplate
              ? `毕业项目 ${String(activeTemplate.capstoneNumber).padStart(2, '0')} · ${activeTemplate.nameZh}。运行时离线，网络禁用，付费 API 成本为 $0。`
              : '当前运行的模板目录信息暂不可用。'}
          </p>
          {activeTemplate && (
            <dl className="mt-4 space-y-3 text-[12.5px]">
              <Definition label="数据" value={activeTemplate.dataset} />
              <Definition label="主指标" value={activeTemplate.primaryMetricLabel} />
              <Definition label="配置数" value={`${activeTemplate.variants.length} 组`} />
            </dl>
          )}
          <button type="button" onClick={() => setCreatingAnother(true)} className={secondaryButtonClass}>
            创建新的研究运行
          </button>
        </aside>
      </div>
    );
  }

  return (
    <form onSubmit={(event) => void submit(event)} className="grid gap-8 lg:grid-cols-[minmax(0,1.2fr)_minmax(280px,0.8fr)]">
      <section>
        <SectionEyebrow>NEW RESEARCHRUN</SectionEyebrow>
        <h2 className="mt-2 font-serif text-[24px] font-semibold">定义本次研究课题</h2>
        <label className="mt-6 block text-[12.5px] font-medium" htmlFor="research-template">
          毕业项目
        </label>
        <select
          id="research-template"
          value={selectedTemplate?.id ?? ''}
          onChange={(event) => {
            const next = templates.find((template) => template.id === event.target.value);
            setTemplateId(event.target.value);
            if (next) setTopic(next.nameZh);
          }}
          required
          className="mt-2 min-h-11 w-full rounded-md border border-hairline bg-paper px-3 text-[14px] outline-none transition-colors focus:border-ink-blue"
        >
          {templates.map((template) => (
            <option key={template.id} value={template.id}>
              {String(template.capstoneNumber).padStart(2, '0')} · {template.nameZh}
            </option>
          ))}
        </select>
        <label className="mt-6 block text-[12.5px] font-medium" htmlFor="research-topic">
          课题名称
        </label>
        <input
          id="research-topic"
          value={topic}
          onChange={(event) => setTopic(event.target.value)}
          minLength={3}
          maxLength={200}
          required
          className="mt-2 min-h-11 w-full rounded-md border border-hairline bg-paper px-3 text-[14px] outline-none transition-colors focus:border-ink-blue"
        />
        <div className="mt-5 border-l-2 border-ink-blue pl-3">
          <div className="font-mono text-[9.5px] tracking-[0.14em] text-faint">FIXED RESEARCH QUESTION</div>
          <p className="mt-2 text-[13.5px] leading-relaxed">
            {selectedTemplate?.researchQuestion ?? '正在读取实验模板……'}
          </p>
        </div>
      </section>
      <aside className="border-t border-hairline pt-6 lg:border-l lg:border-t-0 lg:pl-8 lg:pt-0">
        <SectionEyebrow>TEMPLATE</SectionEyebrow>
        <dl className="mt-4 space-y-3 text-[12.5px]">
          <Definition label="数据" value={selectedTemplate?.dataset ?? '—'} />
          <Definition label="类型" value={benchmarkKindLabel(selectedTemplate?.benchmarkKind)} />
          <Definition label="主指标" value={selectedTemplate?.primaryMetricLabel ?? '—'} />
          <Definition
            label="实验配置"
            value={selectedTemplate?.variants.map((variant) => variant.label).join(' → ') ?? '—'}
          />
          <Definition label="资源" value="1 CPU · 1 GiB · 无网络" />
          <Definition label="Bootstrap" value="1,000 次 · 95% CI" />
          <Definition label="API 费用" value="$0" />
        </dl>
        {selectedTemplate && selectedTemplate.externalValidation.length > 0 && (
          <p className="mt-4 border-l border-hairline pl-3 text-[11.5px] leading-relaxed text-faint">
            尚需外部验证：{selectedTemplate.externalValidation.join('；')}
          </p>
        )}
        <button
          type="submit"
          disabled={pending !== null || !selectedTemplate || topic.trim().length < 3}
          className={primaryButtonClass}
        >
          {pending === 'create' ? '正在创建……' : '创建 ResearchRun'}
        </button>
        {snapshot && (
          <button type="button" onClick={() => setCreatingAnother(false)} className={textButtonClass}>
            取消，保留当前运行
          </button>
        )}
      </aside>
    </form>
  );
}

function HypothesisWorkspace({
  snapshot,
  template,
  pending,
  onSubmit,
}: {
  snapshot: RunSnapshot | null;
  template: ResearchTemplate | null;
  pending: ResearchCommand | null;
  onSubmit: (draft: HypothesisInput) => void;
}) {
  const [statement, setStatement] = useState(
    () => defaultHypothesisStatement(template),
  );
  const [rationale, setRationale] = useState(
    () => defaultHypothesisRationale(template),
  );
  const [expectedDirection, setExpectedDirection] = useState<ExpectedDirection>(
    template?.expectedDirection ?? 'increase',
  );

  if (!snapshot) return <Prerequisite zone="observatory" text="先在模型与系统实验室的项目与数据终端创建一个研究课题。" />;
  if (!template) return <Prerequisite zone="observatory" text="当前实验模板不可用，请重新选择毕业项目。" />;

  if (snapshot.hypothesis) {
    return (
      <div className="grid gap-8 lg:grid-cols-[minmax(0,1.2fr)_minmax(280px,0.8fr)]">
        <section>
          <SectionEyebrow>ACCEPTED HYPOTHESIS</SectionEyebrow>
          <h2 className="mt-2 font-serif text-[24px] font-semibold">已接受的可检验假设</h2>
          <p className="mt-5 border-l-2 border-ink-green pl-3 text-[14px] leading-relaxed">
            {snapshot.hypothesis.statement}
          </p>
          <p className="mt-5 text-[13px] leading-relaxed text-faint">{snapshot.hypothesis.rationale}</p>
        </section>
        <aside className="border-t border-hairline pt-6 lg:border-l lg:border-t-0 lg:pl-8 lg:pt-0">
          <SectionEyebrow>TEST CONTRACT</SectionEyebrow>
          <dl className="mt-4 space-y-3 text-[12.5px]">
            <Definition label="自变量" value={template.independentVariable} />
            <Definition label="主指标" value={template.primaryMetricLabel} />
            <Definition
              label="预期方向"
              value={directionLabel(snapshot.hypothesis.expectedDirection, template.primaryMetricLabel)}
            />
            <Definition label="状态" value="已接受" />
          </dl>
        </aside>
      </div>
    );
  }

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit({ statement: statement.trim(), expectedDirection, rationale: rationale.trim() });
      }}
      className="grid gap-8 lg:grid-cols-[minmax(0,1.2fr)_minmax(280px,0.8fr)]"
    >
      <section>
        <SectionEyebrow>HYPOTHESIS CONTRACT</SectionEyebrow>
        <h2 className="mt-2 font-serif text-[24px] font-semibold">提交可反驳的研究假设</h2>
        <label className="mt-6 block text-[12.5px] font-medium" htmlFor="hypothesis-statement">
          假设陈述
        </label>
        <textarea
          id="hypothesis-statement"
          value={statement}
          onChange={(event) => setStatement(event.target.value)}
          minLength={10}
          maxLength={1000}
          required
          className="mt-2 min-h-28 w-full resize-y rounded-md border border-hairline bg-paper px-3 py-3 text-[14px] leading-relaxed outline-none transition-colors focus:border-ink-blue"
        />
        <label className="mt-5 block text-[12.5px] font-medium" htmlFor="hypothesis-rationale">
          机制解释
        </label>
        <textarea
          id="hypothesis-rationale"
          value={rationale}
          onChange={(event) => setRationale(event.target.value)}
          minLength={10}
          maxLength={2000}
          required
          className="mt-2 min-h-28 w-full resize-y rounded-md border border-hairline bg-paper px-3 py-3 text-[14px] leading-relaxed outline-none transition-colors focus:border-ink-blue"
        />
      </section>
      <aside className="border-t border-hairline pt-6 lg:border-l lg:border-t-0 lg:pl-8 lg:pt-0">
        <SectionEyebrow>EXPECTED DIRECTION</SectionEyebrow>
        <div className="mt-4 grid gap-2">
          {(['increase', 'decrease', 'no_change'] as ExpectedDirection[]).map((direction) => (
            <button
              key={direction}
              type="button"
              onClick={() => setExpectedDirection(direction)}
              className={`min-h-10 rounded-md border px-3 text-left text-[12.5px] transition-colors ${
                expectedDirection === direction
                  ? 'border-ink bg-ink text-paper'
                  : 'border-hairline bg-bone text-faint hover:bg-paper hover:text-ink'
              }`}
            >
              {directionLabel(direction, template.primaryMetricLabel)}
            </button>
          ))}
        </div>
        <dl className="mt-5 space-y-3 text-[12.5px]">
          <Definition label="自变量" value={template.independentVariable} />
          <Definition label="主指标" value={template.primaryMetricLabel} />
        </dl>
        <button
          type="submit"
          disabled={pending !== null || statement.trim().length < 10 || rationale.trim().length < 10}
          className={primaryButtonClass}
        >
          {pending === 'hypothesis' ? '正在校验……' : '提交并接受假设'}
        </button>
      </aside>
    </form>
  );
}

function ComputeWorkspace({
  snapshot,
  template,
  pending,
  onExperiment,
  onRetry,
  onCancel,
}: {
  snapshot: RunSnapshot | null;
  template: ResearchTemplate | null;
  pending: ResearchCommand | null;
  onExperiment: () => void;
  onRetry: () => void;
  onCancel: () => void;
}) {
  if (!snapshot?.hypothesis) {
    return <Prerequisite zone="lab" text="先在模型与系统实验室的研究假设白板提交并接受研究假设。" />;
  }
  if (!template) return <Prerequisite zone="observatory" text="当前实验模板不可用，请重新选择毕业项目。" />;

  const { run, jobs } = snapshot;
  const variantOrder = template.variants.map((variant) => variant.id);
  const orderedJobs = [...jobs].sort(
    (left, right) => variantOrder.indexOf(left.variant) - variantOrder.indexOf(right.variant),
  );
  const secondaryMetric = template.secondaryMetrics[0];
  const canQueue = run.status === 'hypothesis_ready';
  const canCancel = run.status === 'queued' || run.status === 'running';
  const canRetry = run.status === 'failed' && run.currentStage === 'experiment';

  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1.25fr)_minmax(280px,0.75fr)]">
      <section className="min-w-0">
        <SectionEyebrow>FIXED CPU SUITE</SectionEyebrow>
        <h2 className="mt-2 font-serif text-[24px] font-semibold">{template.nameZh}离线基准</h2>
        <p className="mt-3 text-[13.5px] leading-relaxed text-faint">
          {template.variants.length} 个配置共享冻结数据集 {template.dataset}。运行器禁用网络，结论使用
          1,000 次配对 Bootstrap 估计 95% 置信区间。
        </p>

        {jobs.length > 0 && (
          <div className="mt-7 border-t border-hairline">
            {orderedJobs.map((job) => (
              <div
                key={job.id}
                className="flex flex-col gap-3 border-b border-hairline py-4 sm:flex-row sm:items-center sm:justify-between"
              >
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-medium">{variantLabel(job.variant, template)}</span>
                    <span className={`font-mono text-[9.5px] ${jobStatusClass(job.status)}`}>
                      {jobStatusLabel(job.status)}
                    </span>
                    <span className="font-mono text-[9.5px] text-faint">ATTEMPT {job.attempt}</span>
                  </div>
                  <p className="mt-1 truncate font-mono text-[9.5px] text-faint" title={job.id}>
                    {job.id}
                  </p>
                  {job.error && <p className="mt-2 text-[12px] text-ink-red">{job.error.message}</p>}
                </div>
                <div className="grid grid-cols-3 gap-4 sm:min-w-[280px]">
                  <CompactMetric
                    label={template.primaryMetricLabel}
                    value={formatScore(job.metrics?.aggregate.primaryScoreMean)}
                  />
                  <CompactMetric
                    label={secondaryMetric ? metricLabel(secondaryMetric) : 'Cases'}
                    value={
                      secondaryMetric
                        ? formatScore(job.metrics?.aggregate.secondaryMeans[secondaryMetric])
                        : formatCount(job.metrics?.aggregate.caseCount ?? null)
                    }
                  />
                  <CompactMetric
                    label="P95 延迟"
                    value={
                      job.metrics ? `${job.metrics.aggregate.caseLatencyMsP95.toFixed(2)} ms` : '—'
                    }
                  />
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <aside className="border-t border-hairline pt-6 lg:border-l lg:border-t-0 lg:pl-8 lg:pt-0">
        <SectionEyebrow>QUEUE CONTROL</SectionEyebrow>
        <dl className="mt-4 space-y-3 text-[12.5px]">
          <Definition label="运行状态" value={runStatusLabel(run.status)} />
          <Definition label="CPU / 内存" value="1 core / 1024 MiB" />
          <Definition label="单作业时限" value="120 秒" />
          <Definition label="网络" value="禁用" />
          <Definition label="API 费用" value="$0" />
        </dl>
        {canQueue && (
          <button type="button" onClick={onExperiment} disabled={pending !== null} className={primaryButtonClass}>
            {pending === 'experiment'
              ? '正在加入队列……'
              : `运行 ${template.variants.length} 组离线配置`}
          </button>
        )}
        {canRetry && (
          <button type="button" onClick={onRetry} disabled={pending !== null} className={primaryButtonClass}>
            {pending === 'retry' ? '正在重试……' : '重试失败作业'}
          </button>
        )}
        {canCancel && (
          <button type="button" onClick={onCancel} disabled={pending !== null} className={dangerButtonClass}>
            {pending === 'cancel' ? '正在取消……' : '取消当前运行'}
          </button>
        )}
        {run.status === 'analysis_ready' && (
          <Link to="/research-game/paper" className={`${secondaryButtonClass} block text-center`}>
            前往论文与评审工坊
          </Link>
        )}
        {run.status === 'cancelled' && (
          <p className="mt-5 border-l-2 border-hairline pl-3 text-[12.5px] leading-relaxed text-faint">
            该运行已取消。请回到观测站创建新的研究运行。
          </p>
        )}
      </aside>
    </div>
  );
}

function PaperWorkspace({
  snapshot,
  template,
  pending,
  onAnalyze,
  onVerify,
  onRetry,
  onCancel,
}: {
  snapshot: RunSnapshot | null;
  template: ResearchTemplate | null;
  pending: ResearchCommand | null;
  onAnalyze: () => void;
  onVerify: () => void;
  onRetry: () => void;
  onCancel: () => void;
}) {
  if (!snapshot || getResearchRunZoneStatus('compute', snapshot) !== 'complete') {
    return <Prerequisite zone="compute" text="先在算力工坊完成当前毕业项目的全部离线实验配置。" />;
  }
  if (!template) return <Prerequisite zone="observatory" text="当前实验模板不可用，请重新选择毕业项目。" />;

  const analysis = latestArtifact(snapshot, 'analysis');
  const verification = latestArtifact(snapshot, 'verification_bundle');
  const analysisContent = analysis?.content ?? null;
  const verificationContent = verification?.content ?? null;
  const incrementalDeltas = contentRecords(analysisContent, 'incrementalDeltas');
  const confidenceInterval = contentRecord(analysisContent, 'confidenceInterval95');
  const canRetry = snapshot.run.status === 'failed';
  const canCancel = snapshot.run.status === 'analyzing' || snapshot.run.status === 'verifying';

  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1.25fr)_minmax(280px,0.75fr)]">
      <section className="min-w-0">
        <SectionEyebrow>CLAIM & EVIDENCE</SectionEyebrow>
        <h2 className="mt-2 font-serif text-[24px] font-semibold">确定性结果分析</h2>
        {!analysisContent ? (
          <p className="mt-4 text-[13.5px] leading-relaxed text-faint">
            实验指标已就绪。分析器将按预先固定的效应阈值与 95% 配对 Bootstrap 置信区间生成 verdict。
          </p>
        ) : (
          <div className="mt-5">
            <div className="grid grid-cols-2 gap-5 border-y border-hairline py-5 sm:grid-cols-4">
              <CompactMetric label="结论" value={verdictLabel(contentString(analysisContent, 'verdict'))} />
              <CompactMetric
                label={`平均 ${template.primaryMetricLabel} 差值`}
                value={formatDelta(contentNumber(analysisContent, 'meanPrimaryMetricDelta'))}
              />
              <CompactMetric label="95% CI" value={formatConfidenceInterval(confidenceInterval)} />
              <CompactMetric
                label="最佳配置"
                value={variantLabel(contentString(analysisContent, 'bestVariant'), template)}
              />
            </div>
            {incrementalDeltas.length > 0 && (
              <div className="mt-6">
                <h3 className="font-serif text-[18px] font-semibold">逐阶段消融</h3>
                <div className="mt-3 border-t border-hairline">
                  {incrementalDeltas.map((delta, index) => {
                    const value = contentNumber(delta, 'primaryMetricDelta');
                    return (
                      <div
                        key={`${contentString(delta, 'from')}-${contentString(delta, 'to')}-${index}`}
                        className="flex items-center justify-between gap-4 border-b border-hairline py-3 text-[12.5px]"
                      >
                        <span>
                          {variantLabel(contentString(delta, 'from'), template)} →{' '}
                          {variantLabel(contentString(delta, 'to'), template)}
                        </span>
                        <span className={value !== null && value < 0 ? 'font-mono text-ink-red' : 'font-mono text-ink-green'}>
                          {formatDelta(value)} {template.primaryMetricLabel}
                        </span>
                      </div>
                    );
                  })}
                </div>
                <p className="mt-3 text-[12.5px] leading-relaxed text-faint">
                  负值会作为负向消融保留，不会从研究结论中隐藏。
                </p>
              </div>
            )}
            <h3 className="mt-6 font-serif text-[18px] font-semibold">适用边界</h3>
            <ul className="mt-3 space-y-2 text-[12.5px] leading-relaxed text-faint">
              {contentStrings(analysisContent, 'limitations').map((limitation) => (
                <li key={limitation} className="border-l border-hairline pl-3">
                  {limitationLabel(limitation)}
                </li>
              ))}
            </ul>
            {contentStrings(analysisContent, 'externalValidation').length > 0 && (
              <>
                <h3 className="mt-6 font-serif text-[18px] font-semibold">尚需外部验证</h3>
                <ul className="mt-3 space-y-2 text-[12.5px] leading-relaxed text-faint">
                  {contentStrings(analysisContent, 'externalValidation').map((item) => (
                    <li key={item} className="border-l border-hairline pl-3">
                      {item}
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        )}

        {verificationContent && (
          <div className="mt-8 border-t border-hairline pt-6">
            <SectionEyebrow>REPRODUCTION BUNDLE</SectionEyebrow>
            <h3 className="mt-2 font-serif text-[19px] font-semibold">可验证产物</h3>
            <dl className="mt-4 grid gap-x-6 gap-y-3 text-[12.5px] sm:grid-cols-2">
              <Definition label="完整性" value={nestedBoolean(verificationContent, 'integrity', 'passed') ? '通过' : '失败'} />
              <Definition
                label="结果复现"
                value={nestedBoolean(verificationContent, 'reproduction', 'passed') ? '通过' : '失败'}
              />
              <Definition label="验证器" value={contentString(verificationContent, 'verifierVersion') ?? '—'} />
              <Definition label="验证时间" value={formatDate(contentString(verificationContent, 'verifiedAt'))} />
            </dl>
          </div>
        )}
      </section>

      <aside className="border-t border-hairline pt-6 lg:border-l lg:border-t-0 lg:pl-8 lg:pt-0">
        <SectionEyebrow>REVIEW CONTROL</SectionEyebrow>
        <dl className="mt-4 space-y-3 text-[12.5px]">
          <Definition label="运行状态" value={runStatusLabel(snapshot.run.status)} />
          <Definition label="规则" value={claimRuleLabel(template)} />
          <Definition
            label="同向重采样"
            value={`${contentNumber(analysisContent, 'directionAgreementCount') ?? '—'} / 1,000`}
          />
          <Definition label="统计声明" value="配对 Bootstrap 区间，不报告 p-value" />
          <Definition
            label="复现验证"
            value={verification?.verificationStatus === 'passed' ? '通过' : '待验证'}
          />
        </dl>
        {snapshot.run.status === 'analysis_ready' && (
          <button type="button" onClick={onAnalyze} disabled={pending !== null} className={primaryButtonClass}>
            {pending === 'analysis' ? '正在分析……' : '生成结果分析'}
          </button>
        )}
        {snapshot.run.status === 'verification_required' && (
          <button type="button" onClick={onVerify} disabled={pending !== null} className={primaryButtonClass}>
            {pending === 'verification' ? '正在验证……' : '运行独立复现验证'}
          </button>
        )}
        {canRetry && (
          <button type="button" onClick={onRetry} disabled={pending !== null} className={primaryButtonClass}>
            {pending === 'retry' ? '正在恢复……' : '恢复失败步骤'}
          </button>
        )}
        {canCancel && (
          <button type="button" onClick={onCancel} disabled={pending !== null} className={dangerButtonClass}>
            {pending === 'cancel' ? '正在取消……' : '取消当前运行'}
          </button>
        )}
        {snapshot.run.status === 'completed' && (
          <Link to="/research-game/plaza" className={`${secondaryButtonClass} block text-center`}>
            前往学术交流广场
          </Link>
        )}
      </aside>
    </div>
  );
}

function PlazaWorkspace({
  snapshot,
  template,
  pending,
  onPublication,
}: {
  snapshot: RunSnapshot | null;
  template: ResearchTemplate | null;
  pending: ResearchCommand | null;
  onPublication: (artifactId: string, publish: boolean) => void;
}) {
  const card = snapshot ? latestArtifact(snapshot, 'research_card') : null;
  if (!snapshot || !card?.content) {
    return <Prerequisite zone="paper" text="先完成结果分析与独立复现验证，系统才会生成研究卡。" />;
  }

  const isPublic = card.visibility === 'public';
  const confidenceInterval = contentRecord(card.content, 'confidenceInterval95');
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1.25fr)_minmax(280px,0.75fr)]">
      <section className="min-w-0">
        <SectionEyebrow>VERIFIED RESEARCH CARD</SectionEyebrow>
        <h2 className="mt-2 font-serif text-[26px] font-semibold">
          {contentString(card.content, 'title') ?? snapshot.run.topic}
        </h2>
        <p className="mt-4 border-l-2 border-ink-blue pl-3 text-[13.5px] leading-relaxed">
          {contentString(card.content, 'hypothesis') ?? snapshot.hypothesis?.statement}
        </p>
        <div className="mt-6 grid grid-cols-2 gap-5 border-y border-hairline py-5 sm:grid-cols-4">
          <CompactMetric label="结论" value={verdictLabel(contentString(card.content, 'verdict'))} />
          <CompactMetric
            label="最佳配置"
            value={variantLabel(contentString(card.content, 'bestVariant'), template)}
          />
          <CompactMetric
            label={`Baseline ${template?.primaryMetricLabel ?? '主指标'}`}
            value={formatScore(contentNumber(card.content, 'baselinePrimaryScore'))}
          />
          <CompactMetric
            label={`候选 ${template?.primaryMetricLabel ?? '主指标'}`}
            value={formatScore(contentNumber(card.content, 'candidatePrimaryScore'))}
          />
          <CompactMetric
            label="平均主指标差值"
            value={formatDelta(contentNumber(card.content, 'meanPrimaryMetricDelta'))}
          />
          <CompactMetric label="95% CI" value={formatConfidenceInterval(confidenceInterval)} />
          <CompactMetric
            label="同向重采样"
            value={`${formatCount(contentNumber(card.content, 'directionAgreementCount'))} / ${formatCount(contentNumber(card.content, 'bootstrapCount') ?? 1000)}`}
          />
          <CompactMetric label="验证状态" value="复现通过" />
        </div>
        <h3 className="mt-6 font-serif text-[18px] font-semibold">公开限制</h3>
        <ul className="mt-3 space-y-2 text-[12.5px] leading-relaxed text-faint">
          {contentStrings(card.content, 'limitations').map((limitation) => (
            <li key={limitation} className="border-l border-hairline pl-3">
              {limitationLabel(limitation)}
            </li>
          ))}
        </ul>
      </section>
      <aside className="border-t border-hairline pt-6 lg:border-l lg:border-t-0 lg:pl-8 lg:pt-0">
        <SectionEyebrow>PUBLICATION</SectionEyebrow>
        <dl className="mt-4 space-y-3 text-[12.5px]">
          <Definition label="可见性" value={isPublic ? '公开' : '仅自己可见'} />
          <Definition label="产物校验" value={card.verificationStatus === 'passed' ? '通过' : '未通过'} />
          <Definition label="模板版本" value={contentString(card.content, 'templateVersion') ?? '—'} />
          <Definition label="验证时间" value={formatDate(contentString(card.content, 'verifiedAt'))} />
        </dl>
        <button
          type="button"
          onClick={() => onPublication(card.id, !isPublic)}
          disabled={pending !== null}
          className={isPublic ? dangerButtonClass : primaryButtonClass}
        >
          {pending === 'publish'
            ? '正在公开……'
            : pending === 'unpublish'
              ? '正在撤回……'
              : isPublic
                ? '撤回公开研究卡'
                : '公开研究卡'}
        </button>
        <p className="mt-4 text-[12px] leading-relaxed text-faint">
          只公开经过验证、已裁剪为安全字段的研究卡；原始日志和实验文件保持私有。
        </p>
      </aside>
    </div>
  );
}

function Prerequisite({ zone, text }: { zone: ZoneId; text: string }) {
  const target = getZone(zone) as Zone;
  return (
    <div className="min-h-48 border-l-2 border-ink-yellow pl-4">
      <SectionEyebrow>PREREQUISITE</SectionEyebrow>
      <h2 className="mt-2 font-serif text-[23px] font-semibold">当前区域尚未就绪</h2>
      <p className="mt-3 text-[13.5px] leading-relaxed text-faint">{text}</p>
      <Link to={`/research-game/${zone}`} className={`${secondaryButtonClass} mt-5 inline-flex items-center`}>
        前往{target.name}
      </Link>
    </div>
  );
}

function zoneMission(
  zoneId: ZoneId,
  snapshot: RunSnapshot | null,
  template: ResearchTemplate | null,
): string {
  if (zoneId === 'observatory') {
    return snapshot
      ? `当前课题：${snapshot.run.topic}`
      : '从 17 个毕业项目中选择一个离线、可复现的研究子任务。';
  }
  if (zoneId === 'lab') {
    return snapshot?.hypothesis
      ? '研究假设已接受，实验变量与主指标已冻结。'
      : `围绕 ${template?.primaryMetricLabel ?? '主指标'} 写出可反驳的假设和机制解释。`;
  }
  if (zoneId === 'compute') return snapshot ? runStatusLabel(snapshot.run.status) : '等待课题与假设。';
  if (zoneId === 'paper') return snapshot ? runStatusLabel(snapshot.run.status) : '等待实验结果。';
  const card = snapshot ? latestArtifact(snapshot, 'research_card') : null;
  if (!card) return '等待经过复现验证的研究卡。';
  return card.visibility === 'public' ? '研究卡已公开，可随时撤回。' : '检查研究卡后决定是否公开。';
}

function connectedZoneSubtitle(zoneId: ZoneId, template: ResearchTemplate | null): string {
  if (!template) return '选择毕业项目后，系统会绑定对应的离线实验契约。';
  if (zoneId === 'observatory') {
    return `审计 ${template.dataset} 的来源、评测边界与零 API 成本约束`;
  }
  if (zoneId === 'lab') {
    return `冻结自变量 ${template.independentVariable} 与主指标 ${template.primaryMetricLabel}`;
  }
  if (zoneId === 'compute') {
    return `运行 ${template.variants.length} 组固定配置并记录逐 case 结果`;
  }
  if (zoneId === 'paper') {
    return `分析 ${template.primaryMetricLabel}、配对 Bootstrap 区间和外部验证边界`;
  }
  return `公开经过独立复现的 ${template.nameZh} 研究卡`;
}

function embeddedWorkspaceName(zoneId: ZoneId): string {
  if (zoneId === 'observatory') return '项目与数据终端';
  if (zoneId === 'lab') return '研究假设白板';
  if (zoneId === 'compute') return '离线实验集群';
  if (zoneId === 'paper') return '分析与复现稿件';
  return '验证研究海报板';
}

function zoneMetrics(
  zoneId: ZoneId,
  snapshot: RunSnapshot | null,
  template: ResearchTemplate | null,
): Array<{ label: string; value: string }> {
  if (!snapshot) {
    return [
      { label: '运行', value: '未创建' },
      { label: '项目', value: '17 个' },
      { label: '状态', value: '等待' },
    ];
  }
  if (zoneId === 'observatory') {
    return [
      { label: '模板', value: snapshot.run.templateVersion },
      { label: '版本', value: `v${snapshot.run.version}` },
      { label: '状态', value: runStatusLabel(snapshot.run.status) },
    ];
  }
  if (zoneId === 'lab') {
    return [
      { label: '假设', value: snapshot.hypothesis ? '已接受' : '待提交' },
      { label: '变量', value: template?.independentVariable ?? '—' },
      { label: '指标', value: template?.primaryMetricLabel ?? '—' },
    ];
  }
  if (zoneId === 'compute') {
    const succeeded = snapshot.jobs.filter((job) => job.status === 'succeeded').length;
    const caseCount = snapshot.jobs.find((job) => job.metrics)?.metrics?.aggregate.caseCount;
    return [
      { label: '作业', value: `${succeeded}/${snapshot.jobs.length || template?.variants.length || 0}` },
      { label: '样本', value: caseCount ? `${caseCount}` : '等待' },
      { label: '重试', value: `${snapshot.run.retryCount}` },
    ];
  }
  if (zoneId === 'paper') {
    const analysis = latestArtifact(snapshot, 'analysis');
    const verification = latestArtifact(snapshot, 'verification_bundle');
    return [
      { label: '分析', value: analysis ? '已生成' : '待生成' },
      { label: '复现', value: verification?.verificationStatus === 'passed' ? '通过' : '待验证' },
      { label: '状态', value: runStatusLabel(snapshot.run.status) },
    ];
  }
  const card = latestArtifact(snapshot, 'research_card');
  return [
    { label: '研究卡', value: card ? '已生成' : '等待' },
    { label: '校验', value: card?.verificationStatus === 'passed' ? '通过' : '—' },
    { label: '可见性', value: card?.visibility === 'public' ? '公开' : '私有' },
  ];
}

function latestArtifact(snapshot: RunSnapshot, kind: ResearchArtifact['kind']): ResearchArtifact | null {
  return [...snapshot.artifacts]
    .filter((artifact) => artifact.kind === kind)
    .sort((left, right) => right.version - left.version)[0] ?? null;
}

function contentString(content: Record<string, unknown> | null, key: string): string | null {
  const value = content?.[key];
  return typeof value === 'string' ? value : null;
}

function contentNumber(content: Record<string, unknown> | null, key: string): number | null {
  const value = content?.[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function contentStrings(content: Record<string, unknown> | null, key: string): string[] {
  const value = content?.[key];
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
}

function contentRecords(content: Record<string, unknown> | null, key: string): Array<Record<string, unknown>> {
  const value = content?.[key];
  return Array.isArray(value)
    ? value.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object')
    : [];
}

function contentRecord(content: Record<string, unknown> | null, key: string): Record<string, unknown> | null {
  const value = content?.[key];
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function nestedBoolean(content: Record<string, unknown>, parent: string, child: string): boolean {
  const nested = content[parent];
  return Boolean(nested && typeof nested === 'object' && (nested as Record<string, unknown>)[child] === true);
}

function directionLabel(direction: ExpectedDirection, metric: string): string {
  if (direction === 'increase') return `候选方案提高 ${metric}`;
  if (direction === 'decrease') return `候选方案降低 ${metric}`;
  return `候选方案不改变 ${metric}`;
}

function variantLabel(variant: string | null, template: ResearchTemplate | null): string {
  if (!variant) return '—';
  return template?.variants.find((item) => item.id === variant)?.label ?? variant;
}

function defaultHypothesisStatement(template: ResearchTemplate | null): string {
  if (!template) return '候选离线流水线将相较基线改善预先指定的主指标。';
  const baseline = template.variants[0]?.label ?? '基线';
  const candidate = template.variants.at(-1)?.label ?? '候选方案';
  return `相较于 ${baseline}，${candidate} 将提高 ${template.primaryMetricLabel}。`;
}

function defaultHypothesisRationale(template: ResearchTemplate | null): string {
  if (!template) return '候选方案加入了与任务误差来源对应的机制，需要通过冻结数据和配对重采样检验。';
  return `${template.variants.map((variant) => variant.label).join('、')} 代表逐步增加的系统机制；冻结数据与配对重采样用于区分真实改进和样本波动。`;
}

function benchmarkKindLabel(kind: ResearchTemplate['benchmarkKind'] | undefined): string {
  if (kind === 'public_snapshot') return '公开数据冻结快照';
  if (kind === 'packaged_dataset') return '本地打包数据集';
  if (kind === 'deterministic_fixture') return '确定性测试夹具';
  return '—';
}

function metricLabel(metric: string): string {
  return metric.replaceAll('_', ' ');
}

function claimRuleLabel(template: ResearchTemplate): string {
  const threshold = template.effectThreshold.toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
  if (template.expectedDirection === 'decrease') {
    return `Δ${template.primaryMetricLabel} ≤ -${threshold} 且 95% CI 上界 < 0`;
  }
  if (template.expectedDirection === 'no_change') {
    return `|Δ${template.primaryMetricLabel}| < ${threshold} 且 95% CI 包含 0`;
  }
  return `Δ${template.primaryMetricLabel} ≥ ${threshold} 且 95% CI 下界 > 0`;
}

function limitationLabel(limitation: string): string {
  const labels: Record<string, string> = {
    'The primary metric follows CodeSearchNet judged-candidate evaluation; unjudged snippets are ignored.':
      '主指标沿用 CodeSearchNet 的已标注候选评测口径；未标注代码片段不参与该指标。',
    'The frozen slice evaluates 85 Python queries and does not establish cross-language performance.':
      '冻结切片只评测 85 条 Python 查询，不能据此推断跨语言表现。',
    'The local dense retriever uses character n-gram LSA, not a production code-embedding model.':
      '本地稠密检索使用字符 n-gram LSA，不等同于生产级代码向量模型。',
    'Twenty-five source files were unavailable and eleven snippets failed Python 3 AST parsing; all exclusions are recorded.':
      '25 个源码文件不可用，另有 11 个片段未通过 Python 3 AST 解析；所有排除项均已记录。',
    'Latency is diagnostic and machine-dependent; it is excluded from reproduction equality.':
      '延迟依赖运行机器，仅用于诊断，不参与复现一致性判定。',
    'The experiment evaluates retrieval and citations, not answer generation quality.':
      '实验评估检索与引用质量，不评估最终答案的生成质量。',
  };
  return labels[limitation] ?? limitation;
}

function verdictLabel(verdict: string | null): string {
  if (verdict === 'supported') return '支持';
  if (verdict === 'partially_supported') return '部分支持';
  if (verdict === 'refuted') return '不支持';
  return '—';
}

function jobStatusLabel(status: RunSnapshot['jobs'][number]['status']): string {
  const labels = {
    queued: '排队中',
    running: '运行中',
    succeeded: '成功',
    failed: '失败',
    cancelled: '已取消',
    timed_out: '超时',
  };
  return labels[status];
}

function jobStatusClass(status: RunSnapshot['jobs'][number]['status']): string {
  if (status === 'succeeded') return 'text-ink-green';
  if (status === 'failed' || status === 'timed_out') return 'text-ink-red';
  if (status === 'running') return 'text-ink-blue';
  return 'text-faint';
}

function formatScore(value: number | null | undefined): string {
  return value === undefined || value === null ? '—' : value.toFixed(4);
}

function formatCount(value: number | null): string {
  return value === null ? '—' : new Intl.NumberFormat('en-US').format(value);
}

function formatDelta(value: number | null): string {
  if (value === null) return '—';
  return `${value >= 0 ? '+' : ''}${value.toFixed(4)}`;
}

function formatConfidenceInterval(interval: Record<string, unknown> | null): string {
  const lower = contentNumber(interval, 'lower');
  const upper = contentNumber(interval, 'upper');
  if (lower === null || upper === null) return '—';
  return `[${lower.toFixed(4)}, ${upper.toFixed(4)}]`;
}

function formatDate(value: string | null): string {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '—';
  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

function StatusPill({ status }: { status: ZoneStatus }) {
  const cls =
    status === 'complete'
      ? 'border-ink-green/30 bg-pale-green text-ink-green'
      : status === 'active'
        ? 'border-ink-blue/30 bg-pale-blue text-ink-blue'
        : status === 'ready'
          ? 'border-ink-yellow/30 bg-pale-yellow text-ink-yellow'
          : 'border-hairline bg-bone text-faint';
  return (
    <span className={`w-fit rounded-full border px-2.5 py-1 font-mono text-[10px] tracking-[0.12em] ${cls}`}>
      {statusLabel(status)}
    </span>
  );
}

function SectionEyebrow({ children }: { children: string }) {
  return <div className="font-mono text-[9.5px] tracking-[0.16em] text-faint">{children}</div>;
}

function Definition({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-hairline pb-2">
      <dt className="shrink-0 text-faint">{label}</dt>
      <dd className="min-w-0 break-words text-right font-medium">{value}</dd>
    </div>
  );
}

function MiniMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 border-l border-hairline pl-2.5">
      <div className="font-mono text-[9.5px] text-faint">{label}</div>
      <div className="mt-1 break-words text-[12.5px] font-medium">{value}</div>
    </div>
  );
}

function CompactMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <div className="font-mono text-[9px] text-faint">{label}</div>
      <div className="mt-1 break-words text-[12.5px] font-medium">{value}</div>
    </div>
  );
}

const primaryButtonClass =
  'mt-6 min-h-11 w-full rounded-md bg-ink px-4 py-2 text-[13.5px] font-medium text-paper transition-colors hover:bg-[#303942] disabled:cursor-not-allowed disabled:bg-hairline disabled:text-faint';
const secondaryButtonClass =
  'mt-6 min-h-10 w-full rounded-md border border-hairline bg-bone px-4 py-2 text-[13px] font-medium text-ink transition-colors hover:bg-paper';
const dangerButtonClass =
  'mt-3 min-h-10 w-full rounded-md border border-ink-red/30 bg-pale-red px-4 py-2 text-[13px] font-medium text-ink-red transition-colors hover:bg-paper disabled:cursor-not-allowed disabled:opacity-50';
const textButtonClass =
  'mt-4 w-full text-center text-[12px] text-faint underline decoration-hairline underline-offset-4 hover:text-ink';
