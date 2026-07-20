import { researchApiConfigured } from '../lib/researchApi';
import {
  experimentOptions,
  getZone,
  getZoneStatus,
  statusLabel,
  useResearchGame,
  type ExperimentChoice,
  type ZoneId,
} from '../lib/researchGame';
import { EmbeddedConnectedResearchWorkspace } from '../pages/ConnectedResearchZonePage';

export function ResearchWorkspaceOverlay({
  zoneId,
  onClose,
}: {
  zoneId: ZoneId;
  onClose: () => void;
}) {
  const zone = getZone(zoneId);
  if (!zone) return null;

  return (
    <div
      className="absolute inset-0 z-40 flex items-end justify-center bg-[#0b1418]/55 backdrop-blur-[3px] sm:items-center sm:p-5"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        className="case-desk-immersive max-h-[94dvh] w-full max-w-6xl overflow-y-auto rounded-t-lg border border-white/10 shadow-[0_20px_60px_rgba(0,0,0,0.45)] sm:rounded-lg"
        aria-label={workspaceName(zoneId)}
      >
        <header className="case-desk-header sticky top-0 z-10 flex items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <div className="min-w-0">
            <p className="font-mono text-[10px] tracking-[0.12em] text-white/55">现场工作台</p>
            <h2 className="mt-1 truncate font-serif text-[20px] font-semibold text-white">{workspaceName(zoneId)}</h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md border border-white/15 bg-white/5 text-[22px] text-white/75 hover:bg-white/10 hover:text-white"
            aria-label="关闭科研工作台"
          >
            ×
          </button>
        </header>
        <div className="px-4 py-5 text-[#e8eef0] sm:px-6">
          {researchApiConfigured ? (
            <EmbeddedConnectedResearchWorkspace zoneId={zoneId} />
          ) : (
            <LocalResearchWorkspace zoneId={zoneId} />
          )}
        </div>
      </section>
    </div>
  );
}

function LocalResearchWorkspace({ zoneId }: { zoneId: ZoneId }) {
  const zone = getZone(zoneId)!;
  const {
    game,
    artifacts,
    startZone,
    completeZone,
    setZoneNote,
    selectExperiment,
    runExperiment,
  } = useResearchGame();
  const status = getZoneStatus(zoneId, game);
  const note = game.zoneNotes[zoneId] ?? '';
  const blockedCompute = zoneId === 'compute' && status === 'active' && !game.experimentRan;
  const artifact = artifacts.find((item) => item.zoneId === zoneId);

  return (
    <div>
      <div className="border-l-2 border-ink-yellow bg-pale-yellow px-4 py-3 text-[12px] leading-relaxed text-ink-yellow">
        本地演示模式不会启动真实实验。配置 ResearchRun API 后，这里才会显示服务器记录、真实离线运行与可验证产物。
      </div>

      <div className="mt-5 grid gap-7 lg:grid-cols-[minmax(0,1.1fr)_minmax(280px,0.9fr)]">
        <section>
          <p className="font-mono text-[9.5px] tracking-[0.16em] text-faint">{statusLabel(status)}</p>
          <h3 className="mt-2 font-serif text-[24px] font-semibold">{workspaceName(zoneId)}</h3>
          <p className="mt-3 text-[13.5px] leading-relaxed">{zone.task}</p>
          <p className="mt-3 text-[12.5px] leading-relaxed text-faint">{zone.tutorLine}</p>

          {status !== 'locked' && (
            <label className="mt-6 block text-[12.5px] font-medium">
              {zone.noteLabel}
              <textarea
                value={note}
                onChange={(event) => setZoneNote(zoneId, event.target.value)}
                placeholder={zone.notePlaceholder}
                className="mt-2 min-h-28 w-full resize-y rounded-md border border-hairline bg-paper px-3 py-3 text-[13px] leading-relaxed outline-none focus:border-ink-blue"
              />
            </label>
          )}

          {zoneId === 'compute' && status === 'active' && (
            <div className="mt-5">
              <div className="grid gap-2 sm:grid-cols-3" role="group" aria-label="实验配置">
                {experimentOptions.map((option) => (
                  <button
                    key={option.id}
                    type="button"
                    onClick={() => selectExperiment(option.id as ExperimentChoice)}
                    className={`min-h-16 rounded-md border px-3 py-2 text-left text-[12px] transition-colors ${
                      game.experimentChoice === option.id
                        ? 'border-ink bg-ink text-paper'
                        : 'border-hairline bg-bone hover:bg-paper'
                    }`}
                  >
                    <span className="block font-medium">{option.name}</span>
                    <span className="mt-1 block font-mono text-[9.5px] opacity-70">{option.score}</span>
                  </button>
                ))}
              </div>
              {!game.experimentRan && (
                <button type="button" onClick={runExperiment} className={primaryButtonClass}>
                  确认运行静态演示
                </button>
              )}
            </div>
          )}
        </section>

        <aside className="border-t border-hairline pt-6 lg:border-l lg:border-t-0 lg:pl-7 lg:pt-0">
          <p className="font-mono text-[9.5px] tracking-[0.16em] text-faint">HUMAN-GATED ACTION</p>
          <dl className="mt-4 space-y-3 text-[12.5px]">
            {zone.stats.map((item) => (
              <div key={item.label} className="flex justify-between gap-4 border-b border-hairline pb-2">
                <dt className="text-faint">{item.label}</dt>
                <dd className="font-medium">{item.value}</dd>
              </div>
            ))}
          </dl>

          {status === 'locked' && (
            <p className="mt-5 border-l-2 border-ink-yellow pl-3 text-[12.5px] leading-relaxed text-ink-yellow">
              {mergedLockedHint(zoneId)}
            </p>
          )}
          {status === 'ready' && (
            <button type="button" onClick={() => startZone(zoneId)} className={primaryButtonClass}>
              {zone.startAction}
            </button>
          )}
          {status === 'active' && (
            <button
              type="button"
              onClick={() => completeZone(zoneId)}
              disabled={blockedCompute}
              className={primaryButtonClass}
            >
              {zone.completeAction}
            </button>
          )}
          {status === 'complete' && artifact && (
            <div className="mt-5 border-l-2 border-ink-green pl-3">
              <p className="text-[13px] font-medium">{artifact.name}</p>
              <p className="mt-1 text-[11.5px] leading-relaxed text-faint">{artifact.detail}</p>
            </div>
          )}
          <p className="mt-4 text-[11.5px] leading-relaxed text-faint">
            Agent 可以给出建议，但开始、接受、运行和公开动作必须由玩家点击确认。
          </p>
        </aside>
      </div>
    </div>
  );
}

const primaryButtonClass =
  'mt-5 min-h-11 w-full rounded-md bg-ink px-4 py-2 text-[13px] font-medium text-paper transition-colors hover:bg-[#303942] disabled:cursor-not-allowed disabled:bg-hairline disabled:text-faint';

function workspaceName(zoneId: ZoneId): string {
  if (zoneId === 'observatory') return '项目与数据终端';
  if (zoneId === 'lab') return '研究假设白板';
  if (zoneId === 'compute') return '离线实验集群';
  if (zoneId === 'paper') return '分析与复现稿件';
  return '验证研究海报板';
}

function mergedLockedHint(zoneId: ZoneId): string {
  if (zoneId === 'lab') return '先在同一实验室的项目与数据终端完成基准审计。';
  if (zoneId === 'compute') return '先在模型与系统实验室冻结研究假设。';
  if (zoneId === 'paper') return '先在算力工坊得到实验结果。';
  if (zoneId === 'plaza') return '先在论文与评审工坊完成分析和复现。';
  return '从这里开始。';
}
