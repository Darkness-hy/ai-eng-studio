import { Link, Navigate, useParams } from 'react-router-dom';
import { researchApiConfigured } from '../lib/researchApi';
import {
  experimentOptions,
  getZone,
  getZoneStatus,
  nextZone,
  researcherAsset,
  statusLabel,
  tutorOutfits,
  useResearchGame,
  zones,
  type ExperimentChoice,
  type TutorId,
  type Zone,
  type ZoneStatus,
} from '../lib/researchGame';
import { ConnectedResearchZonePage } from './ConnectedResearchZonePage';

export function ResearchZonePage() {
  return researchApiConfigured ? <ConnectedResearchZonePage /> : <LocalResearchZonePage />;
}

function LocalResearchZonePage() {
  const { zoneId } = useParams<{ zoneId: string }>();
  const zone = getZone(zoneId);
  const {
    game,
    artifacts,
    startZone,
    completeZone,
    setTutor,
    setZoneNote,
    selectExperiment,
    runExperiment,
  } = useResearchGame();

  if (!zone) return <Navigate to="/research-game" replace />;

  const status = getZoneStatus(zone.id, game);
  const activeTutor = tutorOutfits.find((tutor) => tutor.id === game.tutorId) ?? tutorOutfits[0];
  const recommendedTutor = tutorOutfits.find((tutor) => tutor.id === zone.tutor) ?? tutorOutfits[0];
  const followingZone = nextZone(zone.id);
  const note = game.zoneNotes[zone.id] ?? '';
  const isComputeBlocked = zone.id === 'compute' && status === 'active' && !game.experimentRan;

  return (
    <div className="research-game-ui bg-canvas">
      <div className="mx-auto max-w-6xl px-4 py-6 sm:px-5 lg:py-8">
        <header className="border-b border-hairline pb-5">
          <Link to="/research-game" className="text-[13px] text-faint transition-colors hover:text-ink">
            ← 返回科研岛
          </Link>
          <div className="mt-5 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <p className="font-mono text-[10px] tracking-[0.18em] text-faint">LOCAL DEMO · STATIC SCORES</p>
              <h1 className="mt-2 font-serif text-[29px] font-semibold sm:text-[38px]">{zone.name}</h1>
              <p className="mt-2 text-[13.5px] leading-relaxed text-faint">{zone.subtitle}</p>
            </div>
            <StatusPill status={status} />
          </div>
        </header>

        <div className="border-b border-hairline bg-pale-yellow px-4 py-3 text-[12.5px] leading-relaxed text-ink-yellow">
          本地演示模式：以下进度和分数仅保存在浏览器，不代表真实实验运行。
        </div>

        <nav className="flex gap-1 overflow-x-auto border-b border-hairline py-3" aria-label="科研岛区域">
          {zones.map((item, index) => {
            const itemStatus = getZoneStatus(item.id, game);
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

        <section className="grid border-b border-hairline lg:grid-cols-[minmax(0,1.1fr)_minmax(320px,0.9fr)]">
          <div className="flex min-h-[310px] items-center justify-center bg-bone px-5 py-8 sm:min-h-[390px]">
            <img src={zone.asset} alt="" className="max-h-[380px] max-w-full object-contain" draggable={false} />
          </div>
          <div className="flex flex-col justify-center border-t border-hairline px-5 py-8 lg:border-l lg:border-t-0 lg:px-8">
            <div className={`w-fit rounded-full border px-2.5 py-1 font-mono text-[9.5px] tracking-[0.14em] ${zone.accent}`}>
              {statusLabel(status)}
            </div>
            <h2 className="mt-5 font-serif text-[23px] font-semibold">当前任务</h2>
            <p className="mt-2 text-[14px] leading-relaxed">{zone.task}</p>
            {status === 'locked' && (
              <p className="mt-4 border-l-2 border-ink-yellow pl-3 text-[13px] leading-relaxed text-ink-yellow">
                {zone.lockedHint}
              </p>
            )}
            {isComputeBlocked && (
              <p className="mt-4 border-l-2 border-ink-yellow pl-3 text-[13px] leading-relaxed text-ink-yellow">
                先运行一组实验，才能封存实验结果。
              </p>
            )}
            <div className="mt-5 grid grid-cols-3 gap-2">
              {zone.stats.map((stat) => (
                <MiniMetric key={stat.label} label={stat.label} value={stat.value} />
              ))}
            </div>
            <button
              type="button"
              onClick={status === 'ready' ? () => startZone(zone.id) : () => completeZone(zone.id)}
              disabled={status === 'locked' || status === 'complete' || isComputeBlocked}
              className="mt-5 min-h-11 rounded-md bg-ink px-4 py-2 text-[14px] font-medium text-paper transition-colors hover:bg-[#303942] disabled:cursor-not-allowed disabled:bg-hairline disabled:text-faint"
            >
              {missionAction(zone, status)}
            </button>
          </div>
        </section>

        <section className="grid gap-8 border-b border-hairline py-8 lg:grid-cols-[minmax(0,1.25fr)_minmax(300px,0.75fr)]">
          <div className="min-w-0">
            <div className="flex items-baseline justify-between gap-4">
              <h2 className="font-serif text-[23px] font-semibold">{zone.noteLabel}</h2>
              <span className="font-mono text-[9.5px] tracking-[0.14em] text-faint">AUTO SAVED</span>
            </div>
            <textarea
              value={note}
              onChange={(event) => setZoneNote(zone.id, event.target.value)}
              placeholder={zone.notePlaceholder}
              className="mt-4 min-h-40 w-full resize-y rounded-md border border-hairline bg-paper px-4 py-3 text-[14px] leading-relaxed text-ink outline-none transition-colors placeholder:text-faint focus:border-ink-blue"
            />

            {zone.id === 'compute' && (
              <ExperimentPanel
                selected={game.experimentChoice}
                ran={game.experimentRan}
                onSelect={selectExperiment}
                onRun={runExperiment}
              />
            )}

            <div className="mt-8 border-t border-hairline pt-6">
              <h3 className="font-serif text-[19px] font-semibold">区域产物</h3>
              <p className="mt-2 text-[13.5px] leading-relaxed text-faint">
                {game.completed.includes(zone.id)
                  ? `已生成：${zone.output}。该产物已进入研究档案。`
                  : `完成当前任务后生成${zone.output}。`}
              </p>
            </div>
          </div>

          <TutorPanel
            activeTutorId={activeTutor.id}
            recommendedTutorId={recommendedTutor.id}
            activeLine={zone.tutorLine}
            onTutorChange={setTutor}
          />
        </section>

        <footer className="flex flex-col gap-4 py-6 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-4">
            <div className="h-14 w-11 overflow-hidden">
              <img src={researcherAsset} alt="" className="h-full w-full object-contain object-top" />
            </div>
            <div>
              <div className="font-mono text-[9.5px] tracking-[0.14em] text-faint">RESEARCH RECORD</div>
              <div className="mt-1 text-[13px]">已获得 {artifacts.length} / 5 项研究产物</div>
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

function missionAction(zone: Zone, status: ZoneStatus): string {
  if (status === 'locked') return zone.lockedHint;
  if (status === 'ready') return zone.startAction;
  if (status === 'active') return zone.completeAction;
  return `已获得${zone.output}`;
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

function ExperimentPanel({
  selected,
  ran,
  onSelect,
  onRun,
}: {
  selected: ExperimentChoice;
  ran: boolean;
  onSelect: (choice: ExperimentChoice) => void;
  onRun: () => void;
}) {
  const current = experimentOptions.find((option) => option.id === selected) ?? experimentOptions[0];
  return (
    <div className="mt-8 border-t border-hairline pt-6">
      <div className="flex items-baseline justify-between gap-3">
        <h3 className="font-serif text-[19px] font-semibold">代码检索演示台</h3>
        <span className="font-mono text-[9.5px] tracking-[0.14em] text-faint">
          {ran ? 'RESULT READY' : 'QUEUE EMPTY'}
        </span>
      </div>
      <div className="mt-4 grid grid-cols-3 gap-2">
        {experimentOptions.map((option) => (
          <button
            key={option.id}
            type="button"
            onClick={() => onSelect(option.id)}
            className={`min-h-10 rounded-md border px-2 py-1 text-[11.5px] leading-tight transition-colors ${
              selected === option.id
                ? 'border-ink bg-ink text-paper'
                : 'border-hairline bg-bone text-faint hover:bg-paper hover:text-ink'
            }`}
          >
            {option.name}
          </button>
        ))}
      </div>
      <p className="mt-3 min-h-10 text-[13px] leading-relaxed text-faint">{current.detail}</p>
      <button
        type="button"
        onClick={onRun}
        className="mt-3 min-h-10 w-full rounded-md border border-ink-green/30 bg-pale-green px-4 py-2 text-[13.5px] font-medium text-ink-green transition-colors hover:bg-paper"
      >
        运行 {current.name}
      </button>
      {ran && (
        <div className="mt-4 border-l-2 border-ink-green pl-3">
          <div className="font-mono text-[9.5px] tracking-[0.14em] text-faint">结果摘要 · {current.score}</div>
          <p className="mt-1 text-[13px] leading-relaxed">{current.result}</p>
        </div>
      )}
    </div>
  );
}

function TutorPanel({
  activeTutorId,
  recommendedTutorId,
  activeLine,
  onTutorChange,
}: {
  activeTutorId: TutorId;
  recommendedTutorId: TutorId;
  activeLine: string;
  onTutorChange: (tutorId: TutorId) => void;
}) {
  const activeTutor = tutorOutfits.find((tutor) => tutor.id === activeTutorId) ?? tutorOutfits[0];
  return (
    <aside className="border-t border-hairline pt-6 lg:border-l lg:border-t-0 lg:pl-8 lg:pt-0">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="font-mono text-[9.5px] tracking-[0.16em] text-faint">AI TUTOR</div>
          <h2 className="mt-2 font-serif text-[21px] font-semibold">{activeTutor.name}</h2>
          <p className="mt-1 text-[12.5px] text-faint">{activeTutor.role}</p>
        </div>
        <div className="h-24 w-16 shrink-0 overflow-hidden">
          <img src={activeTutor.asset} alt="" className="h-full w-full object-contain object-top" />
        </div>
      </div>
      <p className="mt-5 border-l-2 border-ink-blue pl-3 text-[13.5px] leading-relaxed">{activeLine}</p>
      <div className="mt-5 grid gap-1.5">
        {tutorOutfits.map((tutor) => (
          <button
            key={tutor.id}
            type="button"
            onClick={() => onTutorChange(tutor.id)}
            className={`min-h-9 rounded-md border px-3 py-2 text-left text-[12px] transition-colors ${
              activeTutorId === tutor.id
                ? 'border-ink bg-ink text-paper'
                : tutor.id === recommendedTutorId
                  ? 'border-ink-blue/30 bg-pale-blue text-ink-blue hover:bg-paper'
                  : 'border-hairline bg-bone text-faint hover:bg-paper hover:text-ink'
            }`}
          >
            {tutor.name}
          </button>
        ))}
      </div>
    </aside>
  );
}

function MiniMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="border-l border-hairline pl-2.5">
      <div className="font-mono text-[9.5px] text-faint">{label}</div>
      <div className="mt-1 text-[13px] font-medium">{value}</div>
    </div>
  );
}
