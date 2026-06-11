import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import { useLang } from '../../lib/i18n';

/** Loss landscape: non-convex quartic with a local minimum (right) and a global minimum (left). */
const f = (x: number): number => 0.08 * x ** 4 - 0.9 * x * x + 0.15 * x + 4;
const df = (x: number): number => 0.32 * x ** 3 - 1.8 * x + 0.15;

const X_MIN = -4.2;
const X_MAX = 4.2;
const Y_MIN = 0;
const Y_MAX = 14.5;

const W = 560;
const H = 300;
const ML = 42;
const MR = 14;
const MT = 14;
const MB = 28;

const sx = (x: number): number => ML + ((x - X_MIN) / (X_MAX - X_MIN)) * (W - ML - MR);
const sy = (y: number): number => H - MB - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * (H - MT - MB);

const CURVE_PATH = (() => {
  const parts: string[] = [];
  const n = 160;
  for (let i = 0; i <= n; i++) {
    const x = X_MIN + ((X_MAX - X_MIN) * i) / n;
    parts.push(`${i === 0 ? 'M' : 'L'}${sx(x).toFixed(2)},${sy(f(x)).toFixed(2)}`);
  }
  return parts.join(' ');
})();

/** Stationary points of f (numerically pre-solved). */
const GLOBAL_MIN_X = -2.412;
const LOCAL_MIN_X = 2.331;

const MAX_STEPS = 80;
const GRAD_TOL = 1e-3;
const DIVERGE_X = 6;
const DEFAULT_START = 3.4;

type Status = 'idle' | 'running' | 'converged' | 'diverged' | 'stopped';

interface RunState {
  traj: number[];
  status: Status;
}

function fmt(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return '—';
  if (Math.abs(v) >= 1000) return v.toExponential(1);
  return v.toFixed(digits);
}

export function GradientDescentViz() {
  const { lang } = useLang();
  const zh = lang === 'zh';
  const clipId = useId();

  const [start, setStart] = useState<number>(DEFAULT_START);
  const [eta, setEta] = useState<number>(0.1);
  const [playing, setPlaying] = useState<boolean>(false);
  const [run, setRun] = useState<RunState>({ traj: [DEFAULT_START], status: 'idle' });

  const svgRef = useRef<SVGSVGElement>(null);
  const draggingRef = useRef(false);

  const current = run.traj[run.traj.length - 1];
  const steps = run.traj.length - 1;
  const grad = df(current);

  const doStep = useCallback(() => {
    setRun((prev) => {
      if (prev.status === 'converged' || prev.status === 'diverged') return prev;
      const x = prev.traj[prev.traj.length - 1];
      const nx = x - eta * df(x);
      const traj = [...prev.traj, nx];
      let status: Status = 'running';
      if (Math.abs(nx) > DIVERGE_X) status = 'diverged';
      else if (Math.abs(df(nx)) < GRAD_TOL) status = 'converged';
      else if (traj.length - 1 >= MAX_STEPS) status = 'stopped';
      return { traj, status };
    });
  }, [eta]);

  // Playing is effective only while the run can still advance; reaching a
  // terminal status stops the interval without extra state writes.
  const activePlaying = playing && (run.status === 'idle' || run.status === 'running');

  useEffect(() => {
    if (!activePlaying) return;
    const id = window.setInterval(doStep, 250);
    return () => window.clearInterval(id);
  }, [activePlaying, doStep]);

  const reset = useCallback(() => {
    setPlaying(false);
    setRun({ traj: [start], status: 'idle' });
  }, [start]);

  const setStartFromClientX = useCallback((clientX: number) => {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const px = ((clientX - rect.left) / rect.width) * W;
    const x = Math.min(X_MAX, Math.max(X_MIN, X_MIN + ((px - ML) / (W - ML - MR)) * (X_MAX - X_MIN)));
    setPlaying(false);
    setStart(x);
    setRun({ traj: [x], status: 'idle' });
  }, []);

  const onPointerDown = useCallback(
    (e: React.PointerEvent<SVGSVGElement>) => {
      draggingRef.current = true;
      e.currentTarget.setPointerCapture(e.pointerId);
      setStartFromClientX(e.clientX);
    },
    [setStartFromClientX],
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent<SVGSVGElement>) => {
      if (draggingRef.current) setStartFromClientX(e.clientX);
    },
    [setStartFromClientX],
  );

  const onPointerUp = useCallback(() => {
    draggingRef.current = false;
  }, []);

  /** True when the last few steps bounce between the two sides of a valley. */
  const oscillating = useMemo(() => {
    const t = run.traj;
    if (t.length < 4) return false;
    const d1 = t[t.length - 1] - t[t.length - 2];
    const d2 = t[t.length - 2] - t[t.length - 3];
    const d3 = t[t.length - 3] - t[t.length - 4];
    return d1 * d2 < 0 && d2 * d3 < 0 && Math.abs(d1) > 0.3;
  }, [run.traj]);

  const commentary = useMemo(() => {
    if (run.status === 'diverged' || (oscillating && eta > 0.2)) {
      return zh
        ? '学习率太大，在谷底两侧来回震荡甚至发散。'
        : 'Learning rate too large — bouncing across the valley, even diverging.';
    }
    if ((run.status === 'converged' || run.status === 'stopped') && current > 1) {
      return zh
        ? '陷在局部极小值——这就是为什么需要动量或重启。'
        : 'Stuck in a local minimum — this is why momentum and restarts exist.';
    }
    if (run.status === 'converged') {
      return zh ? '收敛到全局最小值附近。' : 'Converged near the global minimum.';
    }
    if (run.status === 'stopped') {
      return zh ? '到达 80 步上限，仍未满足收敛条件。' : 'Hit the 80-step cap before meeting the convergence test.';
    }
    if (eta <= 0.02 && steps >= 5) {
      return zh ? '步子太小，收敛慢。' : 'Steps too small — convergence is slow.';
    }
    if (run.status === 'running') {
      return zh ? '沿负梯度方向一步步滑向谷底。' : 'Sliding step by step down the negative gradient.';
    }
    return zh
      ? '点击或拖动曲线选择起点，然后「单步」或「播放」。'
      : 'Click or drag on the curve to pick a start, then Step or Play.';
  }, [run.status, oscillating, eta, current, steps, zh]);

  const tangent = useMemo(() => {
    const m = df(current);
    const halfW = 0.7;
    return {
      x1: sx(current - halfW),
      y1: sy(f(current) - halfW * m),
      x2: sx(current + halfW),
      y2: sy(f(current) + halfW * m),
    };
  }, [current]);

  const xTicks = [-4, -2, 0, 2, 4];
  const yTicks = [0, 4, 8, 12];

  return (
    <section className="my-10 overflow-hidden rounded-lg border border-hairline bg-paper">
      <header className="flex items-baseline justify-between border-b border-hairline px-6 py-4">
        <div className="flex items-baseline gap-3">
          <span className="rounded-full bg-pale-blue px-2.5 py-0.5 text-[11px] font-medium uppercase tracking-[0.05em] text-ink-blue">
            {zh ? '交互实验' : 'Interactive'}
          </span>
          <h3 className="font-serif text-[15px] text-ink">{zh ? '梯度下降实验台' : 'Gradient Descent Lab'}</h3>
        </div>
        <span className="text-[11px] text-faint">{zh ? '点击曲线设置起点' : 'Click the curve to set a start'}</span>
      </header>

      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="block w-full cursor-crosshair touch-none select-none"
        role="img"
        aria-label={zh ? '一维损失曲线上的梯度下降' : 'Gradient descent on a 1D loss curve'}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        <defs>
          <clipPath id={clipId}>
            <rect x={ML} y={MT} width={W - ML - MR} height={H - MT - MB} />
          </clipPath>
        </defs>

        {/* grid + axes */}
        {xTicks.map((t) => (
          <g key={`x${t}`}>
            <line x1={sx(t)} y1={MT} x2={sx(t)} y2={H - MB} stroke="#eaeaea" strokeWidth={1} />
            <text x={sx(t)} y={H - MB + 16} textAnchor="middle" fontSize={10} fill="#787774" fontFamily="monospace">
              {t}
            </text>
          </g>
        ))}
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={ML} y1={sy(t)} x2={W - MR} y2={sy(t)} stroke="#eaeaea" strokeWidth={1} />
            <text x={ML - 6} y={sy(t) + 3} textAnchor="end" fontSize={10} fill="#787774" fontFamily="monospace">
              {t}
            </text>
          </g>
        ))}
        <text x={W - MR} y={H - MB + 16} textAnchor="end" fontSize={10} fill="#787774" fontFamily="monospace">
          x
        </text>
        <text x={ML - 6} y={MT + 4} textAnchor="end" fontSize={10} fill="#787774" fontFamily="monospace">
          f(x)
        </text>

        {/* valley annotations */}
        <text
          x={sx(GLOBAL_MIN_X)}
          y={sy(f(GLOBAL_MIN_X)) + 16}
          textAnchor="middle"
          fontSize={10}
          fill="#787774"
          fontFamily="monospace"
        >
          {zh ? '全局最小' : 'global min'}
        </text>
        <text
          x={sx(LOCAL_MIN_X)}
          y={sy(f(LOCAL_MIN_X)) + 16}
          textAnchor="middle"
          fontSize={10}
          fill="#787774"
          fontFamily="monospace"
        >
          {zh ? '局部极小' : 'local min'}
        </text>

        <g clipPath={`url(#${clipId})`}>
          {/* loss curve */}
          <path d={CURVE_PATH} fill="none" stroke="#2f3437" strokeWidth={1.5} />

          {/* descent trajectory */}
          {run.traj.length > 1 && (
            <polyline
              points={run.traj.map((x) => `${sx(x).toFixed(2)},${sy(f(x)).toFixed(2)}`).join(' ')}
              fill="none"
              stroke="#1f6c9f"
              strokeWidth={1}
              opacity={0.5}
            />
          )}
          {run.traj.slice(0, -1).map((x, i) => (
            <circle
              key={i}
              cx={sx(x)}
              cy={sy(f(x))}
              r={3}
              fill="#1f6c9f"
              opacity={0.2 + 0.6 * (i / Math.max(1, run.traj.length - 1))}
            />
          ))}

          {/* tangent at current point */}
          <line
            x1={tangent.x1}
            y1={tangent.y1}
            x2={tangent.x2}
            y2={tangent.y2}
            stroke="#9f2f2d"
            strokeWidth={1.25}
            opacity={0.6}
          />

          {/* current point */}
          <circle cx={sx(current)} cy={sy(f(current))} r={5} fill="#9f2f2d" />
        </g>
      </svg>

      {/* controls */}
      <div className="flex flex-wrap items-center gap-3 border-t border-hairline px-6 py-4">
        <button type="button" onClick={() => setPlaying((p) => !p)} className="rounded-md bg-ink px-3 py-1 font-mono text-[11px] text-white hover:bg-ink/80">
          {activePlaying ? (zh ? '暂停' : 'Pause') : zh ? '播放' : 'Play'}
        </button>
        <button type="button" onClick={doStep} className="rounded-md border border-hairline px-3 py-1 font-mono text-[11px] text-faint hover:bg-bone">
          {zh ? '单步' : 'Step'}
        </button>
        <button type="button" onClick={reset} className="rounded-md border border-hairline px-3 py-1 font-mono text-[11px] text-faint hover:bg-bone">
          {zh ? '重置' : 'Reset'}
        </button>

        <label className="ml-2 flex items-center gap-2 font-mono text-[11px] text-faint">
          <span>η</span>
          <input
            type="range"
            min={0.005}
            max={0.6}
            step={0.005}
            value={eta}
            onChange={(e) => setEta(Number(e.target.value))}
            className="h-1 w-28 accent-[#1f6c9f]"
            aria-label={zh ? '学习率' : 'Learning rate'}
          />
          <span className="text-ink">{eta.toFixed(3)}</span>
        </label>

        {run.status === 'converged' && (
          <span className="rounded-full bg-pale-green px-2.5 py-0.5 text-[11px] font-medium text-ink-green">
            {zh ? '收敛' : 'Converged'}
          </span>
        )}
        {run.status === 'diverged' && (
          <span className="rounded-full bg-pale-red px-2.5 py-0.5 text-[11px] font-medium text-ink-red">
            {zh ? '发散' : 'Diverged'}
          </span>
        )}
        {run.status === 'stopped' && (
          <span className="rounded-full bg-pale-yellow px-2.5 py-0.5 text-[11px] font-medium text-ink-yellow">
            {zh ? '已停止' : 'Stopped'}
          </span>
        )}
      </div>

      {/* readouts */}
      <div className="flex flex-wrap gap-x-5 gap-y-1 px-6 pb-2 font-mono text-[11px] text-faint">
        <span>
          x = <span className="text-ink">{fmt(current)}</span>
        </span>
        <span>
          f(x) = <span className="text-ink">{fmt(f(current))}</span>
        </span>
        <span>
          f′(x) = <span className="text-ink">{fmt(grad, 4)}</span>
        </span>
        <span>
          η = <span className="text-ink">{eta.toFixed(3)}</span>
        </span>
        <span>
          {zh ? '步数' : 'steps'} = <span className="text-ink">{steps}</span>
        </span>
      </div>

      <p className="px-6 pb-4 text-[12.5px] text-faint">{commentary}</p>

      <footer className="border-t border-hairline bg-bone/50 px-6 py-3 text-[12.5px] text-faint">
        {zh
          ? '训练神经网络时，优化器做的就是这件事——只是维度从 1 变成了几十亿。'
          : 'This is exactly what an optimizer does when training a neural network — except the dimension goes from 1 to billions.'}
      </footer>
    </section>
  );
}
