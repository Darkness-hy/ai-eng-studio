export interface HistogramBucket {
  label: string;
  count: number;
}

export interface HistogramProps {
  buckets: HistogramBucket[];
}

const W = 640;
const H = 220;
const TOP = 14;
const BOTTOM = 24;
const LEFT = 34;
const RIGHT = 8;
const TICKS = [0, 0.25, 0.5, 0.75, 1];
const BAR = '#1f6c9f';
const GRID = '#eaeaea';
const LABEL = '#787774';

/** Minimal editorial histogram: ink-blue count bars on a hairline grid. */
export function Histogram({ buckets }: HistogramProps) {
  const rawMax = Math.max(0, ...buckets.map((b) => b.count));
  const max = Math.max(4, Math.ceil(rawMax / 4) * 4);
  const innerW = W - LEFT - RIGHT;
  const innerH = H - TOP - BOTTOM;
  const step = buckets.length > 0 ? innerW / buckets.length : innerW;
  const barW = Math.max(2, Math.min(step * 0.7, 52));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block w-full" role="img">
      {TICKS.map((t) => {
        const y = TOP + innerH - t * innerH;
        return (
          <g key={t}>
            <line x1={LEFT} y1={y} x2={W - RIGHT} y2={y} stroke={GRID} strokeWidth={1} />
            <text
              x={LEFT - 6}
              y={y + 3}
              textAnchor="end"
              className="font-mono"
              fontSize={10}
              fill={LABEL}
            >
              {Math.round(max * t)}
            </text>
          </g>
        );
      })}
      {buckets.map((b, i) => {
        const h = (b.count / max) * innerH;
        const x = LEFT + i * step + (step - barW) / 2;
        return (
          <g key={`${b.label}-${i}`}>
            <title>{`${b.label} · ${b.count}`}</title>
            <rect
              x={x}
              y={TOP + innerH - h}
              width={barW}
              height={h}
              rx={1.5}
              fill={BAR}
              fillOpacity={0.85}
            />
            <text
              x={LEFT + i * step + step / 2}
              y={H - 8}
              textAnchor="middle"
              className="font-mono"
              fontSize={10}
              fill={LABEL}
            >
              {b.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
