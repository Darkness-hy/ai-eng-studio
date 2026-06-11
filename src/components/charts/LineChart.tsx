export interface LinePoint {
  label: string;
  value: number;
}

export interface LineChartProps {
  points: LinePoint[];
}

const W = 640;
const H = 220;
const TOP = 14;
const BOTTOM = 24;
const LEFT = 34;
const RIGHT = 12;
const TICKS = [0, 0.25, 0.5, 0.75, 1];
const LINE = '#346538';
const GRID = '#eaeaea';
const LABEL = '#787774';

/** Round the y ceiling up to a multiple of 4 so quarter ticks stay integers. */
function niceMax(raw: number): number {
  return Math.max(4, Math.ceil(raw / 4) * 4);
}

/** Minimal editorial line chart: green-ink polyline with dots, hairline grid. */
export function LineChart({ points }: LineChartProps) {
  const max = niceMax(Math.max(0, ...points.map((p) => p.value)));
  const innerW = W - LEFT - RIGHT;
  const innerH = H - TOP - BOTTOM;
  const x = (i: number) =>
    points.length > 1 ? LEFT + (i / (points.length - 1)) * innerW : LEFT + innerW / 2;
  const y = (v: number) => TOP + innerH - (v / max) * innerH;
  const path = points.map((p, i) => `${x(i)},${y(p.value)}`).join(' ');

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block w-full" role="img">
      {TICKS.map((t) => {
        const gy = TOP + innerH - t * innerH;
        return (
          <g key={t}>
            <line x1={LEFT} y1={gy} x2={W - RIGHT} y2={gy} stroke={GRID} strokeWidth={1} />
            <text
              x={LEFT - 6}
              y={gy + 3}
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
      <polyline
        points={path}
        fill="none"
        stroke={LINE}
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      {points.map((p, i) => (
        <g key={`${p.label}-${i}`}>
          <title>{`${p.label} · ${p.value}`}</title>
          <circle cx={x(i)} cy={y(p.value)} r={2.4} fill={LINE} />
        </g>
      ))}
      {points.map((p, i) =>
        i % 5 === 0 || i === points.length - 1 ? (
          <text
            key={`label-${i}`}
            x={x(i)}
            y={H - 8}
            textAnchor={i === points.length - 1 ? 'end' : 'middle'}
            className="font-mono"
            fontSize={10}
            fill={LABEL}
          >
            {p.label}
          </text>
        ) : null,
      )}
    </svg>
  );
}
