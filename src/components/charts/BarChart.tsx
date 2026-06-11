import { useState } from 'react';

export interface BarDatum {
  label: string;
  value: number;
  hint?: string;
}

export interface BarChartProps {
  data: BarDatum[];
  maxValue?: number;
  format?: (v: number) => string;
}

const W = 640;
const H = 232;
const TOP = 30;
const BOTTOM = 24;
const LEFT = 40;
const RIGHT = 8;
const TICKS = [0, 0.25, 0.5, 0.75, 1];
const BAR = '#1f6c9f';
const GRID = '#eaeaea';
const LABEL = '#787774';
const INK = '#2f3437';

/** Minimal editorial bar chart: ink-blue bars on a hairline grid, no chrome. */
export function BarChart({ data, maxValue, format = (v) => String(Math.round(v)) }: BarChartProps) {
  const [hover, setHover] = useState<number | null>(null);
  const max = maxValue ?? Math.max(1, ...data.map((d) => d.value));
  const innerW = W - LEFT - RIGHT;
  const innerH = H - TOP - BOTTOM;
  const step = data.length > 0 ? innerW / data.length : innerW;
  const barW = Math.max(2, Math.min(step * 0.6, 40));
  const hovered = hover != null ? data[hover] : null;

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
              {format(max * t)}
            </text>
          </g>
        );
      })}
      {data.map((d, i) => {
        const h = max > 0 ? (Math.min(Math.max(d.value, 0), max) / max) * innerH : 0;
        const x = LEFT + i * step + (step - barW) / 2;
        return (
          <g
            key={`${d.label}-${i}`}
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover((prev) => (prev === i ? null : prev))}
          >
            <title>{`${d.hint ?? d.label} · ${format(d.value)}`}</title>
            {/* full-column hit area so hover is easy even on short bars */}
            <rect x={LEFT + i * step} y={TOP} width={step} height={innerH} fill="transparent" />
            <rect
              x={x}
              y={TOP + innerH - h}
              width={barW}
              height={h}
              rx={1.5}
              fill={BAR}
              fillOpacity={hover === null ? 0.85 : hover === i ? 1 : 0.45}
            />
            <text
              x={LEFT + i * step + step / 2}
              y={H - 8}
              textAnchor="middle"
              className="font-mono"
              fontSize={10}
              fill={LABEL}
            >
              {d.label}
            </text>
          </g>
        );
      })}
      {hovered && (
        <text x={LEFT} y={16} className="font-mono" fontSize={10} fill={INK}>
          {`${hovered.hint ?? hovered.label} · ${format(hovered.value)}`}
        </text>
      )}
    </svg>
  );
}
