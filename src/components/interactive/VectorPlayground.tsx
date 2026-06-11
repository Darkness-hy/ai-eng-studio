import { useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { useLang } from '../../lib/i18n';

const SIZE = 420;
const CENTER = SIZE / 2;
const UNIT = 38;
const RANGE = 5;

const COLOR_GRID = '#eaeaea';
const COLOR_AXIS = '#a8a7a3';
const COLOR_A = '#1f6c9f';
const COLOR_B = '#9f2f2d';
const COLOR_PROJ = '#346538';
const COLOR_TEXT = '#787774';

interface Vec {
  x: number;
  y: number;
}

function snap(n: number): number {
  return Math.round(n * 10) / 10;
}

function clampCoord(n: number): number {
  return Math.max(-RANGE, Math.min(RANGE, n));
}

function toPx(v: Vec): { x: number; y: number } {
  return { x: CENTER + v.x * UNIT, y: CENTER - v.y * UNIT };
}

function arrowHead(v: Vec): string | null {
  const len = Math.hypot(v.x, v.y);
  if (len < 0.15) return null;
  const tip = toPx(v);
  const ux = (tip.x - CENTER) / (len * UNIT);
  const uy = (tip.y - CENTER) / (len * UNIT);
  const bx = tip.x - ux * 11;
  const by = tip.y - uy * 11;
  return `${tip.x},${tip.y} ${bx - uy * 4.5},${by + ux * 4.5} ${bx + uy * 4.5},${by - ux * 4.5}`;
}

function tipLabelPos(v: Vec): { x: number; y: number } {
  const len = Math.hypot(v.x, v.y) || 1;
  const tip = toPx(v);
  return { x: tip.x + (v.x / len) * 16, y: tip.y - (v.y / len) * 16 + 4 };
}

export function VectorPlayground() {
  const { lang } = useLang();
  const svgRef = useRef<SVGSVGElement>(null);
  const [a, setA] = useState<Vec>({ x: 3, y: 2 });
  const [b, setB] = useState<Vec>({ x: 1, y: 3 });
  const [dragging, setDragging] = useState<'a' | 'b' | null>(null);
  const [showProj, setShowProj] = useState(true);

  // Derived values, computed inline each render
  const dot = a.x * b.x + a.y * b.y;
  const magA = Math.hypot(a.x, a.y);
  const magB = Math.hypot(b.x, b.y);
  const denom = magA * magB;
  const cos = denom > 1e-9 ? Math.max(-1, Math.min(1, dot / denom)) : 0;
  const angleDeg = denom > 1e-9 ? (Math.acos(cos) * 180) / Math.PI : 0;
  const nearZero = Math.abs(cos) < 0.08 || denom <= 1e-9;

  // Projection of a onto b
  const tScalar = magB > 1e-9 ? dot / (magB * magB) : 0;
  const proj: Vec = { x: tScalar * b.x, y: tScalar * b.y };
  const perpDist = Math.hypot(a.x - proj.x, a.y - proj.y);

  const aPx = toPx(a);
  const bPx = toPx(b);
  const projPx = toPx(proj);

  // Right-angle marker at the projection foot
  let markerPath: string | null = null;
  if (showProj && magB > 0.05 && perpDist > 0.2 && Math.abs(tScalar) * magB > 0.2) {
    const ubx = (bPx.x - CENTER) / (magB * UNIT);
    const uby = (bPx.y - CENTER) / (magB * UNIT);
    const sign = tScalar >= 0 ? -1 : 1;
    const l1x = sign * ubx;
    const l1y = sign * uby;
    const l2x = (aPx.x - projPx.x) / (perpDist * UNIT);
    const l2y = (aPx.y - projPx.y) / (perpDist * UNIT);
    const s = 8;
    markerPath =
      `M ${projPx.x + l1x * s} ${projPx.y + l1y * s} ` +
      `L ${projPx.x + (l1x + l2x) * s} ${projPx.y + (l1y + l2y) * s} ` +
      `L ${projPx.x + l2x * s} ${projPx.y + l2y * s}`;
  }

  const clientToVec = (svg: SVGSVGElement | null, clientX: number, clientY: number): Vec | null => {
    if (!svg) return null;
    const rect = svg.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return null;
    const px = ((clientX - rect.left) / rect.width) * SIZE;
    const py = ((clientY - rect.top) / rect.height) * SIZE;
    return {
      x: clampCoord(snap((px - CENTER) / UNIT)),
      y: clampCoord(snap((CENTER - py) / UNIT)),
    };
  };

  const handleProps = (which: 'a' | 'b') => ({
    onPointerDown: (e: ReactPointerEvent<SVGGElement>) => {
      e.currentTarget.setPointerCapture(e.pointerId);
      setDragging(which);
    },
    onPointerMove: (e: ReactPointerEvent<SVGGElement>) => {
      if (dragging !== which) return;
      const v = clientToVec(e.currentTarget.ownerSVGElement, e.clientX, e.clientY);
      if (!v) return;
      if (which === 'a') setA(v);
      else setB(v);
    },
    onPointerUp: (e: ReactPointerEvent<SVGGElement>) => {
      if (e.currentTarget.hasPointerCapture(e.pointerId)) {
        e.currentTarget.releasePointerCapture(e.pointerId);
      }
      setDragging(null);
    },
  });

  const ticks = Array.from({ length: 2 * RANGE + 1 }, (_, i) => i - RANGE);

  const relation: 'similar' | 'orthogonal' | 'opposite' = nearZero
    ? 'orthogonal'
    : dot > 0
      ? 'similar'
      : 'opposite';

  const chipClass =
    relation === 'similar'
      ? 'bg-pale-green text-ink-green'
      : relation === 'orthogonal'
        ? 'bg-pale-yellow text-ink-yellow'
        : 'bg-pale-red text-ink-red';

  const chipText =
    relation === 'similar'
      ? lang === 'zh'
        ? '相似'
        : 'similar'
      : relation === 'orthogonal'
        ? lang === 'zh'
          ? '近乎正交'
          : 'orthogonal'
        : lang === 'zh'
          ? '相反'
          : 'opposite';

  const explanation =
    relation === 'similar'
      ? lang === 'zh'
        ? 'a·b > 0：两向量方向相近——这就是相似度检索的数学基础。'
        : 'a·b > 0: the vectors point in similar directions — the math behind similarity search.'
      : relation === 'orthogonal'
        ? lang === 'zh'
          ? 'a·b ≈ 0：两向量近乎正交，彼此几乎不携带对方的信息。'
          : 'a·b ≈ 0: nearly orthogonal — the vectors carry almost no information about each other.'
        : lang === 'zh'
          ? 'a·b < 0：两向量方向相反，在语义空间里它们“背道而驰”。'
          : 'a·b < 0: the vectors point in opposite directions — semantically they disagree.';

  const fmtVec = (v: Vec) => `[${v.x.toFixed(1)}, ${v.y.toFixed(1)}]`;

  const rows: Array<{ label: string; value: string; color?: string }> = [
    { label: 'a', value: fmtVec(a), color: COLOR_A },
    { label: 'b', value: fmtVec(b), color: COLOR_B },
    { label: '|a|', value: magA.toFixed(2) },
    { label: '|b|', value: magB.toFixed(2) },
    { label: 'cosθ', value: cos.toFixed(2) },
    { label: 'θ', value: `${angleDeg.toFixed(1)}°` },
  ];

  return (
    <section className="my-10 overflow-hidden rounded-lg border border-hairline bg-paper">
      <header className="flex items-baseline justify-between border-b border-hairline px-6 py-4">
        <div className="flex items-baseline gap-3">
          <span className="rounded-full bg-pale-blue px-2.5 py-0.5 text-[11px] font-medium uppercase tracking-[0.05em] text-ink-blue">
            {lang === 'zh' ? '交互实验' : 'Interactive'}
          </span>
          <h3 className="font-serif text-[17px] text-ink">
            {lang === 'zh' ? '向量游乐场' : 'Vector Playground'}
          </h3>
        </div>
        <p className="text-[12px] text-faint">
          {lang === 'zh' ? '拖动箭头端点' : 'Drag the arrow tips'}
        </p>
      </header>

      <div className="grid gap-6 p-6 md:grid-cols-[1fr_240px]">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          className="w-full touch-none select-none"
          role="img"
          aria-label={lang === 'zh' ? '二维向量坐标系' : '2D vector coordinate grid'}
        >
          {/* Grid */}
          {ticks.map((i) =>
            i === 0 ? null : (
              <g key={i} stroke={COLOR_GRID} strokeWidth="1">
                <line
                  x1={CENTER + i * UNIT}
                  y1={CENTER - RANGE * UNIT}
                  x2={CENTER + i * UNIT}
                  y2={CENTER + RANGE * UNIT}
                />
                <line
                  x1={CENTER - RANGE * UNIT}
                  y1={CENTER + i * UNIT}
                  x2={CENTER + RANGE * UNIT}
                  y2={CENTER + i * UNIT}
                />
              </g>
            ),
          )}
          {/* Axes */}
          <line
            x1={CENTER - RANGE * UNIT}
            y1={CENTER}
            x2={CENTER + RANGE * UNIT}
            y2={CENTER}
            stroke={COLOR_AXIS}
            strokeWidth="1"
          />
          <line
            x1={CENTER}
            y1={CENTER - RANGE * UNIT}
            x2={CENTER}
            y2={CENTER + RANGE * UNIT}
            stroke={COLOR_AXIS}
            strokeWidth="1"
          />
          {/* Tick labels */}
          {ticks.map((i) =>
            i === 0 ? null : (
              <g
                key={`t${i}`}
                fill={COLOR_TEXT}
                fontSize="10"
                fontFamily="'JetBrains Mono', monospace"
              >
                <text x={CENTER + i * UNIT} y={CENTER + 14} textAnchor="middle">
                  {i}
                </text>
                <text x={CENTER - 7} y={CENTER - i * UNIT + 3} textAnchor="end">
                  {i}
                </text>
              </g>
            ),
          )}

          {/* Projection of a onto b */}
          {showProj && magB > 0.05 && (
            <g>
              <line
                x1={CENTER}
                y1={CENTER}
                x2={projPx.x}
                y2={projPx.y}
                stroke={COLOR_PROJ}
                strokeWidth="2"
                strokeDasharray="5 4"
              />
              <line
                x1={aPx.x}
                y1={aPx.y}
                x2={projPx.x}
                y2={projPx.y}
                stroke={COLOR_PROJ}
                strokeWidth="1"
                strokeDasharray="2 4"
                opacity="0.5"
              />
              {markerPath && (
                <path d={markerPath} fill="none" stroke={COLOR_PROJ} strokeWidth="1" />
              )}
              {Math.abs(tScalar) * magB > 0.4 && (
                <text
                  x={(CENTER + projPx.x) / 2 + 8}
                  y={(CENTER + projPx.y) / 2 + 12}
                  fill={COLOR_PROJ}
                  fontSize="10"
                  fontFamily="'JetBrains Mono', monospace"
                >
                  proj
                </text>
              )}
            </g>
          )}

          {/* Vector b (red) */}
          <line
            x1={CENTER}
            y1={CENTER}
            x2={bPx.x}
            y2={bPx.y}
            stroke={COLOR_B}
            strokeWidth="2"
          />
          {arrowHead(b) && <polygon points={arrowHead(b)!} fill={COLOR_B} />}
          <text
            {...tipLabelPos(b)}
            fill={COLOR_B}
            fontSize="11"
            fontFamily="'JetBrains Mono', monospace"
            textAnchor="middle"
          >
            b
          </text>

          {/* Vector a (blue) */}
          <line
            x1={CENTER}
            y1={CENTER}
            x2={aPx.x}
            y2={aPx.y}
            stroke={COLOR_A}
            strokeWidth="2"
          />
          {arrowHead(a) && <polygon points={arrowHead(a)!} fill={COLOR_A} />}
          <text
            {...tipLabelPos(a)}
            fill={COLOR_A}
            fontSize="11"
            fontFamily="'JetBrains Mono', monospace"
            textAnchor="middle"
          >
            a
          </text>

          {/* Drag handles */}
          <g
            {...handleProps('b')}
            className={dragging === 'b' ? 'cursor-grabbing' : 'cursor-grab'}
          >
            <circle cx={bPx.x} cy={bPx.y} r="14" fill="transparent" />
            <circle cx={bPx.x} cy={bPx.y} r="6" fill="#fff" stroke={COLOR_B} strokeWidth="2" />
          </g>
          <g
            {...handleProps('a')}
            className={dragging === 'a' ? 'cursor-grabbing' : 'cursor-grab'}
          >
            <circle cx={aPx.x} cy={aPx.y} r="14" fill="transparent" />
            <circle cx={aPx.x} cy={aPx.y} r="6" fill="#fff" stroke={COLOR_A} strokeWidth="2" />
          </g>
        </svg>

        {/* Readout panel */}
        <div className="flex flex-col gap-3">
          <div className="rounded-lg border border-hairline bg-bone px-4 py-3">
            <div className="flex items-center justify-between py-1">
              <span className="font-mono text-[12px] text-faint">a·b</span>
              <span className="flex items-center gap-2">
                <span className="font-mono text-[13px] text-ink">{dot.toFixed(2)}</span>
                <span
                  className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${chipClass}`}
                >
                  {chipText}
                </span>
              </span>
            </div>
            <div className="my-1 border-t border-hairline" />
            {rows.map((row) => (
              <div key={row.label} className="flex items-center justify-between py-1">
                <span className="flex items-center gap-1.5 font-mono text-[12px] text-faint">
                  {row.color && (
                    <span
                      className="inline-block h-2 w-2 rounded-full"
                      style={{ background: row.color }}
                    />
                  )}
                  {row.label}
                </span>
                <span className="font-mono text-[13px] text-ink">{row.value}</span>
              </div>
            ))}
          </div>

          <label className="flex cursor-pointer items-center gap-2 text-[12px] text-faint">
            <input
              type="checkbox"
              checked={showProj}
              onChange={(e) => setShowProj(e.target.checked)}
              className="h-3.5 w-3.5 accent-[#346538]"
            />
            {lang === 'zh' ? '投影' : 'Projection'}
            <span className="font-mono text-[11px]">proj_b(a)</span>
          </label>

          <p className="text-[13px] leading-relaxed text-faint">{explanation}</p>
        </div>
      </div>

      <footer className="border-t border-hairline bg-bone/50 px-6 py-3 text-[12.5px] text-faint">
        {lang === 'zh'
          ? 'embedding 相似度、注意力分数，本质都是这个点积。'
          : 'Embedding similarity and attention scores are, at their core, this same dot product.'}
      </footer>
    </section>
  );
}
