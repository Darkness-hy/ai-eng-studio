import { useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { phaseTitle, useLang } from '../lib/i18n';
import { useProgress } from '../lib/progress';
import type { CourseIndex, PhaseIdx } from '../lib/types';

// Dependency layers of the 20-phase DAG (from upstream README flowchart).
const LAYERS: number[][] = [
  [0],
  [1],
  [2],
  [3],
  [4, 5, 6, 9],
  [7],
  [8, 10],
  [11, 12],
  [13],
  [14],
  [15, 17],
  [16, 18],
  [19],
];

const ROW_H = 116;
const NODE_H = 78;

interface NodePos {
  phase: PhaseIdx;
  x: number; // center, px
  y: number; // top, px
}

export function RoadmapSpine({ index }: { index: CourseIndex }) {
  const { lang } = useLang();
  const progress = useProgress();
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const byNum = useMemo(() => new Map(index.phases.map((p) => [p.num, p])), [index]);

  const nodes = useMemo<NodePos[]>(() => {
    if (!width) return [];
    return LAYERS.flatMap((layer, li) =>
      layer.map((num, i) => ({
        phase: byNum.get(num)!,
        x: ((i + 1) / (layer.length + 1)) * width,
        y: li * ROW_H,
      })),
    ).filter((n) => n.phase);
  }, [width, byNum]);

  const nodeByNum = useMemo(() => new Map(nodes.map((n) => [n.phase.num, n])), [nodes]);

  const edges = useMemo(() => {
    const out: { from: NodePos; to: NodePos }[] = [];
    for (const node of nodes) {
      for (const dep of node.phase.deps) {
        const from = nodeByNum.get(dep);
        if (from) out.push({ from, to: node });
      }
    }
    return out;
  }, [nodes, nodeByNum]);

  const height = LAYERS.length * ROW_H - (ROW_H - NODE_H);

  const doneCount = (p: PhaseIdx) =>
    p.lessons.filter((l) => progress.lessons[`${p.slug}/${l.slug}`]?.done).length;

  return (
    <div ref={containerRef} className="relative hidden md:block" style={{ height }}>
      <svg
        className="pointer-events-none absolute inset-0"
        width={width}
        height={height}
        aria-hidden
      >
        {edges.map(({ from, to }, i) => {
          const x1 = from.x;
          const y1 = from.y + NODE_H;
          const x2 = to.x;
          const y2 = to.y;
          const mid = (y1 + y2) / 2;
          return (
            <path
              key={i}
              d={`M ${x1} ${y1} C ${x1} ${mid}, ${x2} ${mid}, ${x2} ${y2}`}
              fill="none"
              stroke="#dddcd8"
              strokeWidth="1.2"
            />
          );
        })}
      </svg>

      {nodes.map(({ phase, x, y }, i) => {
        const done = doneCount(phase);
        const total = phase.lessons.length;
        const pct = total ? done / total : 0;
        const state = pct === 1 ? 'done' : pct > 0 ? 'active' : 'idle';
        return (
          <Link
            key={phase.slug}
            to={`/phase/${phase.slug}`}
            className="rise group absolute w-48 -translate-x-1/2 rounded-lg border border-hairline bg-paper px-4 py-3 transition-all duration-200 hover:-translate-y-0.5 hover:shadow-lift"
            style={{ left: x, top: y, height: NODE_H, ['--stagger' as string]: i % 8 }}
          >
            <div className="flex items-center justify-between">
              <span className="font-mono text-[10px] tracking-[0.14em] text-faint">
                PHASE {String(phase.num).padStart(2, '0')}
              </span>
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  state === 'done' ? 'bg-ink-green' : state === 'active' ? 'bg-ink-yellow' : 'bg-hairline'
                }`}
              />
            </div>
            <div className="mt-0.5 truncate text-[13.5px] font-medium group-hover:text-ink">
              {phaseTitle(phase, lang)}
            </div>
            <div className="mt-1.5 flex items-center gap-2">
              <div className="h-[3px] flex-1 overflow-hidden rounded-full bg-bone">
                <div
                  className="h-full rounded-full bg-ink-green transition-all"
                  style={{ width: `${pct * 100}%` }}
                />
              </div>
              <span className="font-mono text-[10px] text-faint">
                {done}/{total}
              </span>
            </div>
          </Link>
        );
      })}
    </div>
  );
}

/** Mobile fallback: simple stacked phase cards. */
export function RoadmapList({ index }: { index: CourseIndex }) {
  const { lang } = useLang();
  const progress = useProgress();
  return (
    <div className="grid gap-3 md:hidden">
      {index.phases.map((phase) => {
        const done = phase.lessons.filter((l) => progress.lessons[`${phase.slug}/${l.slug}`]?.done).length;
        return (
          <Link
            key={phase.slug}
            to={`/phase/${phase.slug}`}
            className="rounded-lg border border-hairline bg-paper px-4 py-3"
          >
            <div className="font-mono text-[10px] tracking-[0.14em] text-faint">
              PHASE {String(phase.num).padStart(2, '0')}
            </div>
            <div className="mt-0.5 flex items-baseline justify-between">
              <span className="text-[14.5px] font-medium">{phaseTitle(phase, lang)}</span>
              <span className="font-mono text-[11px] text-faint">
                {done}/{phase.lessons.length}
              </span>
            </div>
          </Link>
        );
      })}
    </div>
  );
}
