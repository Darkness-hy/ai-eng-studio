import { useEffect, useRef, useState } from 'react';

type MermaidApi = typeof import('mermaid').default;

let mermaidPromise: Promise<MermaidApi> | null = null;
let renderSeq = 0;

function getMermaid(): Promise<MermaidApi> {
  if (!mermaidPromise) {
    mermaidPromise = import('mermaid').then((mod) => {
      mod.default.initialize({
        startOnLoad: false,
        theme: 'neutral',
        fontFamily: "'SF Pro Display', -apple-system, 'PingFang SC', sans-serif",
        themeVariables: {
          primaryColor: '#f7f6f3',
          primaryBorderColor: '#d9d8d4',
          primaryTextColor: '#2f3437',
          lineColor: '#a8a7a3',
          secondaryColor: '#e1f3fe',
          tertiaryColor: '#edf3ec',
          fontSize: '13px',
        },
      });
      return mod.default;
    });
  }
  return mermaidPromise;
}

export function MermaidDiagram({ chart }: { chart: string }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const [result, setResult] = useState<{ key: string; svg?: string; failed?: boolean } | null>(null);
  // Render only once the diagram is near the viewport. With several diagrams on a
  // page, rendering them all on mount makes each one resize from the placeholder to
  // its (often much taller) natural height at different times — content above the
  // viewport grows late and the page scroll jumps. Deferring keeps the growth at or
  // below the fold (diagrams you've already scrolled past are settled).
  const [near, setNear] = useState(() => typeof IntersectionObserver === 'undefined');

  useEffect(() => {
    if (near) return;
    const el = hostRef.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setNear(true);
          io.disconnect();
        }
      },
      { rootMargin: '0px 0px 800px 0px' },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [near]);

  useEffect(() => {
    if (!near) return;
    let live = true;
    getMermaid()
      .then((mermaid) => mermaid.render(`aes-mmd-${++renderSeq}`, chart))
      .then(({ svg: rendered }) => {
        if (live) setResult({ key: chart, svg: rendered });
      })
      .catch(() => {
        if (live) setResult({ key: chart, failed: true });
      });
    return () => {
      live = false;
    };
  }, [chart, near]);

  const svg = result && result.key === chart ? result.svg : undefined;
  const failed = result != null && result.key === chart && result.failed === true;

  // The outer host element is stable across loading → rendered so the browser keeps
  // its scroll anchor; only the inner content swaps.
  return (
    <div ref={hostRef} className="my-6">
      {failed ? (
        <pre className="overflow-x-auto rounded-lg border border-hairline bg-bone p-4 font-mono text-xs text-faint">
          {chart}
        </pre>
      ) : svg ? (
        <div
          className="mermaid-box overflow-x-auto rounded-lg border border-hairline bg-paper p-5"
          dangerouslySetInnerHTML={{ __html: svg }}
        />
      ) : (
        <div className="h-40 animate-pulse rounded-lg border border-hairline bg-bone" />
      )}
    </div>
  );
}
