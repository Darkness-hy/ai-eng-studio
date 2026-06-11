import { useEffect, useState } from 'react';

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
  const [result, setResult] = useState<{ key: string; svg?: string; failed?: boolean } | null>(null);
  const svg = result && result.key === chart ? result.svg : undefined;
  const failed = result != null && result.key === chart && result.failed === true;

  useEffect(() => {
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
  }, [chart]);

  if (failed) {
    return (
      <pre className="overflow-x-auto rounded-lg border border-hairline bg-bone p-4 font-mono text-xs text-faint">
        {chart}
      </pre>
    );
  }
  if (!svg) {
    return <div className="my-6 h-40 animate-pulse rounded-lg border border-hairline bg-bone" />;
  }
  return (
    <div
      className="mermaid-box my-6 overflow-x-auto rounded-lg border border-hairline bg-paper p-5"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
