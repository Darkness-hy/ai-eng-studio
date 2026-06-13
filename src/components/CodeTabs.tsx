import { useState } from 'react';
import { useEffect } from 'react';
import { highlightCode } from '../lib/highlight';
import { useLang } from '../lib/i18n';
import type { CodeFile } from '../lib/types';
import { CopyButton } from './CodeBlock';

export function CodeTabs({ files }: { files: CodeFile[] }) {
  const { t } = useLang();
  const [active, setActive] = useState(0);
  const [rendered, setRendered] = useState<{ key: string; html: string } | null>(null);
  const [runState, setRunState] = useState<'idle' | 'running' | 'done'>('idle');
  const [runOutput, setRunOutput] = useState<{ output: string; error: string | null; ms: number } | null>(null);

  const file = files[active];
  const html = rendered && rendered.key === file.name ? rendered.html : null;

  useEffect(() => {
    let live = true;
    highlightCode(file.content, file.lang).then((out) => {
      if (live) setRendered({ key: file.name, html: out });
    });
    return () => {
      live = false;
    };
  }, [file]);

  if (files.length === 0) return null;

  const run = async () => {
    setRunState('running');
    setRunOutput(null);
    try {
      const { runPython } = await import('../lib/pyodide');
      const result = await runPython(file.content);
      setRunOutput(result);
    } catch (err) {
      // Pyodide failed to load — show the error instead of hanging on "running".
      setRunOutput({ output: '', error: err instanceof Error ? err.message : String(err), ms: 0 });
    } finally {
      setRunState('done');
    }
  };

  return (
    <section className="my-10">
      <h2 className="mb-4 border-t border-hairline pt-6 font-serif text-[28px] font-semibold tracking-tight">
        {t('code_files')}
      </h2>
      <div className="code-shell overflow-hidden rounded-lg border border-hairline bg-paper">
        <div className="flex items-center justify-between gap-4 border-b border-hairline bg-bone/60 px-2">
          <div className="flex overflow-x-auto">
            {files.map((f, i) => (
              <button
                key={f.name}
                type="button"
                onClick={() => {
                  setActive(i);
                  setRunState('idle');
                  setRunOutput(null);
                }}
                className={`whitespace-nowrap border-b-2 px-3 py-2.5 font-mono text-[12px] transition-colors ${
                  i === active
                    ? 'border-ink text-ink'
                    : 'border-transparent text-faint hover:text-ink'
                }`}
              >
                {f.name}
              </button>
            ))}
          </div>
          <div className="flex shrink-0 items-center gap-4 pr-3">
            {file.runnable && (
              <button
                type="button"
                disabled={runState === 'running'}
                onClick={run}
                className="rounded-md bg-ink px-3 py-1 font-mono text-[11px] text-white transition-colors hover:bg-ink/80 disabled:opacity-50"
              >
                {runState === 'running' ? t('running') : `▸ ${t('run')}`}
              </button>
            )}
            <CopyButton text={file.content} />
          </div>
        </div>
        {/* No vertical max-height: a fixed-height inner scroller hijacks the
            page wheel (you get stuck scrolling up over long code). Code flows
            with the page like the prose code blocks; only x overflows. */}
        <div className="overflow-x-auto">
          {html ? (
            <div dangerouslySetInnerHTML={{ __html: html }} />
          ) : (
            <pre className="p-4 font-mono text-[13px] leading-relaxed">{file.content}</pre>
          )}
        </div>
        {(runState === 'running' || runOutput) && (
          <div className="border-t border-hairline bg-[#fcfcfb]">
            <div className="flex items-center justify-between px-4 py-1.5">
              <span className="font-mono text-[11px] uppercase tracking-[0.08em] text-faint">
                stdout
              </span>
              {runOutput && (
                <span className="font-mono text-[11px] text-faint">{Math.round(runOutput.ms)} ms</span>
              )}
            </div>
            <pre className="overflow-x-auto px-4 pb-4 font-mono text-[12.5px] leading-relaxed whitespace-pre-wrap">
              {runState === 'running' && !runOutput ? '加载 Python 运行时（首次约 10 MB）…' : null}
              {runOutput?.output}
              {runOutput?.error && <span className="text-ink-red">{runOutput.error}</span>}
            </pre>
          </div>
        )}
      </div>
    </section>
  );
}
