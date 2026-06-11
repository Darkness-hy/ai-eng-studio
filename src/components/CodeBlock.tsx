import { useEffect, useState } from 'react';
import { highlightCode } from '../lib/highlight';
import { useLang } from '../lib/i18n';

export function CopyButton({ text }: { text: string }) {
  const { t } = useLang();
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className="font-mono text-[11px] tracking-wide text-faint transition-colors hover:text-ink"
      onClick={() => {
        navigator.clipboard.writeText(text).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1600);
        });
      }}
    >
      {copied ? t('copied') : t('copy')}
    </button>
  );
}

export function CodeBlock({ code, lang }: { code: string; lang: string }) {
  const [rendered, setRendered] = useState<{ key: string; html: string } | null>(null);
  const html = rendered && rendered.key === code ? rendered.html : null;

  useEffect(() => {
    let live = true;
    highlightCode(code, lang).then((out) => {
      if (live) setRendered({ key: code, html: out });
    });
    return () => {
      live = false;
    };
  }, [code, lang]);

  return (
    <div className="code-shell my-6 overflow-hidden rounded-lg border border-hairline bg-paper">
      <div className="flex items-center justify-between border-b border-hairline bg-bone/60 px-4 py-1.5">
        <span className="font-mono text-[11px] uppercase tracking-[0.08em] text-faint">{lang}</span>
        <CopyButton text={code} />
      </div>
      {html ? (
        <div dangerouslySetInnerHTML={{ __html: html }} />
      ) : (
        <pre className="overflow-x-auto p-4 font-mono text-[13px] leading-relaxed text-ink">
          {code}
        </pre>
      )}
    </div>
  );
}
