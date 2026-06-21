import type { ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { slugify } from '../lib/md';
import { CodeBlock } from './CodeBlock';
import { LessonFigure } from './LessonFigure';
import { MermaidDiagram } from './Mermaid';

function flatten(node: ReactNode): string {
  if (node == null || typeof node === 'boolean') return '';
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(flatten).join('');
  if (typeof node === 'object' && 'props' in node) {
    return flatten((node.props as { children?: ReactNode }).children);
  }
  return '';
}

// Lessons write math as inline `code` with ASCII/Unicode pseudo-notation
// (e.g. `V^π(s) = (1/N) Σ_i G^{(i)}(s)`, `V_{n-1}(s)`), not LaTeX. Detect such
// spans conservatively — real code like `max_steps` / `env.reset()` / `O(T)` is
// left untouched — and render ^ / _ as proper super/sub-scripts in the prose font.
const MATH_SYM = /[ΣπγαβθλμνρσφψωΔΩ∞≈≤≥≠∇∑√×∈∂±∝→·]/;
function isMathCode(t: string): boolean {
  return t.includes('^') || /_\{/.test(t) || MATH_SYM.test(t) || (t.includes('=') && t.includes('_'));
}
function renderMath(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /([_^])(\{[^}]*\}|[^\s])/g;
  let last = 0;
  let key = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const inner = m[2].startsWith('{') ? m[2].slice(1, -1) : m[2];
    out.push(m[1] === '^' ? <sup key={key++}>{inner}</sup> : <sub key={key++}>{inner}</sub>);
    last = re.lastIndex;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export function Markdown({ content }: { content: string }) {
  // Heading ids must match extractToc(): same slugify + same duplicate suffixing.
  // Render-local map: ReactMarkdown invokes the component overrides within this
  // same render pass, in document order.
  const seen = new Map<string, number>();

  const headingId = (children: ReactNode) => {
    const text = flatten(children);
    let id = slugify(text);
    const n = seen.get(id) ?? 0;
    seen.set(id, n + 1);
    if (n > 0) id = `${id}-${n}`;
    return id;
  };

  return (
    <div className="prose-doc">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => <h2>{children}</h2>,
          h2: ({ children }) => <h2 id={headingId(children)}>{children}</h2>,
          h3: ({ children }) => <h3 id={headingId(children)}>{children}</h3>,
          a: ({ href, children }) => (
            <a href={href} target={href?.startsWith('http') ? '_blank' : undefined} rel="noreferrer">
              {children}
            </a>
          ),
          img: ({ src, alt }) => {
            const s = String(src ?? '');
            // Upstream lessons reference ../assets/*.svg figures that don't exist in
            // this build (no source file, no interactive equivalent). Degrade to a
            // caption from the alt text instead of showing a broken-image icon.
            const missing = !s || s.startsWith('../assets/') || /\/\.\.\/assets\//.test(s);
            if (missing) {
              return alt ? (
                <figure className="my-6 rounded-lg border border-hairline bg-bone px-4 py-3">
                  <figcaption className="font-mono text-[12px] leading-relaxed text-faint">{alt}</figcaption>
                </figure>
              ) : null;
            }
            return <img src={s} alt={alt ?? ''} loading="lazy" className="mx-auto my-6 max-w-full rounded-lg" />;
          },
          pre: ({ children }) => <>{children}</>,
          code: ({ className, children }) => {
            const text = flatten(children).replace(/\n$/, '');
            const lang = /language-(\w+)/.exec(className ?? '')?.[1];
            const isBlock = lang != null || text.includes('\n');
            if (lang === 'figure') return <LessonFigure name={text.trim()} />;
            if (!isBlock) return isMathCode(text) ? <span className="lesson-math">{renderMath(text)}</span> : <code>{text}</code>;
            if (lang === 'mermaid') return <MermaidDiagram chart={text} />;
            return <CodeBlock code={text} lang={lang ?? 'text'} />;
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
