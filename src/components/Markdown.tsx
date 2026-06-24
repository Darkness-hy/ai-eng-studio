import { Children, type ReactNode } from 'react';
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
function renderMath(text: string, bracedOnly = false): ReactNode[] {
  const out: ReactNode[] = [];
  // bracedOnly (for prose) only lifts X^{..}/X_{..}; full mode also lifts single
  // chars (X^2, V_n) — safe inside spans/blocks already known to be math.
  const re = bracedOnly ? /([_^])(\{[^}]*\})/g : /([_^])(\{[^}]*\}|[^\s])/g;
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

// An unlabeled ``` fence whose content is a display formula (not real code or
// ASCII art). Conservative: bail on code syntax, require an '=' plus a math cue.
function isMathBlock(text: string): boolean {
  // shell prompts, URLs, paths, CLI flags, arrows, box-drawing → not a formula
  if (/(^|\n)\s*\$\s/.test(text)) return false;
  if (/https?:\/\/|www\.|curl |wget |\.sh\b|\.py\b|\.json\b|\.ya?ml\b|\s--[a-zA-Z]|->|[│├└─┌┐┘┬┴┼╮╯╰╭]/.test(text)) return false;
  // code syntax
  if (/;|=>|\bdef\b|\bimport\b|\breturn\b|\bprint\s*\(|\bfunction\b|\bconst\b|\blet\b|\bvar\b|#include|<\/|::/.test(text)) {
    return false;
  }
  if (!text.includes('=')) return false;
  // require a real math cue (not just a bare underscore, which is common in code)
  return /\^|_\{|\|\|[^|]|\bsqrt\(|\bsum\(|[ΣπγαβλμθρσφψωΔΩ∞≈≤≥≠∇∑√×∈∂±]/.test(text);
}

// Display math block: monospace `<pre>` (keeps multi-line alignment) with ^/_
// raised/lowered, minus the code-block chrome.
function MathBlock({ text }: { text: string }) {
  const lines = text.split('\n');
  return (
    <pre className="lesson-math-block">
      {lines.map((ln, i) => (
        <span key={i}>
          {renderMath(ln)}
          {i < lines.length - 1 ? '\n' : ''}
        </span>
      ))}
    </pre>
  );
}

// Lift braced sub/superscripts that appear in plain prose (rare: x_{t-1}, s_{i+1}).
function mathifyProse(children: ReactNode): ReactNode {
  return Children.map(children, (c) =>
    typeof c === 'string' && /[_^]\{/.test(c) ? <>{renderMath(c, true)}</> : c,
  );
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
          p: ({ children }) => <p>{mathifyProse(children)}</p>,
          li: ({ children }) => <li>{mathifyProse(children)}</li>,
          img: ({ src, alt }) => {
            const s = String(src ?? '');
            // Upstream lessons reference ../assets/*.svg figures that don't exist in
            // this build (no source file, no interactive equivalent). Degrade to a
            // caption from the alt text instead of showing a broken-image icon.
            const missing = !s || s.startsWith('../assets/') || /\/\.\.\/assets\//.test(s);
            if (missing) {
              // markdown images sit inside a <p>, so use a phrasing element
              // (block-styled <span>) — a <figure>/<div> here is invalid nesting.
              return alt ? (
                <span className="my-4 block rounded-lg border border-hairline bg-bone px-4 py-3 font-mono text-[12px] leading-relaxed text-faint">
                  {alt}
                </span>
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
            if (lang == null && isMathBlock(text)) return <MathBlock text={text} />;
            return <CodeBlock code={text} lang={lang ?? 'text'} />;
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
