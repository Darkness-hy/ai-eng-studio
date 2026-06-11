import { useMemo, useState } from 'react';
import { useLang } from '../../lib/i18n';

/** Embedding / head dimension of the toy model. */
const D = 16;
const MAX_TOKENS = 12;
const SEED_WQ = 0x5eed01;
const SEED_WK = 0x5eed02;
const DEFAULT_SENTENCE = 'the cat sat on the mat because it was tired';

/** Deterministic PRNG (mulberry32). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** FNV-1a hash of a string, used to seed per-token embeddings. */
function hashString(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

function randMatrix(seed: number): number[][] {
  const rng = mulberry32(seed);
  return Array.from({ length: D }, () =>
    Array.from({ length: D }, () => (rng() * 2 - 1) * 0.5),
  );
}

const W_Q: number[][] = randMatrix(SEED_WQ);
const W_K: number[][] = randMatrix(SEED_WK);

/** Seeded token embedding in [-1,1]^D plus sinusoidal positional encoding (x0.4). */
function tokenEmbedding(token: string, pos: number): number[] {
  const rng = mulberry32(hashString(token.toLowerCase()));
  const v: number[] = Array.from({ length: D }, () => rng() * 2 - 1);
  for (let i = 0; i < D; i++) {
    const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / D);
    v[i] += 0.4 * (i % 2 === 0 ? Math.sin(angle) : Math.cos(angle));
  }
  return v;
}

function matVec(w: number[][], x: number[]): number[] {
  return w.map((row) => row.reduce((s, wi, i) => s + wi * x[i], 0));
}

function dot(a: number[], b: number[]): number {
  return a.reduce((s, ai, i) => s + ai * b[i], 0);
}

function softmax(xs: number[]): number[] {
  if (xs.length === 0) return [];
  const m = Math.max(...xs);
  const exps = xs.map((x) => Math.exp(x - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/** Real (toy) self-attention: scores = QK^T / sqrt(D), softmax per row. */
function computeAttention(tokens: string[], causal: boolean): number[][] {
  const xs = tokens.map((t, i) => tokenEmbedding(t, i));
  const qs = xs.map((x) => matVec(W_Q, x));
  const ks = xs.map((x) => matVec(W_K, x));
  const scale = Math.sqrt(D);
  return qs.map((q, i) => {
    const limit = causal ? i + 1 : tokens.length;
    const probs = softmax(ks.slice(0, limit).map((k) => dot(q, k) / scale));
    const row: number[] = new Array<number>(tokens.length).fill(0);
    probs.forEach((p, j) => {
      row[j] = p;
    });
    return row;
  });
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

export function AttentionHeatmap() {
  const { lang } = useLang();
  const [text, setText] = useState<string>(DEFAULT_SENTENCE);
  const [causal, setCausal] = useState<boolean>(false);
  const [pinned, setPinned] = useState<number | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);

  const allTokens = text.split(/\s+/).filter((t) => t.length > 0);
  const truncated = allTokens.length > MAX_TOKENS;
  const tokens = allTokens.slice(0, MAX_TOKENS);
  const n = tokens.length;

  const attn = useMemo(() => computeAttention(tokens, causal), [text, causal]); // eslint-disable-line react-hooks/exhaustive-deps

  const activeRow = hovered ?? (pinned !== null && pinned < n ? pinned : null);
  const cell = Math.min(26, Math.max(18, Math.floor(336 / Math.max(n, 1))));
  const labelW = 64;

  const breakdown = useMemo(() => {
    if (activeRow === null || activeRow >= n) return [];
    return attn[activeRow]
      .map((w, j) => ({ token: tokens[j], j, w }))
      .filter((e) => e.w > 0.001)
      .sort((a, b) => b.w - a.w)
      .slice(0, 5);
  }, [attn, activeRow, n]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <section className="my-10 overflow-hidden rounded-lg border border-hairline bg-paper">
      {/* Header bar */}
      <div className="border-b border-hairline px-6 py-4">
        <div className="flex items-center gap-3">
          <span className="rounded-lg bg-pale-blue px-2 py-0.5 text-[11px] font-mono text-ink-blue">
            {lang === 'zh' ? '交互实验' : 'Interactive'}
          </span>
          <h3 className="font-serif text-[17px] text-ink">
            {lang === 'zh' ? '自注意力热力图' : 'Self-Attention Heatmap'}
          </h3>
        </div>
        <p className="mt-1 text-[12.5px] text-faint">
          {lang === 'zh'
            ? '真实的 QKᵀ/√d + softmax 计算 — 悬停或点按任意行，看每个词在“看”谁。'
            : 'A real QKᵀ/√d + softmax computation — hover or tap any row to see what each token attends to.'}
        </p>
      </div>

      <div className="px-6 py-5">
        {/* Input */}
        <label className="block text-[12px] text-faint">
          {lang === 'zh' ? '输入句子（按空格分词，最多 12 个 token）' : 'Input sentence (split on spaces, up to 12 tokens)'}
        </label>
        <input
          type="text"
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            setPinned(null);
            setHovered(null);
          }}
          spellCheck={false}
          className="mt-1.5 w-full rounded-lg border border-hairline bg-canvas px-3 py-2 font-mono text-[13px] text-ink outline-none"
        />
        {truncated && (
          <p className="mt-1 text-[11.5px] text-faint">
            {lang === 'zh'
              ? `已截断：仅使用前 ${MAX_TOKENS} 个 token。`
              : `Truncated: only the first ${MAX_TOKENS} tokens are used.`}
          </p>
        )}

        {/* Token chips */}
        {n > 0 && (
          <div className="mt-4 flex flex-wrap gap-1.5">
            {tokens.map((t, i) => (
              <button
                key={`${t}-${i}`}
                type="button"
                onMouseEnter={() => setHovered(i)}
                onMouseLeave={() => setHovered(null)}
                onClick={() => setPinned(pinned === i ? null : i)}
                className={`rounded-md px-2 py-0.5 font-mono text-[12px] transition-colors ${
                  activeRow === i ? 'bg-pale-blue text-ink-blue' : 'bg-bone text-ink'
                }`}
              >
                {t}
              </button>
            ))}
          </div>
        )}

        {/* Causal mask toggle */}
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => setCausal(!causal)}
            className={`rounded-lg border border-hairline px-3 py-1 font-mono text-[12px] transition-colors ${
              causal ? 'bg-pale-yellow text-ink-yellow' : 'bg-bone text-faint'
            }`}
          >
            {lang === 'zh' ? '因果遮罩' : 'Causal mask'} {causal ? 'ON' : 'OFF'}
          </button>
          <span className="text-[12px] text-faint">
            {causal
              ? lang === 'zh'
                ? '每个 token 只能看自己和左边的 token，权重重新归一化 — GPT 就是这样工作的。'
                : 'Each token only sees itself and tokens to its left, weights renormalized — this is how GPT works.'
              : lang === 'zh'
                ? '所有 token 互相可见（编码器式注意力）。'
                : 'Every token can attend to every token (encoder-style attention).'}
          </span>
        </div>

        {/* Heatmap */}
        {n === 0 ? (
          <p className="mt-6 text-[13px] text-faint">
            {lang === 'zh' ? '输入一句话开始实验。' : 'Type a sentence to start.'}
          </p>
        ) : (
          <div className="mt-5 overflow-x-auto" onMouseLeave={() => setHovered(null)}>
            <div
              className="inline-grid"
              style={{ gridTemplateColumns: `${labelW}px repeat(${n}, ${cell}px)` }}
            >
              {/* Corner: axis hint */}
              <div className="flex h-10 items-end justify-end pb-0.5 pr-1.5 font-mono text-[9px] text-faint">
                {'Q↓ K→'}
              </div>
              {/* Column (key) labels, rotated */}
              {tokens.map((t, j) => (
                <div key={`col-${j}`} className="relative h-10" style={{ width: cell }}>
                  <span
                    className="absolute bottom-0.5 left-1/2 origin-bottom-left whitespace-nowrap font-mono text-[9px] text-faint"
                    style={{ transform: 'rotate(-50deg)' }}
                  >
                    {truncate(t, 7)}
                  </span>
                </div>
              ))}
              {/* Rows */}
              {tokens.map((qt, i) => {
                const dim = activeRow !== null && activeRow !== i;
                return (
                  <div key={`row-${i}`} className="contents">
                    <div
                      className={`flex cursor-pointer items-center justify-end pr-1.5 font-mono text-[10px] ${
                        activeRow === i ? 'text-ink' : 'text-faint'
                      } ${dim ? 'opacity-40' : ''}`}
                      style={{ height: cell }}
                      onMouseEnter={() => setHovered(i)}
                      onClick={() => setPinned(pinned === i ? null : i)}
                    >
                      {truncate(qt, 7)}
                    </div>
                    {tokens.map((_, j) => {
                      const masked = causal && j > i;
                      const w = attn[i][j];
                      return (
                        <div
                          key={`c-${i}-${j}`}
                          className={`flex cursor-pointer items-center justify-center border border-hairline/40 font-mono text-[9px] ${
                            masked ? 'bg-bone' : ''
                          } ${dim ? 'opacity-40' : ''}`}
                          style={{
                            width: cell,
                            height: cell,
                            backgroundColor: masked ? undefined : `rgba(31,108,159,${w})`,
                            color: !masked && w > 0.55 ? '#ffffff' : undefined,
                          }}
                          onMouseEnter={() => setHovered(i)}
                          onClick={() => setPinned(pinned === i ? null : i)}
                        >
                          {!masked && cell >= 20 && (
                            <span className={w > 0.55 ? '' : 'text-faint'}>
                              {Math.round(w * 100)}
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Breakdown */}
        <div className="mt-5 rounded-lg border border-hairline bg-canvas px-4 py-3">
          {activeRow !== null && breakdown.length > 0 ? (
            <>
              <p className="font-mono text-[12px] text-ink">
                {tokens[activeRow]}
                {' → '}
                <span className="text-faint">
                  {breakdown
                    .slice(0, 3)
                    .map((e) => `${e.token} ${Math.round(e.w * 100)}%`)
                    .join(' · ')}
                </span>
              </p>
              <div className="mt-2.5 space-y-1.5">
                {breakdown.map((e) => (
                  <div key={`b-${e.j}`} className="flex items-center gap-2">
                    <span className="w-20 shrink-0 truncate text-right font-mono text-[11px] text-faint">
                      {e.token}
                    </span>
                    <div className="h-3 flex-1 rounded-sm bg-bone">
                      <div
                        className="h-3 rounded-sm"
                        style={{
                          width: `${Math.max(e.w * 100, 1.5)}%`,
                          backgroundColor: 'rgba(31,108,159,0.85)',
                        }}
                      />
                    </div>
                    <span className="w-10 shrink-0 font-mono text-[11px] text-ink">
                      {Math.round(e.w * 100)}%
                    </span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p className="text-[12px] text-faint">
              {lang === 'zh'
                ? '悬停或点按上方任意 token / 行，查看它的注意力分布。'
                : 'Hover or tap any token / row above to see its attention distribution.'}
            </p>
          )}
        </div>
      </div>

      {/* Footer strip */}
      <div className="border-t border-hairline bg-bone/50 px-6 py-3 text-[12.5px] text-faint">
        {lang === 'zh'
          ? '真实模型里同样的计算在 96 层 × 几十个头上并行发生，且 W_q/W_k 是学出来的——这里用固定随机矩阵只为展示结构。'
          : 'In a real model this same computation runs in parallel across 96 layers × dozens of heads, and W_q/W_k are learned — fixed random matrices are used here purely to show the structure.'}
      </div>
    </section>
  );
}
