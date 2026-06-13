/**
 * Loads the upstream lesson-figure scripts (interactive SVG widgets) and
 * hydrates ```figure``` blocks. The scripts are vanilla JS IIFEs copied from
 * the upstream site into public/figures/:
 *   - lesson-figures.js defines window.LF (+ register) and window.mountLessonFigures
 *   - figures-<topic>.js call LF.register({...}) for interactive figures (FIGS)
 *   - figures.js registers window.AIFS_FIGURES for animated explainers
 * mountLessonFigures(root) scans `.lesson-figure[data-figure]` and renders both
 * kinds, guarding against double-mount via a dataset flag.
 */

declare global {
  interface Window {
    mountLessonFigures?: (root: Document | HTMLElement) => void;
    LF?: { register: (obj: Record<string, unknown>) => void };
  }
}

const BASE = import.meta.env.BASE_URL;

// Topic modules register into FIGS (interactive) or AIFS_FIGURES (animated).
// lesson-figures.js MUST load first (the others read window.LF on eval).
const MODULES = [
  'figures.js',
  'figures-math.js',
  'figures-math2.js',
  'figures-ml.js',
  'figures-dl.js',
  'figures-nlp2.js',
  'figures-transformers.js',
  'figures-vision-speech.js',
  'figures-genai-rl.js',
  'figures-llms2.js',
  'figures-llms-systems.js',
  'figures-agents-alignment.js',
  'figures-infra.js',
  'figures-frontier.js',
];

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[data-fig="${src}"]`);
    if (existing) {
      resolve();
      return;
    }
    const s = document.createElement('script');
    s.src = src;
    s.dataset.fig = src;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error(`failed to load ${src}`));
    document.head.appendChild(s);
  });
}

let ready: Promise<void> | null = null;

export function ensureFigures(): Promise<void> {
  if (!ready) {
    ready = loadScript(`${BASE}figures/lesson-figures.js`)
      .then(() => Promise.all(MODULES.map((m) => loadScript(`${BASE}figures/${m}`))))
      .then(() => undefined);
  }
  return ready;
}

export function mountFigures(): void {
  window.mountLessonFigures?.(document);
}
