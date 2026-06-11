import type { HighlighterCore } from 'shiki/core';

let highlighterPromise: Promise<HighlighterCore> | null = null;

function getHighlighter(): Promise<HighlighterCore> {
  if (!highlighterPromise) {
    highlighterPromise = (async () => {
      const [{ createHighlighterCore }, { createOnigurumaEngine }] = await Promise.all([
        import('shiki/core'),
        import('shiki/engine/oniguruma'),
      ]);
      return createHighlighterCore({
        themes: [import('@shikijs/themes/vitesse-light')],
        langs: [
          import('@shikijs/langs/python'),
          import('@shikijs/langs/typescript'),
          import('@shikijs/langs/javascript'),
          import('@shikijs/langs/rust'),
          import('@shikijs/langs/julia'),
          import('@shikijs/langs/json'),
          import('@shikijs/langs/bash'),
          import('@shikijs/langs/yaml'),
          import('@shikijs/langs/markdown'),
          import('@shikijs/langs/toml'),
          import('@shikijs/langs/docker'),
        ],
        engine: createOnigurumaEngine(import('shiki/wasm')),
      });
    })();
  }
  return highlighterPromise;
}

export async function highlightCode(code: string, lang: string): Promise<string> {
  const hl = await getHighlighter();
  const resolved = hl.getLoadedLanguages().includes(lang) ? lang : 'text';
  return hl.codeToHtml(code, { lang: resolved, theme: 'vitesse-light' });
}
