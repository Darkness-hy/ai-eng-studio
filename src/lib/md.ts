/** Markdown helpers shared by the lesson reader and TOC. */

export function slugify(text: string): string {
  return (
    text
      .toLowerCase()
      .trim()
      .replace(/[`*_]/g, '')
      .replace(/[^\p{L}\p{N}]+/gu, '-')
      .replace(/^-+|-+$/g, '') || 'section'
  );
}

/**
 * Lesson bodies open with an H1 title, a pull-quote, and a bold meta block
 * (Type/Languages/Prerequisites/Time). The reader renders those as designed
 * header components, so strip them from the markdown before rendering.
 */
export function stripLessonHeader(md: string): string {
  const lines = md.split('\n');
  let i = 0;
  const isNoise = (line: string) =>
    /^#\s/.test(line) ||
    /^>\s?/.test(line) ||
    /^\*\*(Type|Languages|Prerequisites|Time|类型|语言|前置|时长)[:：]?\*\*/.test(line) ||
    line.trim() === '';
  while (i < lines.length && i < 14 && isNoise(lines[i])) i += 1;
  return lines.slice(i).join('\n');
}

export interface TocItem {
  depth: 2 | 3;
  text: string;
  id: string;
}

/** Extract h2/h3 headings, skipping fenced code blocks. */
export function extractToc(md: string): TocItem[] {
  const items: TocItem[] = [];
  const seen = new Map<string, number>();
  let inFence = false;
  for (const line of md.split('\n')) {
    if (/^\s*(```|~~~)/.test(line)) {
      inFence = !inFence;
      continue;
    }
    if (inFence) continue;
    const m = line.match(/^(#{2,3})\s+(.+?)\s*$/);
    if (!m) continue;
    const text = m[2].replace(/[`*_]/g, '');
    let id = slugify(text);
    const n = seen.get(id) ?? 0;
    seen.set(id, n + 1);
    if (n > 0) id = `${id}-${n}`;
    items.push({ depth: m[1].length as 2 | 3, text, id });
  }
  return items;
}
