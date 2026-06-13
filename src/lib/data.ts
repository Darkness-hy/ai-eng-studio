import type { CourseIndex, GlossaryTerm, Lesson } from './types';

const base = import.meta.env.BASE_URL;
const cache = new Map<string, Promise<unknown>>();

function fetchJson<T>(path: string): Promise<T> {
  if (!cache.has(path)) {
    const p = fetch(`${base}data/${path}`).then((res) => {
      if (!res.ok) throw new Error(`加载失败: ${path} (${res.status})`);
      return res.json();
    });
    // Don't cache a failed fetch — a transient network hiccup must not wedge the
    // page on a permanently-rejected promise; drop it so the next call retries.
    p.catch(() => cache.delete(path));
    cache.set(path, p);
  }
  return cache.get(path) as Promise<T>;
}

export const fetchIndex = () => fetchJson<CourseIndex>('index.json');

export const fetchLesson = (phase: string, slug: string) =>
  fetchJson<Lesson>(`lessons/${phase}/${slug}.json`);

export const fetchGlossary = () => fetchJson<{ terms: GlossaryTerm[] }>('glossary.json');
