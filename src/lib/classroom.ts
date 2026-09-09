import { useSyncExternalStore } from 'react';
import type { Lang } from './types';

export type Bilingual = { zh: string; en: string };

export type ClassroomWidget = 'vector' | 'attention' | 'agent-loop';

export type ClassroomScene =
  | {
      id: string;
      kind: 'lecture';
      title: Bilingual;
      bullets: Bilingual[];
      teacher: Bilingual;
      board?: Bilingual;
      peer?: Bilingual;
    }
  | {
      id: string;
      kind: 'lab';
      title: Bilingual;
      teacher: Bilingual;
      widget: ClassroomWidget;
    }
  | {
      id: string;
      kind: 'quiz';
      title: Bilingual;
      teacher: Bilingual;
      question: Bilingual;
      options: Bilingual[];
      correct: number;
      explanation: Bilingual;
    }
  | {
      id: string;
      kind: 'wrap';
      title: Bilingual;
      teacher: Bilingual;
      takeaways: Bilingual[];
    };

export interface ClassroomPack {
  lessonId: string;
  durationMin: number;
  title: Bilingual;
  subtitle: Bilingual;
  scenes: ClassroomScene[];
}

export function tx(text: Bilingual, lang: Lang): string {
  return lang === 'zh' ? text.zh : text.en;
}

const KEY = 'aes:classroom:v1';

interface ClassroomStore {
  [lessonId: string]: { completedAt: number };
}

let cache: ClassroomStore | null = null;
const listeners = new Set<() => void>();

function readStore(): ClassroomStore {
  if (cache) return cache;
  try {
    cache = JSON.parse(localStorage.getItem(KEY) ?? '{}') as ClassroomStore;
  } catch {
    cache = {};
  }
  return cache;
}

function emit(): void {
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function markClassroomComplete(lessonId: string): void {
  const next = { ...readStore(), [lessonId]: { completedAt: Date.now() } };
  cache = next;
  localStorage.setItem(KEY, JSON.stringify(next));
  emit();
}

export function classroomCompleted(lessonId: string): boolean {
  return Boolean(readStore()[lessonId]);
}

export function useClassroomCompleted(lessonId: string): boolean {
  return useSyncExternalStore(subscribe, () => classroomCompleted(lessonId), () => false);
}
