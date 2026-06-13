import { loadPlacement } from './placement';
import { streakDays, type ProgressState } from './progress';
import type { CourseIndex } from './types';

export type BadgeCategory = 'streak' | 'lessons' | 'phase' | 'placement' | 'quiz';

export interface Badge {
  key: string;
  category: BadgeCategory;
  zh: string;
  en: string;
  current: number;
  target: number;
  unlocked: boolean;
}

function tier(
  category: BadgeCategory,
  zh: string,
  en: string,
  current: number,
  target: number,
): Badge {
  return { key: `${category}-${target}`, category, zh, en, current, target, unlocked: current >= target };
}

/** Derive all badges from existing progress — no separate storage needed. */
export function computeBadges(progress: ProgressState, index: CourseIndex): Badge[] {
  // done count (catalog-based, ignores stale ids)
  let done = 0;
  let phasesComplete = 0;
  for (const p of index.phases) {
    const d = p.lessons.filter((l) => progress.lessons[`${p.slug}/${l.slug}`]?.done).length;
    done += d;
    if (p.lessons.length > 0 && d === p.lessons.length) phasesComplete += 1;
  }
  const streak = streakDays();
  const perfectQuizzes = Object.values(progress.lessons).filter(
    (l) => l.postTotal != null && l.postTotal > 0 && l.postScore === l.postTotal,
  ).length;
  const placement = loadPlacement();
  const placementDone = placement ? 1 : 0;
  const placementPerfect = placement && placement.total >= 50 ? 1 : 0;

  return [
    tier('streak', '坚持一周', '7-day streak', streak, 7),
    tier('streak', '坚持一月', '30-day streak', streak, 30),
    tier('streak', '百日筑基', '100-day streak', streak, 100),
    tier('lessons', '入门 10 课', 'First 10 lessons', done, 10),
    tier('lessons', '小成 50 课', '50 lessons', done, 50),
    tier('lessons', '百课通', '100 lessons', done, 100),
    tier('lessons', '深耕 250 课', '250 lessons', done, 250),
    tier('lessons', '全部完成', 'All lessons', done, index.stats.lessons),
    tier('phase', '首个阶段', 'First phase', phasesComplete, 1),
    tier('phase', '五阶段精通', '5 phases', phasesComplete, 5),
    tier('phase', '全阶段大师', 'All 20 phases', phasesComplete, 20),
    tier('placement', '完成定级', 'Took placement', placementDone, 1),
    tier('placement', '定级满分', 'Perfect placement', placementPerfect, 1),
    tier('quiz', '首次满分', 'First perfect quiz', perfectQuizzes, 1),
    tier('quiz', '十全十美', '10 perfect quizzes', perfectQuizzes, 10),
    tier('quiz', '五十满分', '50 perfect quizzes', perfectQuizzes, 50),
  ];
}

/** Map of YYYY-MM-DD -> lessons completed that day (for the activity heatmap). */
export function dailyCompletions(progress: ProgressState): Map<string, number> {
  const counts = new Map<string, number>();
  for (const l of Object.values(progress.lessons)) {
    if (!l.done || !l.completedAt) continue;
    const d = new Date(l.completedAt);
    const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return counts;
}
