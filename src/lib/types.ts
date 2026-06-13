export interface LessonMetaIdx {
  slug: string;
  title: string;
  titleZh: string | null;
  type: string | null;
  time: string | null;
  languages: string[];
  hasQuiz: boolean;
  hasZh: boolean;
  runnable: boolean;
}

export interface PhaseIdx {
  slug: string;
  num: number;
  titleEn: string;
  titleZh: string;
  descZh: string;
  deps: number[];
  hours: number | null;
  lessons: LessonMetaIdx[];
}

export interface CourseIndex {
  generatedAt: string;
  stats: { phases: number; lessons: number; quizzes: number; zhLessons: number };
  phases: PhaseIdx[];
}

export interface QuizQuestion {
  stage: 'pre' | 'post' | 'check';
  question: string;
  options: string[];
  correct: number;
  explanation: string;
}

export interface CodeFile {
  name: string;
  lang: string;
  content: string;
  runnable: boolean;
}

export interface Challenge {
  titleZh: string;
  titleEn: string;
  promptZh: string;
  promptEn: string;
  starter: string;
  tests: string;
  solution: string;
}

export interface Lesson {
  id: string;
  phase: string;
  slug: string;
  title: string;
  titleZh: string | null;
  quote: string | null;
  meta: Record<string, string>;
  bodyEn: string;
  bodyZh: string | null;
  quizEn: QuizQuestion[] | null;
  quizZh: QuizQuestion[] | null;
  code: CodeFile[];
  challenge: Challenge | null;
}

export interface GlossaryTerm {
  term: string;
  saying: string;
  meaning: string;
  origin: string;
  zh?: { term: string; saying: string; meaning: string; origin: string };
}

export type Lang = 'zh' | 'en';
