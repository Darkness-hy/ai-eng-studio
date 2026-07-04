import { createContext, useCallback, useContext, useState, type ReactNode } from 'react';
import type { Lang, LessonMetaIdx, PhaseIdx } from './types';

const STRINGS = {
  zh: {
    nav_map: '课程地图',
    nav_glossary: '术语表',
    nav_progress: '学习进度',
    nav_leaderboards: '排行榜',
    search: '搜索',
    search_placeholder: '搜索课程、术语……',
    search_empty: '没有匹配结果',
    continue_learning: '继续学习',
    start_learning: '开始学习',
    lessons: '节课',
    phases: '个阶段',
    quizzes: '套测验',
    hours: '小时',
    phase: '阶段',
    pre_quiz: '课前热身',
    post_quiz: '课后检验',
    pre_quiz_hint: '先摸底：这几道题课文里都有答案',
    post_quiz_hint: '学完了？检验一下',
    check_answer: '查看结果',
    correct: '答对了',
    wrong: '不对',
    retry: '重做',
    score: '得分',
    mark_done: '标记完成',
    done: '已完成',
    not_translated: '本课中文翻译制作中，暂以英文原文呈现。',
    show_original: '查看英文原文',
    show_translation: '查看中文译文',
    prev_lesson: '上一课',
    next_lesson: '下一课',
    code_files: '代码实现',
    run: '运行',
    running: '运行中…',
    copy: '复制',
    copied: '已复制',
    loading: '加载中…',
    load_failed: '加载失败',
    toc: '本课目录',
    quiz_label: '测验',
    runnable_label: '可运行',
    zh_ready: '中文',
    minutes: '分钟',
    progress_title: '学习进度',
    progress_overall: '总进度',
    streak: '连续学习',
    days: '天',
    export: '导出进度',
    import: '导入进度',
    quiz_avg: '测验平均分',
    glossary_title: 'AI 术语表',
    glossary_sub: '人们怎么说，它实际是什么，以及它为什么叫这个名字。',
    g_saying: '人们怎么说',
    g_meaning: '实际含义',
    g_origin: '得名由来',
    back_to_map: '返回课程地图',
    lesson_count: (n: number) => `${n} 节课`,
    done_count: (n: number, total: number) => `已完成 ${n} / ${total}`,
  },
  en: {
    nav_map: 'Roadmap',
    nav_glossary: 'Glossary',
    nav_progress: 'Progress',
    nav_leaderboards: 'Leaderboards',
    search: 'Search',
    search_placeholder: 'Search lessons, terms…',
    search_empty: 'No results',
    continue_learning: 'Continue learning',
    start_learning: 'Start learning',
    lessons: 'lessons',
    phases: 'phases',
    quizzes: 'quizzes',
    hours: 'hours',
    phase: 'Phase',
    pre_quiz: 'Warm-up quiz',
    post_quiz: 'Check yourself',
    pre_quiz_hint: 'Pre-test: the lesson answers all of these',
    post_quiz_hint: 'Finished reading? Verify it',
    check_answer: 'Check',
    correct: 'Correct',
    wrong: 'Not quite',
    retry: 'Retry',
    score: 'Score',
    mark_done: 'Mark done',
    done: 'Done',
    not_translated: 'Chinese translation in progress; showing the English original.',
    show_original: 'Show English original',
    show_translation: 'Show Chinese translation',
    prev_lesson: 'Previous',
    next_lesson: 'Next',
    code_files: 'Implementations',
    run: 'Run',
    running: 'Running…',
    copy: 'Copy',
    copied: 'Copied',
    loading: 'Loading…',
    load_failed: 'Failed to load',
    toc: 'On this page',
    quiz_label: 'Quiz',
    runnable_label: 'Runnable',
    zh_ready: '中文',
    minutes: 'min',
    progress_title: 'Progress',
    progress_overall: 'Overall',
    streak: 'Streak',
    days: 'days',
    export: 'Export',
    import: 'Import',
    quiz_avg: 'Quiz average',
    glossary_title: 'AI Glossary',
    glossary_sub: "What people say, what it actually means, and why it's called that.",
    g_saying: 'What people say',
    g_meaning: 'What it means',
    g_origin: 'Why the name',
    back_to_map: 'Back to roadmap',
    lesson_count: (n: number) => `${n} lessons`,
    done_count: (n: number, total: number) => `${n} / ${total} done`,
  },
} as const;

export type StringKey = keyof typeof STRINGS.zh;

interface LangCtx {
  lang: Lang;
  setLang: (l: Lang) => void;
  t: <K extends StringKey>(key: K) => (typeof STRINGS.zh)[K];
}

const Ctx = createContext<LangCtx>(null!);

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(() =>
    localStorage.getItem('aes:lang') === 'en' ? 'en' : 'zh',
  );
  const setLang = useCallback((l: Lang) => {
    localStorage.setItem('aes:lang', l);
    setLangState(l);
  }, []);
  const t = useCallback(
    <K extends StringKey>(key: K) => STRINGS[lang][key] as (typeof STRINGS.zh)[K],
    [lang],
  );
  return <Ctx.Provider value={{ lang, setLang, t }}>{children}</Ctx.Provider>;
}

// eslint-disable-next-line react-refresh/only-export-components
export function useLang() {
  return useContext(Ctx);
}

// eslint-disable-next-line react-refresh/only-export-components
export function lessonTitle(l: Pick<LessonMetaIdx, 'title' | 'titleZh'>, lang: Lang): string {
  return lang === 'zh' && l.titleZh ? l.titleZh : l.title;
}

// eslint-disable-next-line react-refresh/only-export-components
export function phaseTitle(p: Pick<PhaseIdx, 'titleEn' | 'titleZh'>, lang: Lang): string {
  return lang === 'zh' ? p.titleZh : p.titleEn;
}
