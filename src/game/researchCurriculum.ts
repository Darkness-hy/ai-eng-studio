import { fetchLesson } from '../lib/data';
import type { CourseIndex, LessonMetaIdx, PhaseIdx } from '../lib/types';
import type { TutorContext } from '../lib/tutor';

export type CurriculumLessonRef = {
  id: string;
  phaseSlug: string;
  lessonSlug: string;
  phaseTitle: string;
  title: string;
  time: string | null;
  score: number;
};

export type ResearchLearningPath = {
  id: string;
  name: string;
  description: string;
  lessonIds: string[];
};

export const researchLearningPaths: ResearchLearningPath[] = [
  {
    id: 'first-research-cycle',
    name: '第一次完整科研循环',
    description: '从问题、文献和假设开始，跑完实验、分析、写作与复现。',
    lessonIds: [
      '19-capstone-projects/50-hypothesis-generator',
      '19-capstone-projects/51-literature-retrieval',
      '19-capstone-projects/52-experiment-runner',
      '19-capstone-projects/53-result-evaluator',
      '19-capstone-projects/54-paper-writer',
      '19-capstone-projects/57-end-to-end-research-demo',
    ],
  },
  {
    id: 'experiment-foundations',
    name: '实验设计与可信评测',
    description: '补齐统计、数据、模型评估、A/B 测试和 benchmark 设计。',
    lessonIds: [
      '00-setup-and-tooling/09-data-management',
      '01-math-foundations/15-statistics-for-ml',
      '02-ml-fundamentals/09-model-evaluation',
      '02-ml-fundamentals/10-bias-variance',
      '11-llm-engineering/10-evaluation',
      '17-infrastructure-and-production/21-ab-testing-llm-features',
    ],
  },
  {
    id: 'automated-research',
    name: '自动化科研与 Agent',
    description: '理解自主科研 Agent、工具编排、评估框架和多 Agent 协作。',
    lessonIds: [
      '14-agent-engineering/28-orchestration-patterns',
      '14-agent-engineering/19-benchmarks-swebench-gaia',
      '15-autonomous-systems/05-ai-scientist-v2',
      '16-multi-agent-and-swarms/08-role-specialization',
      '16-multi-agent-and-swarms/24-evaluation-coordination-benchmarks',
      '19-capstone-projects/05-autonomous-research-agent',
    ],
  },
  {
    id: 'reproducibility-safety',
    name: '复现、治理与研究边界',
    description: '记录数据来源、系统限制、安全评测和可以公开的证据。',
    lessonIds: [
      '18-ethics-safety-alignment/12-red-teaming-pair-automated-attacks',
      '14-agent-engineering/38-verification-gates',
      '18-ethics-safety-alignment/26-model-system-dataset-cards',
      '18-ethics-safety-alignment/27-data-provenance-training-governance',
      '19-capstone-projects/27-eval-harness-fixture-tasks',
      '19-capstone-projects/75-end-to-end-eval-runner',
    ],
  },
];

const aliases: Array<{ pattern: RegExp; terms: string[] }> = [
  { pattern: /选题|问题|假设|idea|hypothesis/i, terms: ['hypothesis', 'research', '假设', '科研'] },
  { pattern: /文献|论文阅读|literature|related work/i, terms: ['literature', 'retrieval', '文献', '检索'] },
  { pattern: /实验|对照|消融|ablation|experiment/i, terms: ['experiment', 'ab testing', '实验', '评估'] },
  { pattern: /统计|置信|显著|方差|statistics/i, terms: ['statistics', 'bias variance', '统计', '评估'] },
  { pattern: /评测|指标|benchmark|metric|evaluation/i, terms: ['evaluation', 'metrics', 'benchmark', '评估'] },
  { pattern: /写作|论文|审稿|paper|writing|review/i, terms: ['paper', 'writer', '论文', 'review'] },
  { pattern: /复现|数据来源|治理|reproduc|provenance/i, terms: ['provenance', 'dataset cards', 'eval harness', '复现'] },
  { pattern: /agent|自动化科研|ai scientist/i, terms: ['agent', 'ai scientist', 'research', '自主科研'] },
];

export function lessonsForPath(index: CourseIndex, path: ResearchLearningPath): CurriculumLessonRef[] {
  const lookup = lessonLookup(index);
  return path.lessonIds.flatMap((id, order) => {
    const item = lookup.get(id);
    return item ? [{ ...item, score: path.lessonIds.length - order }] : [];
  });
}

export function recommendResearchLessons(
  index: CourseIndex,
  query: string,
  limit = 8,
): CurriculumLessonRef[] {
  const normalized = query.trim().toLowerCase();
  const terms = new Set(
    normalized
      .split(/[\s,，。；;：:、/]+/)
      .map((term) => term.trim())
      .filter((term) => term.length >= 2),
  );
  for (const alias of aliases) {
    if (alias.pattern.test(query)) alias.terms.forEach((term) => terms.add(term));
  }
  if (terms.size === 0) ['research', 'experiment', 'evaluation'].forEach((term) => terms.add(term));

  const results: CurriculumLessonRef[] = [];
  for (const phase of index.phases) {
    for (const lesson of phase.lessons) {
      const haystack = [phase.slug, phase.titleEn, phase.titleZh, lesson.slug, lesson.title, lesson.titleZh]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();
      let score = 0;
      for (const term of terms) {
        if (haystack.includes(term)) score += term.length + 2;
        if (lesson.slug.includes(term.replaceAll(' ', '-'))) score += 3;
      }
      if (phase.slug === '19-capstone-projects') score += 0.6;
      if (score > 0) results.push(toRef(phase, lesson, score));
    }
  }
  return results
    .sort((left, right) => right.score - left.score || left.id.localeCompare(right.id))
    .slice(0, limit);
}

export async function buildResearchMentorContext(
  index: CourseIndex,
  query: string,
  recommendations: CurriculumLessonRef[],
): Promise<TutorContext> {
  const lessons = await Promise.all(
    recommendations.slice(0, 3).map((item) =>
      fetchLesson(item.phaseSlug, item.lessonSlug).catch(() => null),
    ),
  );
  const phaseSummary = index.phases
    .map((phase) => `${phase.num}. ${phase.titleZh} (${phase.lessons.length} lessons)`)
    .join('\n');
  const recommendationSummary = recommendations
    .map((item) => `- ${item.title} [${item.id}]`)
    .join('\n');
  const lessonBodies = lessons
    .filter((lesson): lesson is NonNullable<typeof lesson> => Boolean(lesson))
    .map((lesson) => {
      const title = lesson.titleZh ?? lesson.title;
      const body = lesson.bodyZh ?? lesson.bodyEn;
      return `## ${title}\n${body.slice(0, 2800)}`;
    })
    .join('\n\n');
  const text = [
    '# 科研训练馆导师契约',
    '你是面向 CS/AI/Robotics 研究生的科研方法导师。回答要短、具体、可验证。',
    '先判断学习者处于选题、文献、假设、实验、分析、复现还是写作阶段。',
    '只能把课程内容作为教学依据；不能声称课程或 fixture 已证明真实科研效果。',
    '涉及启动实验、接受假设或公开产物时，只能提出建议，必须由玩家确认。',
    `用户当前问题：${query}`,
    `课程规模：${index.stats.phases} phases, ${index.stats.lessons} lessons`,
    '## 课程阶段',
    phaseSummary,
    '## 本地检索推荐',
    recommendationSummary,
    lessonBodies,
  ].join('\n\n');
  return {
    lessonId: 'research-game/research-mentor',
    title: '科研训练馆导师',
    text: text.slice(0, 12000),
  };
}

function lessonLookup(index: CourseIndex): Map<string, CurriculumLessonRef> {
  const lookup = new Map<string, CurriculumLessonRef>();
  for (const phase of index.phases) {
    for (const lesson of phase.lessons) {
      const ref = toRef(phase, lesson, 0);
      lookup.set(ref.id, ref);
    }
  }
  return lookup;
}

function toRef(phase: PhaseIdx, lesson: LessonMetaIdx, score: number): CurriculumLessonRef {
  return {
    id: `${phase.slug}/${lesson.slug}`,
    phaseSlug: phase.slug,
    lessonSlug: lesson.slug,
    phaseTitle: phase.titleZh,
    title: lesson.titleZh ?? lesson.title,
    time: lesson.time,
    score,
  };
}
