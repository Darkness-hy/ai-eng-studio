import type { ResearchCase, ReviewFinding, ReviewFindingCode } from './types';
import { danglingClaimIds, hasExperiment, relatedWorkCount } from './completeness';

const finding = (
  code: ReviewFindingCode,
  zh: string,
  en: string,
  fixZh: string,
  fixEn: string,
): ReviewFinding => ({
  code,
  message: { zh, en },
  fixHint: { zh: fixZh, en: fixEn },
});

/** Build review findings from current case state. Empty = pass. */
export function evaluateReview(c: ResearchCase): ReviewFinding[] {
  const out: ReviewFinding[] = [];

  if (c.researchQuestion.trim().length < 20) {
    out.push(
      finding(
        'no_question',
        '研究问题过短或缺失，审稿人无法判断你在侦破什么。',
        'The research question is missing or too short.',
        '把问题写清楚（至少约 20 字），点明矛盾双方。',
        'Write a clear question (~20+ chars) naming the contradiction.',
      ),
    );
  }

  if (!c.hypothesis.text.trim() || c.hypothesis.maturity === 'wild_guess') {
    out.push(
      finding(
        'no_hypothesis',
        '假设仍停留在猜测，未操作化。',
        'The hypothesis is still a wild guess.',
        '改写假设，并在 Lab 将成熟度至少调到「可检验」。',
        'Rewrite the hypothesis and set maturity to at least operationalizable.',
      ),
    );
  }

  if (relatedWorkCount(c) < 2) {
    out.push(
      finding(
        'no_baseline',
        '相关工作/对立证据不足，缺少可对照的叙事基线。',
        'Not enough related-work or opposing evidence.',
        '至少钉上 2 条引用或现场证据，并连到 claim。',
        'Pin at least two citations/observations and link them to claims.',
      ),
    );
  }

  if (!hasExperiment(c)) {
    out.push(
      finding(
        'no_experiment',
        '没有任何验证步或实验结果挂到卷宗上。',
        'No verification or experiment is linked.',
        '在机房跑一次对照实验（或使用模拟结果）并关联。',
        'Run the control experiment (or simulation) and link it.',
      ),
    );
  }

  if (!c.resultsSummary.trim()) {
    out.push(
      finding(
        'no_results',
        '结果摘要为空。',
        'Results summary is empty.',
        '用几句话写清实验后你观察到什么。',
        'Summarize what you observed after verification.',
      ),
    );
  }

  if (c.limitations.length < 1 || !c.limitations.some((x) => x.trim())) {
    out.push(
      finding(
        'no_limitation',
        '未声明任何局限。',
        'No limitations stated.',
        '至少写一条局限（数据、模板、外推边界等）。',
        'Add at least one limitation.',
      ),
    );
  }

  const dangling = danglingClaimIds(c);
  if (dangling.length > 0) {
    out.push(
      finding(
        'dangling_claim',
        `存在 ${dangling.length} 条悬空主张（无 supports 边）。`,
        `${dangling.length} claim(s) have no supporting edges.`,
        '给每条仍 active 的 claim 连上至少一条 supports 证据，或撤回该 claim。',
        'Link at least one supporting evidence to each active claim, or retract it.',
      ),
    );
  }

  // Soft overclaim: method-magic claim still active while protocol-bias has support + experiment
  const magic = c.claims.find((cl) => cl.id === 'cl_method_a_magic' && cl.state === 'active');
  const biasSupported = c.edges.some(
    (e) => e.type === 'supports' && e.to === 'cl_protocol_bias',
  );
  if (magic && biasSupported && hasExperiment(c)) {
    out.push(
      finding(
        'overclaim',
        '在协议有偏已有证据与实验支持时，仍保留「方法甲魔法」主张，属于过度叙事。',
        'Keeping “Method A magic” while protocol bias is evidenced is an overclaim.',
        '撤回或改写「方法甲魔法」claim，使其与证据一致。',
        'Retract or rewrite the Method A magic claim to match the board.',
      ),
    );
  }

  return out;
}
