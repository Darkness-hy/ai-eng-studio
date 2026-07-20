import type { ResearchCase } from './types';

export function buildL1Markdown(c: ResearchCase, lang: 'zh' | 'en' = 'zh'): string {
  const title = lang === 'zh' ? c.title.zh : c.title.en;
  const lim = c.limitations.filter((x) => x.trim());
  const evLines = c.evidence.map((e, i) => {
    const t = lang === 'zh' ? e.title.zh : e.title.en;
    const b = lang === 'zh' ? e.body.zh : e.body.en;
    return `${i + 1}. **[${e.kind}] ${t}** — ${b}`;
  });

  if (lang === 'en') {
    return [
      `# ${title}`,
      '',
      '## 1. Research question',
      c.researchQuestion.trim() || '_(empty)_',
      '',
      '## 2. Related work and contradictions',
      c.evidence.length
        ? c.evidence
            .filter((e) => e.kind !== 'experiment_result')
            .map((e) => `- ${e.title.en}`)
            .join('\n') || '_(none)_'
        : '_(none)_',
      '',
      '## 3. Hypothesis',
      `${c.hypothesis.text.trim() || '_(empty)_'} _(maturity: ${c.hypothesis.maturity})_`,
      '',
      '## 4. Method and verification',
      c.experimentRuns.length
        ? c.experimentRuns.map((r) => `- ${r.label}: ${r.summary}`).join('\n')
        : c.evidence
            .filter((e) => e.kind === 'experiment_result')
            .map((e) => `- ${e.title.en}`)
            .join('\n') || '_(none)_',
      '',
      '## 5. Results',
      c.resultsSummary.trim() || '_(empty)_',
      '',
      '## 6. Limitations',
      lim.length ? lim.map((x) => `- ${x}`).join('\n') : '_(none)_',
      '',
      '## 7. Appendix: evidence list',
      evLines.length ? evLines.join('\n') : '_(none)_',
      '',
      '---',
      `_L1 training report · case ${c.id} · not a peer-reviewed publication._`,
      '',
    ].join('\n');
  }

  return [
    `# ${title}`,
    '',
    '## 1. 问题',
    c.researchQuestion.trim() || '_（空）_',
    '',
    '## 2. 相关工作与矛盾',
    c.evidence.length
      ? c.evidence
          .filter((e) => e.kind !== 'experiment_result')
          .map((e) => `- ${e.title.zh}`)
          .join('\n') || '_（无）_'
      : '_（无）_',
    '',
    '## 3. 假设',
    `${c.hypothesis.text.trim() || '_（空）_'} _（成熟度：${maturityZh(c.hypothesis.maturity)}）_`,
    '',
    '## 4. 方法与验证',
    c.experimentRuns.length
      ? c.experimentRuns.map((r) => `- ${r.label}：${r.summary}`).join('\n')
      : c.evidence
          .filter((e) => e.kind === 'experiment_result')
          .map((e) => `- ${e.title.zh}`)
          .join('\n') || '_（无）_',
    '',
    '## 5. 结果',
    c.resultsSummary.trim() || '_（空）_',
    '',
    '## 6. 局限',
    lim.length ? lim.map((x) => `- ${x}`).join('\n') : '_（无）_',
    '',
    '## 7. 附录：证据清单',
    evLines.length ? evLines.join('\n') : '_（无）_',
    '',
    '---',
    `_L1 训练论文 · 案件 ${c.id} · 非同行评审发表稿_`,
    '',
  ].join('\n');
}

function maturityZh(m: ResearchCase['hypothesis']['maturity']): string {
  if (m === 'operationalized') return '可操作';
  if (m === 'plausible') return '可检验';
  return '猜测';
}
