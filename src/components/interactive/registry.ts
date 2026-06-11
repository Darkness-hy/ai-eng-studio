import { lazy, type ComponentType, type LazyExoticComponent } from 'react';

const VectorPlayground = lazy(() =>
  import('./VectorPlayground').then((m) => ({ default: m.VectorPlayground })),
);
const GradientDescentViz = lazy(() =>
  import('./GradientDescentViz').then((m) => ({ default: m.GradientDescentViz })),
);
const AttentionHeatmap = lazy(() =>
  import('./AttentionHeatmap').then((m) => ({ default: m.AttentionHeatmap })),
);
const TokenizerLab = lazy(() => import('./TokenizerLab').then((m) => ({ default: m.TokenizerLab })));
const AgentLoopSim = lazy(() => import('./AgentLoopSim').then((m) => ({ default: m.AgentLoopSim })));

/** Signature interactive explainers, keyed by lesson id (phaseSlug/lessonSlug). */
export const INTERACTIVE: Record<string, LazyExoticComponent<ComponentType>> = {
  '01-math-foundations/01-linear-algebra-intuition': VectorPlayground,
  '01-math-foundations/02-vectors-matrices-operations': VectorPlayground,
  '01-math-foundations/08-optimization': GradientDescentViz,
  '02-ml-fundamentals/02-linear-regression': GradientDescentViz,
  '07-transformers-deep-dive/02-self-attention-from-scratch': AttentionHeatmap,
  '07-transformers-deep-dive/03-multi-head-attention': AttentionHeatmap,
  '10-llms-from-scratch/01-tokenizers': TokenizerLab,
  '10-llms-from-scratch/02-building-a-tokenizer': TokenizerLab,
  '14-agent-engineering/01-the-agent-loop': AgentLoopSim,
};
