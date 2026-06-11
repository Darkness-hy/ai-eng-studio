/** Placement quiz ("find your level") — faithful port of the upstream
 *  /find-your-level agent skill: 10 questions, 5 areas, score → entry phase. */

export interface AreaDef {
  key: string;
  zh: string;
  en: string;
  /** Phases marked "Review" instead of "Skip" when this area scores 1/2. */
  reviewPhases: number[];
}

export const AREAS: AreaDef[] = [
  { key: 'math', zh: '数学与统计', en: 'Math & Statistics', reviewPhases: [1] },
  { key: 'ml', zh: '经典机器学习', en: 'Classical ML', reviewPhases: [2] },
  { key: 'dl', zh: '深度学习', en: 'Deep Learning', reviewPhases: [3] },
  { key: 'nlp', zh: 'NLP 与 Transformer', en: 'NLP & Transformers', reviewPhases: [5, 7] },
  { key: 'applied', zh: '应用 AI', en: 'Applied AI', reviewPhases: [14] },
];

export interface PlacementQuestion {
  area: string;
  zh: string;
  en: string;
  optionsZh: string[];
  optionsEn: string[];
  correct: number;
}

export const QUESTIONS: PlacementQuestion[] = [
  {
    area: 'math',
    zh: '向量 a = [1, 2, 3] 与 b = [4, 5, 6] 的点积是多少？',
    en: 'You have two vectors, a = [1, 2, 3] and b = [4, 5, 6]. What is their dot product?',
    optionsZh: ['21', '32', '15', '27'],
    optionsEn: ['21', '32', '15', '27'],
    correct: 1,
  },
  {
    area: 'math',
    zh: '一枚均匀硬币抛 3 次，恰好出现 2 次正面的概率是？',
    en: 'A fair coin is flipped 3 times. What is the probability of getting exactly 2 heads?',
    optionsZh: ['1/4', '3/8', '1/2', '1/8'],
    optionsEn: ['1/4', '3/8', '1/2', '1/8'],
    correct: 1,
  },
  {
    area: 'ml',
    zh: '分类任务中 90% 是负样本、10% 是正样本，模型把所有样本都预测为负，它的准确率是？',
    en: 'In a classification task with 90% negative and 10% positive samples, a model predicts everything as negative. What is its accuracy?',
    optionsZh: ['50%', '10%', '90%', '0%'],
    optionsEn: ['50%', '10%', '90%', '0%'],
    correct: 2,
  },
  {
    area: 'ml',
    zh: '以下哪一项是随机森林（Random Forest）的超参数？',
    en: 'Which of the following is a hyperparameter of a Random Forest?',
    optionsZh: ['学到的分裂阈值', '树的数量', '叶节点的预测值', '每个节点的 Gini 不纯度'],
    optionsEn: [
      'The learned split thresholds',
      'The number of trees',
      'The leaf node predictions',
      'The Gini impurity at each node',
    ],
    correct: 1,
  },
  {
    area: 'dl',
    zh: '反向传播过程中，链式法则计算的是什么？',
    en: 'During backpropagation, what does the chain rule compute?',
    optionsZh: [
      '最优学习率',
      '损失对每个权重的梯度',
      '需要的网络层数',
      'batch size',
    ],
    optionsEn: [
      'The optimal learning rate',
      'The gradient of the loss with respect to each weight',
      'The number of layers needed',
      'The batch size',
    ],
    correct: 1,
  },
  {
    area: 'dl',
    zh: 'ResNet 中的残差连接（skip connection）主要解决什么问题？',
    en: 'What problem do residual connections (skip connections) in ResNet primarily address?',
    optionsZh: ['小数据集上的过拟合', '深层网络的梯度消失', '数据加载太慢', '内存占用过高'],
    optionsEn: [
      'Overfitting on small datasets',
      'Vanishing gradients in deep networks',
      'Slow data loading',
      'High memory usage',
    ],
    correct: 1,
  },
  {
    area: 'nlp',
    zh: 'Transformer 架构中，注意力机制在哪些对象之间进行计算？',
    en: 'In the Transformer architecture, what does the attention mechanism compute between?',
    optionsZh: [
      '像素与标签',
      'Query、Key 和 Value',
      '仅编码器与解码器',
      '仅嵌入与位置',
    ],
    optionsEn: [
      'Pixels and labels',
      'Queries, Keys, and Values',
      'Encoder and Decoder only',
      'Embeddings and positions only',
    ],
    correct: 1,
  },
  {
    area: 'nlp',
    zh: '微调大语言模型时，LoRA（低秩适配）的主要优点是什么？',
    en: 'What is the main benefit of LoRA (Low-Rank Adaptation) when fine-tuning a large language model?',
    optionsZh: [
      '从零训练全部参数',
      '冻结大部分权重，只训练小的低秩更新矩阵',
      '不再需要任何训练数据',
      '把模型扩大一倍以获得更好效果',
    ],
    optionsEn: [
      'It trains all parameters from scratch',
      'It freezes most weights and trains small low-rank update matrices',
      'It removes the need for any training data',
      'It doubles the model size for better results',
    ],
    correct: 1,
  },
  {
    area: 'applied',
    zh: 'RAG（检索增强生成）系统中，LLM 生成回答之前会发生什么？',
    en: 'In a RAG system, what happens before the LLM generates an answer?',
    optionsZh: [
      '模型基于查询重新训练',
      '检索相关文档并注入提示词',
      '用户手动选择上下文',
      '模型搜索自己的权重',
    ],
    optionsEn: [
      'The model is retrained on the query',
      'Relevant documents are retrieved and injected into the prompt',
      'The user manually selects context',
      'The model searches its own weights',
    ],
    correct: 1,
  },
  {
    area: 'applied',
    zh: '多智能体系统中，「协调者（orchestrator）」agent 的主要作用是什么？',
    en: 'In a multi-agent system, what is the primary purpose of a "coordinator" or "orchestrator" agent?',
    optionsZh: [
      '取代所有其他 agent',
      '分配任务、路由消息并管理协作',
      '增加 token 消耗',
      '充当备用模型',
    ],
    optionsEn: [
      'To replace all other agents',
      'To assign tasks, route messages, and manage agent collaboration',
      'To increase token usage',
      'To serve as a backup model',
    ],
    correct: 1,
  },
];

/** Upstream score → entry-phase mapping. */
export function entryPhase(total: number): number {
  if (total <= 3) return 1;
  if (total <= 5) return 3;
  if (total <= 7) return 7;
  if (total <= 9) return 11;
  return 14;
}

export type PathStatus = 'skip' | 'review' | 'do';

/** Per-phase status per the upstream rules (Phase 0 always skip). */
export function phaseStatus(phaseNum: number, entry: number, areaScores: Record<string, number>): PathStatus {
  if (phaseNum === 0) return 'skip';
  if (phaseNum >= entry) return 'do';
  const reviewArea = AREAS.find(
    (a) => a.reviewPhases.includes(phaseNum) && (areaScores[a.key] ?? 0) === 1,
  );
  return reviewArea ? 'review' : 'skip';
}

export interface PlacementResult {
  v: 1;
  answers: number[];
  areaScores: Record<string, number>;
  total: number;
  entry: number;
  date: string;
}

const KEY = 'aes:placement:v1';

export function savePlacement(result: PlacementResult) {
  localStorage.setItem(KEY, JSON.stringify(result));
}

export function loadPlacement(): PlacementResult | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as PlacementResult;
    return parsed?.v === 1 ? parsed : null;
  } catch {
    return null;
  }
}

export function clearPlacement() {
  localStorage.removeItem(KEY);
}
