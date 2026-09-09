import type { ClassroomPack } from '../lib/classroom';

const LINALG: ClassroomPack = {
  lessonId: '01-math-foundations/01-linear-algebra-intuition',
  durationMin: 12,
  title: { zh: '线性代数直觉', en: 'Linear Algebra Intuition' },
  subtitle: {
    zh: '每个 AI 模型，本质上都是戴着帽子的矩阵运算。',
    en: 'Every AI model is matrix math wearing a fancy hat.',
  },
  scenes: [
    {
      id: 'open',
      kind: 'lecture',
      title: { zh: '为什么先学这个', en: 'Why this first' },
      bullets: [
        { zh: '词、图片、用户，在模型里都是向量。', en: 'Words, images, and users are all vectors inside a model.' },
        { zh: '权重是矩阵：把一个向量变成另一个向量。', en: 'Weights are matrices: they turn one vector into another.' },
        { zh: '检索、注意力、LoRA，用的都是同一套几何。', en: 'Retrieval, attention, and LoRA all reuse this geometry.' },
      ],
      teacher: {
        zh: '先别背公式。这堂课只建立几何直觉：向量是方向，矩阵是变换，点积是相似度。',
        en: 'Skip the memorizing. This class is geometry: vectors as directions, matrices as transforms, dot products as similarity.',
      },
    },
    {
      id: 'vectors',
      kind: 'lecture',
      title: { zh: '向量是点，也是方向', en: 'A vector is a point and a direction' },
      bullets: [
        { zh: '[3, 2] 从原点指向平面上的 (3, 2)。', en: '[3, 2] points from the origin to (3, 2).' },
        { zh: '模长 √13，方向朝右上。', en: 'Length √13, pointing up-right.' },
        { zh: '一个词可以是 768 维向量；一张图是数百万维。', en: 'A word can be 768-d; an image can be millions of pixels.' },
      ],
      board: {
        zh: '[3, 2]  →  点 (3, 2)\n|v| = √(3² + 2²) = √13',
        en: '[3, 2]  →  point (3, 2)\n|v| = √(3² + 2²) = √13',
      },
      teacher: {
        zh: '数字有坐标含义。嵌入空间里，相近的意思会靠在一起。',
        en: 'The numbers are coordinates. Nearby meanings sit nearby in embedding space.',
      },
    },
    {
      id: 'dot',
      kind: 'lecture',
      title: { zh: '点积衡量相似度', en: 'Dot product measures similarity' },
      bullets: [
        { zh: '同向：a·b > 0，相似。', en: 'Same direction: a·b > 0, similar.' },
        { zh: '垂直：a·b = 0，无关。', en: 'Perpendicular: a·b = 0, unrelated.' },
        { zh: '反向：a·b < 0，不相似。', en: 'Opposite: a·b < 0, dissimilar.' },
      ],
      board: {
        zh: 'a · b = a₁b₁ + a₂b₂ + … + aₙbₙ',
        en: 'a · b = a₁b₁ + a₂b₂ + … + aₙbₙ',
      },
      peer: {
        zh: '所以 RAG 就是把问题和文档都变成向量，再算点积？',
        en: 'So RAG is just embedding the question and the docs, then taking dot products?',
      },
      teacher: {
        zh: '对。搜索、推荐、RAG 都是在找点积高的向量。下一页你可以亲手拖。',
        en: 'Yes. Search, recommenders, and RAG all hunt for high dot products. Next, you can drag them yourself.',
      },
    },
    {
      id: 'lab',
      kind: 'lab',
      title: { zh: '动手：拖动两个向量', en: 'Lab: drag two vectors' },
      widget: 'vector',
      teacher: {
        zh: '拖箭头端点。看 a·b 的符号怎么变，投影落在哪条边上。',
        en: 'Drag the tips. Watch the sign of a·b, and where the projection lands.',
      },
    },
    {
      id: 'matrix',
      kind: 'lecture',
      title: { zh: '矩阵是变换', en: 'A matrix is a transform' },
      bullets: [
        { zh: '矩阵把向量旋转、缩放、拉伸或投影。', en: 'A matrix rotates, scales, stretches, or projects a vector.' },
        { zh: '神经网络的权重，就是一层层这样的变换。', en: 'Neural-net weights are stacked transforms of this kind.' },
        { zh: '注意力分数本身也是一张矩阵。', en: 'Attention scores are a matrix too.' },
      ],
      teacher: {
        zh: '模型不是魔法。输入向量乘上权重矩阵，得到新的表示。',
        en: 'No magic: multiply the input vector by a weight matrix, get a new representation.',
      },
    },
    {
      id: 'rank',
      kind: 'lecture',
      title: { zh: '线性无关、基与秩', en: 'Independence, basis, and rank' },
      bullets: [
        { zh: '若某个向量是其余向量的组合，整组就线性相关。', en: 'If one vector is a mix of the others, the set is dependent.' },
        { zh: 'v3 = [2,1,0] = 2·v1 + v2，三个向量仍困在 xy 平面。', en: 'v3 = [2,1,0] = 2·v1 + v2, so all three stay in the xy-plane.' },
        { zh: '秩 = 线性无关的列数 = 有用信息的维数。', en: 'Rank = number of independent columns = useful dimensions.' },
      ],
      board: {
        zh: 'v1 = [1, 0, 0]\nv2 = [0, 1, 0]\nv3 = [2, 1, 0]   # v3 = 2 v1 + v2',
        en: 'v1 = [1, 0, 0]\nv2 = [0, 1, 0]\nv3 = [2, 1, 0]   # v3 = 2 v1 + v2',
      },
      teacher: {
        zh: '特征冗余会让权重没有唯一解。LoRA 反过来利用这件事：更新往往落在低维子空间。',
        en: 'Redundant features make weights non-unique. LoRA exploits the opposite: updates often live in a low-dimensional subspace.',
      },
    },
    {
      id: 'q-indep',
      kind: 'quiz',
      title: { zh: '课堂练习', en: 'In-class check' },
      teacher: {
        zh: '用刚才那三个向量想一想。答错也没关系，这页不计分。',
        en: 'Use those three vectors. Wrong is fine — this page does not count toward your score.',
      },
      question: {
        zh: 'v1=[1,0,0]、v2=[0,1,0]、v3=[2,1,0]，它们线性无关吗？',
        en: 'Are v1=[1,0,0], v2=[0,1,0], v3=[2,1,0] linearly independent?',
      },
      options: [
        { zh: '无关，因为三维空间中有三个向量', en: 'Independent, because there are three vectors in 3D' },
        { zh: '相关，因为 v3 = 2·v1 + v2', en: 'Dependent, because v3 = 2·v1 + v2' },
        { zh: '无关，因为没有任何两个向量完全相同', en: 'Independent, because no two vectors are identical' },
        { zh: '相关，因为它们都含有零分量', en: 'Dependent, because they contain zeros' },
      ],
      correct: 1,
      explanation: {
        zh: 'v3 是其余两个的组合，三个向量都在 xy 平面，张不成 [0,0,1]。',
        en: 'v3 is a combination of the other two; all three live in the xy-plane and cannot reach [0,0,1].',
      },
    },
    {
      id: 'q-lora',
      kind: 'quiz',
      title: { zh: '课堂练习', en: 'In-class check' },
      teacher: {
        zh: '把「低秩」连回微调。',
        en: 'Connect low rank back to fine-tuning.',
      },
      question: {
        zh: 'LoRA 如何利用线性代数高效微调大模型？',
        en: 'How does LoRA use linear algebra to fine-tune large models cheaply?',
      },
      options: [
        { zh: '删除权重矩阵中未使用的行', en: 'Delete unused rows from the weight matrix' },
        { zh: '把权重更新分解成两个小的低秩矩阵', en: 'Factor the weight update into two small low-rank matrices' },
        { zh: '把所有权重从 float32 转为 int8', en: 'Cast every weight from float32 to int8' },
        { zh: '冻结嵌入层，只训练输出头', en: 'Freeze embeddings and train only the output head' },
      ],
      correct: 1,
      explanation: {
        zh: '例如把 4096×4096 的更新写成 4096×16 与 16×4096，假设更新落在低维子空间。',
        en: 'A 4096×4096 update becomes 4096×16 and 16×4096, assuming the update lives in a low-dimensional subspace.',
      },
    },
    {
      id: 'wrap',
      kind: 'wrap',
      title: { zh: '带走这三句', en: 'Take these three' },
      takeaways: [
        { zh: '向量是空间中的方向；嵌入把事物放进这个空间。', en: 'Vectors are directions; embeddings place things in that space.' },
        { zh: '点积是相似度；矩阵是变换。', en: 'Dot product is similarity; a matrix is a transform.' },
        { zh: '秩告诉你有多少维真正有用——LoRA 就是在用低秩。', en: 'Rank is how many dimensions actually matter — LoRA is low-rank on purpose.' },
      ],
      teacher: {
        zh: '课文里还有投影、Gram-Schmidt 和代码。互动课只过几何，细节回正文。',
        en: 'The lesson still has projection, Gram-Schmidt, and the code. This class was the geometry; details live in the text.',
      },
    },
  ],
};

const ATTENTION: ClassroomPack = {
  lessonId: '07-transformers-deep-dive/02-self-attention-from-scratch',
  durationMin: 14,
  title: { zh: '从零实现自注意力', en: 'Self-Attention from Scratch' },
  subtitle: {
    zh: '每个词都在问：谁对我重要？然后学会答案。',
    en: 'Every word asks "who matters to me?" — and learns the answer.',
  },
  scenes: [
    {
      id: 'open',
      kind: 'lecture',
      title: { zh: '软查找，不是精确匹配', en: 'Soft lookup, not exact match' },
      bullets: [
        { zh: '传统数据库：query 精确命中一条记录。', en: 'A database: the query hits one exact record.' },
        { zh: '注意力：query 和所有 key 比相似度，再混合所有 value。', en: 'Attention: compare the query to every key, then blend every value.' },
        { zh: 'Q = 我在找什么；K = 我包含什么；V = 被选中时我输出什么。', en: 'Q = what I seek; K = what I contain; V = what I emit if chosen.' },
      ],
      peer: {
        zh: '那不就是模糊搜索吗？',
        en: "Isn't that just fuzzy search?",
      },
      teacher: {
        zh: '更进一步：每个词同时问所有词，答案是加权混合，不是只取第一名。',
        en: 'One step further: every token queries every token. The answer is a weighted blend, not the top hit.',
      },
    },
    {
      id: 'qkv',
      kind: 'lecture',
      title: { zh: 'Q、K、V 从哪来', en: 'Where Q, K, V come from' },
      bullets: [
        { zh: '输入 X 形状 (n, d)：n 个 token，每个 d 维。', en: 'Input X has shape (n, d): n tokens, d dimensions each.' },
        { zh: '三个可学习矩阵 Wq、Wk、Wv。', en: 'Three learned matrices: Wq, Wk, Wv.' },
        { zh: 'Q = X Wq，K = X Wk，V = X Wv。', en: 'Q = X Wq, K = X Wk, V = X Wv.' },
      ],
      board: {
        zh: 'X  ─Wq→  Q   「我在找什么？」\nX  ─Wk→  K   「我包含什么？」\nX  ─Wv→  V   「我能提供什么？」',
        en: 'X  ─Wq→  Q   "What am I looking for?"\nX  ─Wk→  K   "What do I contain?"\nX  ─Wv→  V   "What do I offer?"',
      },
      teacher: {
        zh: '同一段输入，三条不同的线性投影。自注意力的「自」就在这里：Q、K、V 都来自 X。',
        en: 'One input, three different linear maps. That is the "self": Q, K, and V all come from X.',
      },
    },
    {
      id: 'scores',
      kind: 'lecture',
      title: { zh: '注意力矩阵', en: 'The attention matrix' },
      bullets: [
        { zh: 'Scores = Q Kᵀ，形状 (n, n)。', en: 'Scores = Q Kᵀ, shape (n, n).' },
        { zh: '第 i 行：token i 对整句每一个词的打分。', en: 'Row i: how much token i attends to every word.' },
        { zh: '所以时间和显存都是 O(n²)。', en: 'Hence time and memory are both O(n²).' },
      ],
      board: {
        zh: '              k1   k2   k3   k4\n        q1  [ 2.1  0.3  0.1  0.8 ]\n        q2  [ 0.4  1.9  0.7  0.1 ]',
        en: '              k1   k2   k3   k4\n        q1  [ 2.1  0.3  0.1  0.8 ]\n        q2  [ 0.4  1.9  0.7  0.1 ]',
      },
      teacher: {
        zh: '长上下文贵，就是因为这张 n×n 表。FlashAttention 优化的是它的实现，不是取消这张表。',
        en: 'Long context is expensive because of this n×n table. FlashAttention speeds it up; it does not remove it.',
      },
    },
    {
      id: 'scale',
      kind: 'lecture',
      title: { zh: '为什么除以 √d_k', en: 'Why divide by √d_k' },
      bullets: [
        { zh: '点积的大小随 √d_k 增长。', en: 'Dot-product magnitude grows with √d_k.' },
        { zh: 'd_k=64 时，原始分数可到几十，softmax 接近 one-hot。', en: 'At d_k=64, raw scores can be tens, and softmax goes nearly one-hot.' },
        { zh: '梯度几乎消失，模型学不动。', en: 'Gradients vanish; the model stops learning.' },
      ],
      board: {
        zh: 'Attention(Q,K,V) = softmax( Q Kᵀ / √d_k ) V',
        en: 'Attention(Q,K,V) = softmax( Q Kᵀ / √d_k ) V',
      },
      teacher: {
        zh: '缩放不是为了把输出压到 0 和 1，而是让 softmax 待在还有梯度的区域。',
        en: 'The scale is not to squash outputs into 0–1. It keeps softmax in a region that still has gradient.',
      },
    },
    {
      id: 'lab',
      kind: 'lab',
      title: { zh: '动手：看每个词在看谁', en: 'Lab: see who each word watches' },
      widget: 'attention',
      teacher: {
        zh: '改句子，点一行。热力图就是 softmax(QKᵀ/√d)。也可以打开因果掩码。',
        en: 'Edit the sentence, tap a row. The heatmap is softmax(QKᵀ/√d). Try the causal mask too.',
      },
    },
    {
      id: 'mix',
      kind: 'lecture',
      title: { zh: '权重乘上 Value', en: 'Weights times values' },
      bullets: [
        { zh: 'softmax 把一行分数变成和为 1 的权重。', en: 'Softmax turns a row of scores into weights that sum to 1.' },
        { zh: '输出 = Σ 权重ⱼ · vⱼ。', en: 'Output = Σ weightⱼ · vⱼ.' },
        { zh: '因果掩码：把未来位置设为 −∞，softmax 后权重为 0。', en: 'Causal mask: set future positions to −∞ so their weights become 0.' },
      ],
      teacher: {
        zh: '生成下一个 token 时，位置 t 不能偷看 t+1。掩码保证这一点。',
        en: 'When generating token t, it must not peek at t+1. The mask is that guarantee.',
      },
    },
    {
      id: 'q-scale',
      kind: 'quiz',
      title: { zh: '课堂练习', en: 'In-class check' },
      teacher: {
        zh: '这题课文课前热身也有。这里只练，不计分。',
        en: 'This is also in the lesson warm-up. Practice only — it does not count.',
      },
      question: {
        zh: '为什么要把点积乘以 1/√d_k？',
        en: 'Why scale the dot product by 1/√d_k?',
      },
      options: [
        { zh: '为了让输出落在 0 到 1 之间', en: 'To force outputs into 0–1' },
        { zh: '防止高维点积过大，把 softmax 推进梯度极小区', en: 'To stop large-d_k dots from pushing softmax into tiny gradients' },
        { zh: '把 Q 和 K 归一成单位长度', en: 'To normalize Q and K to unit length' },
        { zh: '降低矩阵乘法的计算量', en: 'To make the matmul cheaper' },
      ],
      correct: 1,
      explanation: {
        zh: '不缩放时 softmax 接近 one-hot，梯度消失。除以 √d_k 是为了稳住数值。',
        en: 'Without scaling, softmax becomes nearly one-hot and gradients vanish. √d_k keeps the numbers sane.',
      },
    },
    {
      id: 'q-quad',
      kind: 'quiz',
      title: { zh: '课堂练习', en: 'In-class check' },
      teacher: {
        zh: '再确认一次复杂度从哪来。',
        en: 'One more: where the quadratic cost comes from.',
      },
      question: {
        zh: '为什么自注意力对序列长度 n 是 O(n²)？',
        en: 'Why is self-attention O(n²) in sequence length n?',
      },
      options: [
        { zh: '因为堆了 n 层', en: 'Because the model stacks n layers' },
        { zh: '因为每个 token 都要和所有 token 打分，得到 n×n 矩阵', en: 'Because every token scores every token, making an n×n matrix' },
        { zh: '因为后面的前馈层是二次的', en: 'Because the feed-forward layer is quadratic' },
        { zh: '因为反向传播要算 n² 次梯度', en: 'Because backprop computes n² gradients' },
      ],
      correct: 1,
      explanation: {
        zh: 'QKᵀ 就是那张 n×n 表。计算和内存都按 n² 涨。',
        en: 'QKᵀ is that n×n table. Both compute and memory grow as n².',
      },
    },
    {
      id: 'wrap',
      kind: 'wrap',
      title: { zh: '带走这三句', en: 'Take these three' },
      takeaways: [
        { zh: '自注意力 = 每个 token 对全句做一次软查找。', en: 'Self-attention is a soft lookup from every token over the whole sentence.' },
        { zh: '公式：softmax(QKᵀ/√d_k) V。', en: 'The formula: softmax(QKᵀ/√d_k) V.' },
        { zh: 'n×n 矩阵带来 O(n²)；因果掩码挡住未来。', en: 'The n×n matrix is why it is O(n²); the causal mask hides the future.' },
      ],
      teacher: {
        zh: '多头、从零实现 softmax、ASCII 热力图，都在课文里。这里只把几何跑通。',
        en: 'Multi-head, softmax from scratch, and the ASCII heatmap live in the lesson text. This class was the geometry.',
      },
    },
  ],
};

const AGENT: ClassroomPack = {
  lessonId: '14-agent-engineering/01-the-agent-loop',
  durationMin: 12,
  title: { zh: 'Agent 循环：观察、思考、行动', en: 'The Agent Loop: Observe, Think, Act' },
  subtitle: {
    zh: '2026 年每个智能体框架，底层仍是 2022 年的 ReAct。',
    en: 'Every 2026 agent framework is still the 2022 ReAct loop underneath.',
  },
  scenes: [
    {
      id: 'open',
      kind: 'lecture',
      title: { zh: '补全器不是智能体', en: 'A completer is not an agent' },
      bullets: [
        { zh: '单独的 LLM 只能根据权重往外吐字。', en: 'A bare LLM can only emit tokens from its weights.' },
        { zh: '它读不了文件，也没法对照外部世界验证。', en: 'It cannot read a file or check the outside world.' },
        { zh: '有循环、有工具，才从聊天变成智能体。', en: 'A loop plus tools is what turns chat into an agent.' },
      ],
      teacher: {
        zh: '记住分界：不能观察、不能行动的模型，只是自动补全。',
        en: 'The line is simple: if it cannot observe or act, it is autocomplete.',
      },
    },
    {
      id: 'react',
      kind: 'lecture',
      title: { zh: 'ReAct 的三行', en: 'The three ReAct lines' },
      bullets: [
        { zh: 'Thought：我打算做什么、为什么。', en: 'Thought: what I intend, and why.' },
        { zh: 'Action：调用哪个工具、参数是什么。', en: 'Action: which tool, with which arguments.' },
        { zh: 'Observation：工具返回了什么。', en: 'Observation: what the tool returned.' },
      ],
      board: {
        zh: 'Thought: I need the capital of France.\nAction: search("capital of France")\nObservation: Paris is the capital of France.\nThought: The answer is Paris.\nAction: finish("Paris")',
        en: 'Thought: I need the capital of France.\nAction: search("capital of France")\nObservation: Paris is the capital of France.\nThought: The answer is Paris.\nAction: finish("Paris")',
      },
      teacher: {
        zh: 'Yao 等人 2022 年的格式。推理轨迹让模型能做计划、跟踪计划，并在意外观察时改主意。',
        en: 'Yao et al., 2022. The trace lets the model plan, track the plan, and recover when an observation is surprising.',
      },
    },
    {
      id: 'native',
      kind: 'lecture',
      title: { zh: '2026：思考进了独立通道', en: '2026: thinking got its own channel' },
      bullets: [
        { zh: '当年用提示词逼模型写出 Thought:。', en: 'We used to prompt the model to print Thought:.' },
        { zh: '现在推理内容走独立通道，跨轮次透传。', en: 'Reasoning now travels in a dedicated channel, carried across turns.' },
        { zh: '控制流没变：观察 → 思考 → 行动 → 再观察。', en: 'The control flow did not change: observe → think → act → observe.' },
      ],
      teacher: {
        zh: '包装变了，循环没变。别被 Responses API 的新字段晃过去。',
        en: 'The packaging changed; the loop did not. Do not get distracted by new Response API fields.',
      },
    },
    {
      id: 'five',
      kind: 'lecture',
      title: { zh: '五大要素，缺一不可', en: 'Five parts, all required' },
      bullets: [
        { zh: '消息缓冲区：用户、助手、工具，轮次往上堆。', en: 'Message buffer: user, assistant, tool turns stacked in order.' },
        { zh: '工具注册表：按名字调用，schema 进、字符串出。', en: 'Tool registry: call by name, schema in, string out.' },
        { zh: '停止条件 + 轮次预算 + 观察格式化器。', en: 'Stop condition + turn budget + observation formatter.' },
      ],
      peer: {
        zh: '没有轮次预算会怎样？',
        en: 'What if there is no turn budget?',
      },
      teacher: {
        zh: '循环可能不停。2026 年的智能体常跑 40 到 400 步，上限要按任务设，不能一刀切。',
        en: 'The loop may never halt. 2026 agents often run 40–400 steps; the cap should match the task, not a single global number.',
      },
    },
    {
      id: 'lab',
      kind: 'lab',
      title: { zh: '动手：看三轮循环', en: 'Lab: watch three turns' },
      widget: 'agent-loop',
      teacher: {
        zh: '点播放。注意每一轮都是思考 → 行动 → 观察 → 是否该停。',
        en: 'Hit play. Each turn is think → act → observe → should we stop?',
      },
    },
    {
      id: 'invariant',
      kind: 'lecture',
      title: { zh: '框架换皮，循环不换', en: 'Frameworks change; the loop does not' },
      bullets: [
        { zh: 'Claude Agent SDK、OpenAI Agents、LangGraph、AutoGen、CrewAI…', en: 'Claude Agent SDK, OpenAI Agents, LangGraph, AutoGen, CrewAI…' },
        { zh: '差异在检查点、Actor、角色模板、追踪。', en: 'They differ in checkpoints, actors, role templates, tracing.' },
        { zh: '底下跑的仍是 ReAct。', en: 'Underneath it is still ReAct.' },
      ],
      teacher: {
        zh: '学循环本身，比追框架名词更值钱。',
        en: 'Learning the loop pays off more than chasing framework names.',
      },
    },
    {
      id: 'trap',
      kind: 'lecture',
      title: { zh: '工具输出不可信', en: 'Tool output is untrusted' },
      bullets: [
        { zh: '网上检索到的 PDF 可能夹带「删掉仓库」之类指令。', en: 'A fetched PDF may hide an instruction like "delete the repo".' },
        { zh: '只有用户亲口说的才算授权。', en: "Only the user's direct words count as authorization." },
        { zh: '智能体还常把 400 错误幻觉成成功。', en: 'Agents also hallucinate success out of HTTP 400s.' },
      ],
      peer: {
        zh: '工具返回的内容也能当指令用吗？',
        en: "Can the tool's text be treated as an instruction?",
      },
      teacher: {
        zh: '不能。把它当成不可信输入。这是本课点名的 2026 陷阱。',
        en: 'No. Treat it as untrusted input. That is the 2026 trap this lesson names.',
      },
    },
    {
      id: 'q-five',
      kind: 'quiz',
      title: { zh: '课堂练习', en: 'In-class check' },
      teacher: {
        zh: '五个要素里，哪一项是混进来的？不计分。',
        en: 'Which of these is not one of the five parts? Practice only.',
      },
      question: {
        zh: '下列哪一项不属于智能体循环的五大要素？',
        en: 'Which of the following is not one of the five agent-loop ingredients?',
      },
      options: [
        { zh: '消息缓冲区', en: 'Message buffer' },
        { zh: '工具注册表', en: 'Tool registry' },
        { zh: '观察结果格式化器', en: 'Observation formatter' },
        { zh: '梯度优化器', en: 'Gradient optimizer' },
      ],
      correct: 3,
      explanation: {
        zh: '五大要素是消息缓冲、工具注册表、停止条件、轮次预算、观察格式化器。优化器属于训练，不属于推理循环。',
        en: 'The five are message buffer, tool registry, stop condition, turn budget, and observation formatter. An optimizer belongs to training, not the inference loop.',
      },
    },
    {
      id: 'q-trust',
      kind: 'quiz',
      title: { zh: '课堂练习', en: 'In-class check' },
      teacher: {
        zh: '最后一问：信任边界。',
        en: 'Last question: the trust boundary.',
      },
      question: {
        zh: '为什么说工具输出是不可信输入？',
        en: 'Why is tool output treated as untrusted input?',
      },
      options: [
        { zh: '工具运行时又慢又不可靠', en: 'Tools are slow and unreliable' },
        { zh: '检索内容可能携带隐藏指令，只有用户直接输入才算授权', en: 'Retrieved text may hide instructions; only the user authorizes' },
        { zh: '工具结果总是大于上下文窗口', en: 'Tool results always overflow the context window' },
        { zh: '提供商默认会丢弃工具输出', en: 'Providers drop tool output by default' },
      ],
      correct: 1,
      explanation: {
        zh: 'OpenAI CUA 写得很清楚：只有用户的直接指令才算授权。',
        en: "OpenAI's CUA docs are explicit: only the user's direct instruction is authorization.",
      },
    },
    {
      id: 'wrap',
      kind: 'wrap',
      title: { zh: '带走这三句', en: 'Take these three' },
      takeaways: [
        { zh: '智能体 = LLM + 工具 + 循环，不是更会聊天的模型。', en: 'An agent is an LLM + tools + a loop, not a chattier model.' },
        { zh: '观察-思考-行动是不变量；框架只是外壳。', en: 'Observe-think-act is the invariant; frameworks are shells.' },
        { zh: '轮次要有预算；工具输出不可信。', en: 'Budget the turns; never trust tool output.' },
      ],
      teacher: {
        zh: '从零实现和生产实践在课文里。互动课只把循环和边界讲清楚。',
        en: 'The from-scratch build and production notes are in the lesson. This class was the loop and the boundary.',
      },
    },
  ],
};

export const CLASSROOM_PACKS: Record<string, ClassroomPack> = {
  [LINALG.lessonId]: LINALG,
  [ATTENTION.lessonId]: ATTENTION,
  [AGENT.lessonId]: AGENT,
};

export function classroomPack(lessonId: string): ClassroomPack | undefined {
  return CLASSROOM_PACKS[lessonId];
}
