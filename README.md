# ai-eng-studio · 从零开始的 AI 工程（交互式学习站）

把开源课程 [ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch)（503 课 / 20 阶段 / 338 套测验）做成中文优先、可交互学习的静态站点。设计风格基于 taste-skill 的 minimalist-skill（编辑部极简：warm monochrome + 衬线标题 + 淡彩 accent）。

## 功能

- **课程地图**：20 阶段依赖 DAG（roadmap.sh 式），每个节点显示学习进度
- **课程阅读器**：中文译文优先（未译完自动回退英文 + 标记），mermaid 图渲染、多语言代码 tabs（shiki 高亮）、章节目录滚动跟随
- **测验闭环**：课前热身（摸底）→ 正文 → 课后检验，即时判分 + 解析，成绩入进度
- **交互实验**（poloclub 式微型 explainer，嵌入对应课程）：
  - 向量游乐场（Phase 1 线性代数）— 拖动向量看点积/投影
  - 梯度下降实验台（Phase 1 优化）— 调学习率看收敛/震荡/发散
  - 自注意力热力图（Phase 7）— 真实 QKᵀ/√d + softmax 计算
  - BPE 分词器实验室（Phase 10）— 浏览器内现场训练 BPE
  - Agent 循环模拟器（Phase 14）— 一步步看 Think→Act→Observe 循环
- **浏览器内跑 Python**：纯 Python/numpy 课程代码可一键运行（Pyodide，CDN 懒加载）
- **⌘K 全局搜索**：课程（中英文标题）+ 术语表
- **术语表**：83 个术语三段式中文解释（人们怎么说 / 实际含义 / 得名由来）
- **学习进度**：localStorage 持久化（完成状态、测验得分、连续学习天数），可导出/导入 JSON
- **中英切换**：UI 与课程正文一键切换

## 运行

```bash
npm install
npm run build:content            # 从 ../ai-engineering-from-scratch 编译课程数据
npm run dev                      # http://localhost:5180
npm run build                    # 产物在 dist/，任意静态服务器可托管
```

> 注意：`scripts/build-content.mjs` 依赖同级目录的上游课程仓库 `../ai-engineering-from-scratch/`。

## 中文翻译

译文存放在 `content/zh/`（不污染上游仓库，方便上游 `git pull` 更新）：

```
content/zh/<phase-slug>/
├── titles.json             # 该阶段所有课程的中文标题
├── <lesson-slug>.md        # 课程正文中文译文
└── <lesson-slug>.quiz.json # 测验中文译文
content/zh/glossary.json    # 术语表中文译文
```

翻译完一批后重新执行 `npm run build:content` 即可生效；未翻译的课程自动回退英文原文并显示「翻译制作中」标记。

## 技术栈

Vite + React 19 + TypeScript + Tailwind CSS 4，纯静态 SPA、无后端。markdown 渲染 react-markdown + remark-gfm，代码高亮 shiki（fine-grained core），图表 mermaid，Python 运行时 Pyodide（CDN）。

## 数据来源与许可

课程内容版权归 [ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch)（MIT License, Rohit Ghumare）。本站点代码同样以 MIT 发布。
