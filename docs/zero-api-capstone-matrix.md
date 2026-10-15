# Zero-API Capstone Research Matrix

## Scope

科研岛为 17 个毕业项目各实现一个受控、可测量的离线研究子任务。这里的
`$0 API cost` 指实验、分析和复现阶段不调用付费模型、向量、语音、视觉、
judge 或生成 API。CPU、存储、人工时间和未来生产验证不计入该口径。

这些结果不能证明完整产品已经达到生产可用。确定性 fixture 只适合验证有
明确 ground truth 的系统机制；真实用户效果、主观质量和生产安全仍列为外部
验证项。

## Implemented Portfolio

| # | 毕业项目 / 离线研究子任务 | 数据类型 | 主指标 | Baseline -> Candidate | 当前结论 | 仍需外部验证 |
|---:|---|---|---|---:|---|---|
| 01 | 终端编程智能体 / harness 恢复与安全 | 25 个版本化故障场景 | 安全任务完成率 | 0.2000 -> 1.0000 | supported | SWE-bench、真实 PR 质量 |
| 02 | 代码库 RAG / CodeSearchNet Python 检索 | 公开冻结快照，85 条人工查询 | nDCG@10 | 0.7940 -> 0.8302 | supported | 跨语言检索、答案生成质量 |
| 03 | 实时语音助手 / 离线 VAD 抗噪 | 24 条 seeded 波形 | VAD Frame F1 | 0.8640 -> 0.9362 | partially supported | 真实通话 WER、MOS、打断延迟 |
| 04 | 多模态文档问答 / 版面证据检索 | 24 个页面、表格和跨页场景 | Evidence nDCG@3 | 0.5436 -> 1.0000 | supported | OCR、ViDoRe、M3DocVQA、人类 QA |
| 05 | 自主科研智能体 / 有界实验搜索 | 24 个确定性目标景观 | 归一化搜索质量 | 0.9364 -> 0.9617 | partially supported | 文献质量、创新性、论文盲审 |
| 06 | Kubernetes 排障 / 根因排序 | 30 个注入故障图 | RCA Top-1 Utility | 0.5833 -> 0.6667 | partially supported | 真实集群事故、审批与回滚 |
| 07 | 端到端微调流水线 / 本地监督训练 | sklearn digits，8 个固定 split | 测试集准确率 | 0.8769 -> 0.9683 | supported | LLM 微调、GPU 吞吐、下游任务质量 |
| 08 | 合规 RAG / ACL 感知检索 | 28 个用户、文档和权限场景 | 安全检索效用 | 0.7143 -> 1.0000 | supported | 真实 IAM、越权红队、多租户延迟 |
| 09 | 代码迁移 / AST recipe | 25 个 typing 迁移 fixture | 迁移成功率 | 0.0000 -> 1.0000 | supported | 大型仓库、跨语言迁移、代码审查 |
| 10 | 多智能体协作 / 合同式调度 | 24 个依赖任务图 | 协作任务效用 | 0.3333 -> 0.8354 | supported | 真实模型协作、token 成本、人类监督 |
| 11 | LLM 可观测性 / trace 回归检测 | 60 条多信号 trace | 回归检测 F1 | 0.4000 -> 1.0000 | supported | 生产 trace、漂移、告警疲劳 |
| 12 | 视频检索 / 时序 grounding | 24 个分段序列 | Temporal IoU | 0.1917 -> 0.7436 | supported | 真实视频编码器、长视频 benchmark |
| 13 | MCP Registry / 工具治理 | 36 个 schema、scope、PII 场景 | 治理通过率 | 0.3056 -> 1.0000 | supported | 真实 MCP server、组织策略、审计 |
| 14 | 推测解码 / draft 自适应 | 17 个精确 Markov 序列 | 接受率效用 | 0.2500 -> 0.8571 | supported | 真实模型 logits、吞吐和显存 |
| 15 | Constitutional AI / 安全 gate | 36 个中英文策略场景 | 安全决策效用 | 0.5278 -> 0.8889 | supported | 专家红队、真实模型响应、误拒成本 |
| 16 | Issue-to-PR / patch sandbox | 40 个路径、测试和补丁场景 | 安全补丁效用 | 0.6250 -> 1.0000 | supported | 真实仓库 CI、维护者评审、SWE-bench |
| 17 | 自适应导师 / Bayesian Knowledge Tracing | 24 个 seeded 学习者轨迹 | 学习增益效用 | 0.5773 -> 0.6327 | supported | 学生实验、长期保持率、公平性 |

## Shared Protocol

- 每个模板固定 3 到 4 个 variant，用户不能提交代码、命令、URL、数据集或
  任意实验配置。
- 每个 job 使用 1 CPU、1 GiB、120 秒上限，容器根文件系统只读且网络禁用。
- 主指标按 case 记录；分析使用 seed `20260711` 的 1,000 次配对 Bootstrap
  和 95% 置信区间。
- 复现器重新运行全部 variant，并比较不含机器相关 latency 的确定性结果哈希。
- 负向 incremental ablation 会保留。Capstone 03 和 06 的最佳方案不是最后
  一个 variant，这一事实会显示在分析和研究卡中。
- 只有通过完整性和复现检查的研究卡可以公开；原始日志和私有产物不公开。

## Cost Boundary

运行时付费 API 成本为 `$0`。Capstone 02 的公开源码快照只在显式构建步骤访问
GitHub；实验容器本身离线。其余模板使用本地打包数据或版本化 deterministic
fixtures。未来若加入真实 LLM、ASR、VLM、embedding、judge 或云端推理，必须
单独记录调用量、模型、单价和总成本，不能继续标记为 `$0 API cost`。
