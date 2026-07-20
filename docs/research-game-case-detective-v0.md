# 科研岛玩法 v0 · 案件探案（设计执行稿）

> 状态：已对齐默认选择，可进入实现排期  
> 日期：2026-07-20  
> 仓库：`ai-eng-studio`（规范副本 `~/code/learn-ai/ai-eng-studio`）  
> 原则：机制先于画质；单人；高密度小岛；L1 必达；导师+审稿；社交后置

---

## 0. 锁定决策（不可在实现中悄悄改掉）

| ID | 决策 |
|---|---|
| D1 | 主爽点 = **探案感**（侦查 / 线索 / 排除假设 / 结案） |
| D2 | 主路径保证 **L1 训练论文**；L2/真投稿不在 v0 |
| D3 | 世界 = **一座高密度小岛**，不铺群岛 |
| D4 | NPC 仅 **导师 + 审稿人**（不做同学 NPC / 恋爱） |
| D5 | **单人为主**；社交仅可选导出后参观/短码，不挡主线 |
| D6 | 探索掉落必须改 **同一份卷宗（Case File）**，禁止「逛完再开表单」 |

产品一句话：

> 在厚实小岛上独自侦破科研案件；导师开门、审稿施压；结案卷宗 = L1 训练论文。

---

## 1. 玩家幻想与胜利条件

### 1.1 幻想
你是见习研究员。岛上通报一起**可研究的异常**。你查现场、问导师、跑一次可信实验，在 Lab 里把卷宗钉完整，挺过审稿漏洞检查，导出结案报告。

### 1.2 胜利（L1 结案）
系统全部可检查（规则引擎，不靠玄学分）：

| 字段 | 最低要求（v0 数值可调） |
|---|---|
| `researchQuestion` | 非空，≥ 20 字 |
| `relatedWork` | ≥ 2 条已钉证据，且类型含 `citation` 或 `testimony` |
| `hypothesis` | 非空，且状态 ≠ `wild_guess` |
| `methodOrExperiment` | ≥ 1 次 `experiment_result` 已关联，或等价 `verification` |
| `results` | 非空摘要 |
| `limitations` | ≥ 1 条 |
| `danglingClaims` | 0（每条 claim 至少 1 条 support 边） |
| `reviewPasses` | ≥ 1 次审稿轮次 `status=passed` |

导出物：`L1 Report`（Markdown/结构化 JSON → 前端可读页 + 可复制文本）。  
**不承诺**期刊接收。

### 1.3 失败与重试
- 无 Game Over。  
- 审稿失败 = 卷宗标红洞 + 给出可执行补证提示。  
- 实验失败（模板内）= 可读原因 + 允许重跑（对齐 ResearchRun 既有语义）。

---

## 2. 三十分钟玩家剧本（垂直切片验收脚本）

**前提**：新档或 Reset Case；单人；中文 UI。

| 分钟 | 节拍 | 玩家行动 | 系统反馈 |
|---|---|---|---|
| 0–2 | 抵达 | 出生告示场，读异常通报 | 获得 Case `case_demo_01`，目标板：「查清矛盾结果从何而来」 |
| 2–6 | 第一现场 | 走到档案廊，检视海报 A（方法甲的高分报告） | 证据 `ev_poster_a` 入包；板上出现 claim「方法甲有效」 |
| 6–10 | 第二现场 | 检视失败残骸 B（复现日志红字） | 证据 `ev_repro_fail`；与上条 claim **冲突**高亮 |
| 10–13 | 信息门 | 学院找导师（需在其日程窗口；v0 可默认在岗） | 导师要你先陈述矛盾；陈述后解锁概念「对照/数据泄漏」笔记，软开下一线索 |
| 13–18 | 第三现场 | 数据碑/日志柜：发现切分方式可疑 | 证据 `ev_split_leak`；假设可从「方法甲魔法」改为「评估协议有偏」 |
| 18–23 | 实验 | 机房对白名单模板跑一次实验 | ResearchRun 完成 → `ev_exp_01`；支持或削弱某假设 |
| 23–26 | 整板 | 回 Lab：钉边、写问题与局限、删悬空 claim | 完整度条达到可送审阈值 |
| 26–28 | 审稿 | 送审稿庭 | 审稿人打出 1 个洞（如「缺对立」）；补钉后重送 → `passed` |
| 28–30 | 结案 | 导出 L1 | 生成报告页；可复制；可选「生成参观快照」（不挡结束） |

**剧本验收句**：观察者不问「任务做完了吗」，而会问「案子破了吗」。

---

## 3. Case 与证据数据格式

实现时可落 `localStorage` 优先；云同步后置。下列为逻辑模型（TypeScript 意向）。

### 3.1 Case

```ts
type CaseId = string;

type CaseStatus =
  | 'briefed'      // 已接
  | 'investigating'
  | 'ready_for_review'
  | 'revision'
  | 'closed';     // L1 已导出

interface ResearchCase {
  id: CaseId;
  title: { zh: string; en: string };
  // 玩家可见的异常简介
  bulletin: { zh: string; en: string };
  status: CaseStatus;
  // 卷宗正文槽
  researchQuestion: string;
  hypothesis: {
    text: string;
    // wild_guess | plausible | operationalized
    maturity: 'wild_guess' | 'plausible' | 'operationalized';
  };
  resultsSummary: string;
  limitations: string[];
  // 图
  claims: ClaimNode[];
  evidence: EvidenceNode[];
  edges: CaseEdge[];
  // 进度
  reviewRounds: ReviewRound[];
  experimentRuns: ExperimentLink[]; // 指向 ResearchRun id 或本地模拟 id
  completeness: number; // 0–1，由规则计算
  createdAt: string;
  updatedAt: string;
}
```

### 3.2 Evidence / Claim / Edge

```ts
type EvidenceKind =
  | 'citation'       // 文献/海报上的他人结果
  | 'testimony'      // 导师陈述（方法提示可降级为 note）
  | 'observation'    // 现场观察
  | 'log'            // 日志/复现失败记录
  | 'experiment_result'
  | 'note';           // 玩家或系统笔记

interface EvidenceNode {
  id: string;
  kind: EvidenceKind;
  title: { zh: string; en: string };
  body: { zh: string; en: string };
  // 世界来源
  sourcePoiId?: string;
  sourceNpcId?: 'mentor' | 'reviewer' | null;
  discoveredAt: string;
  tags: string[]; // e.g. ['leak', 'baseline', 'repro']
}

interface ClaimNode {
  id: string;
  text: string;
  // 玩家或系统生成的可被攻击单元
  state: 'active' | 'retracted' | 'accepted';
}

type EdgeType = 'supports' | 'contradicts' | 'refines' | 'instances';

interface CaseEdge {
  id: string;
  from: string; // evidenceId or claimId
  to: string;   // claimId or claimId
  type: EdgeType;
}
```

### 3.3 POI（世界兴趣点）

```ts
interface CasePoi {
  id: string;
  caseId: CaseId;
  sceneId: 'campus' | 'academy' | 'research' | 'compute' | 'paper' | 'plaza';
  // 世界坐标：2D legacy 或 3D，由适配层转换
  position: { x: number; y: number } | { x: number; y: number; z: number };
  label: { zh: string; en: string };
  // 吸引子表现：glow | smoke | poster | wreck | terminal
  attractor: string;
  // 检视后给予的证据
  grantsEvidenceIds: string[];
  // 软门槛：需要已发现的证据或 mentor flag
  requires?: { evidenceIds?: string[]; flags?: string[] };
  inspectOnce: boolean;
}
```

### 3.4 导师 / 审稿

```ts
interface MentorState {
  id: 'mentor_primary';
  // v0 可简化为 always_available；预留日程
  schedule: { days: number[]; startHour: number; endHour: number };
  // 已解锁的方法 flag
  unlockedFlags: string[];
  // 对话阶段
  stage: 'idle' | 'awaiting_contradiction' | 'taught_controls' | 'done';
}

interface ReviewRound {
  id: string;
  index: number;
  status: 'pending' | 'failed' | 'passed';
  findings: {
    code: 'no_baseline' | 'dangling_claim' | 'no_limitation' | 'overclaim' | 'no_repro';
    message: { zh: string; en: string };
    // 修复提示绑定到动作
    fixHint: { zh: string; en: string };
  }[];
  submittedAt: string;
  resolvedAt?: string;
}
```

### 3.5 完整度计算（v0 伪代码）

```
completeness =
  0.15 * has(researchQuestion) +
  0.15 * clamp(relatedEvidenceCount / 2) +
  0.15 * hypothesis.maturity != wild_guess +
  0.20 * hasExperimentLinked +
  0.15 * has(resultsSummary) +
  0.10 * limitations.length >= 1 +
  0.10 * danglingClaims == 0

ready_for_review when completeness >= 0.85
closed when review passed && export done
```

### 3.6 Demo Case 内容规格：`case_demo_01`

**标题（zh）**：被夸大的准确率  
**通报**：公开海报称方法甲在基准上远超方法乙；独立复现却得到持平甚至更差。谁在说谎——还是协议在说谎？

**预置 POI（建议 5 个）**

| poiId | 场景 | 证据 |
|---|---|---|
| `poi_poster_a` | plaza / campus | 海报：甲 92% |
| `poi_repro_log` | compute 附近残骸 | 复现失败/持平日志 |
| `poi_split_doc` | research 档案 | 可疑数据切分说明 |
| `poi_mentor` | academy | 对话，不直接给证据，给 flag `knows_leak` |
| `poi_bench` | compute | 触发实验槽 |

**预置审稿第一刀**：优先 `no_baseline` 或 `dangling_claim`（按玩家板状态动态选一个未满足项）。

---

## 4. Lab 证据板与 L1 导出

### 4.1 Lab UI（功能，不规定像素风）

1. **卷宗头**：问题 / 假设 / 结果 / 局限（文本）  
2. **双栏图**：左 Claims，右 Evidence；拖拽或点击建 `supports` / `contradicts`  
3. **冲突灯**：任意 `contradicts` 未解释时高亮  
4. **送审按钮**：`completeness` 不足时列出缺项  
5. **导出 L1**：生成只读报告

### 4.2 L1 报告大纲（固定节）

```markdown
# {title}
## 1. 问题
## 2. 相关工作与矛盾
## 3. 假设
## 4. 方法与验证
## 5. 结果
## 6. 局限
## 7. 附录：证据清单
```

附录自动从 `evidence[]` 渲染；正文槽由玩家填写 + 可选「从板生成草稿」按钮（模板拼接，v0 不要求 LLM）。

---

## 5. 与现有代码的挂载点（实现地图）

| 模块 | 建议 |
|---|---|
| 路由 `/research-game` | 保持；进岛即 Case 或显式「开始案件」 |
| `src/game` / `src/game3d` | POI 交互 → `discoverEvidence` |
| `ResearchRun` / `researchApi` | 实验 POI 完成后 `linkExperimentToCase` |
| 本地存档 | `localStorage` key 如 `aes:case:v0:{userOrAnon}` |
| Tutor | 可选：解释概念；**不**写卷宗权威状态 |
| 班级/Presence | v0 不依赖 |

**新建代码建议落点（实现阶段）**

- `src/game/case/types.ts` — 上文类型  
- `src/game/case/caseStore.ts` — 状态 + 完整度 + localStorage  
- `src/game/case/demoCase.ts` — `case_demo_01` 内容  
- `src/game/case/reviewRules.ts` — 审稿规则  
- `src/game/case/L1Export.ts` — 报告生成  
- UI：`CaseBulletinOverlay` / `EvidenceBoardPanel` / `ReviewOverlay` / `L1ReportModal`

---

## 6. 垂直切片范围与非目标

### 6.1 做（Must）

- [ ] 1 个 demo case 全流程可打通（第二节剧本）  
- [ ] ≥ 4 个可检视 POI + 1 实验挂钩  
- [ ] 证据板 supports/contradicts  
- [ ] 导师 1 段门控对话  
- [ ] 审稿 ≥ 1 轮可失败可重来  
- [ ] L1 导出  
- [ ] 本地存档与 Reset Case  
- [ ] 桌面可玩；移动端不崩（可简化拖拽）

### 6.2 不做（v0）

- 多 case 编年、赛季  
- 同学 NPC、组队、公会  
- 论点塔防、多审稿人 AI 人格树  
- 云同步 case、排行榜绑定  
- L2 投稿、Spark 开户绑定  
- 重做渲染管线/大世界扩张  

---

## 7. 实现顺序（批准后的工程波次）

| 波次 | 交付 | 验收 |
|---|---|---|
| **W1** | 类型 + store + demo 数据 + 无 3D 的板 UI 可点通 | ✅ `/research-game/case` 已可立案→L1（2026-07-20, b6169eb） |
| **W2** | 挂 3D/2D POI 检视 → 真发现证据 | ✅ 3D 注入 case landmarks + 导师门控 + CaseIslandHud；2D 显示 HUD（2026-07-20） |
| **W3** | ResearchRun 链接 + 导师对话状态机 | 实验与导师门可过 |
| **W4** | 审稿规则 + L1 导出 + Reset + 存档 | 剧本 0–30 分钟一次通过 |
| **W5** | 打磨文案、完整度提示、移动端降级 | Playwright 冒烟可选 |

---

## 8. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 仍像任务列表 | 文案全部用案件语言；目标板写「矛盾未解」不写「完成 3/5」 |
| 证据板太抽象 | 第一案线索强叙事、少抽象节点 |
| ResearchRun 环境未配置 | v0 允许「本地模拟实验结果」开关，语义与真 run 同构 |
| 导师变客服 | 无矛盾陈述不给关键 flag |
| 范围膨胀 | 严格 6.2；新想法进 backlog 文档而非本文件 |

---

## 9. Backlog（明确不在 v0）

- Case 02+ 内容包  
- 异步展览 / 短码互评  
- 审稿人多原型  
- 与课程 lesson 深度双向链  
- 班级周常  
- 证据构筑塔防化  

---

## 10. 签字栏（产品）

- [x] 主循环 = 案件探案 → L1 结案卷宗  
- [x] NPC = 1 导师 + 1 审稿  
- [x] 本文件为执行依据  

**下一工程动作**：启动 **W1**（case store + 板 UI + demo 数据），不依赖 3D 即可演示逻辑闭环。

---

*本文取代发散讨论中的默认方案描述；修改决策须改 D1–D6 并更新版本号。*
