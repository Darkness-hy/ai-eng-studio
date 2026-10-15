# ResearchRun Design

## Decision

ResearchRun 是科研岛五个区域共用的后端状态机：

```text
选题 -> 假设 -> 固定 CPU 实验 -> 结果分析 -> 独立复现 -> 可公开研究卡
```

版本 `3.0.0` 提供 17 个固定离线模板。模板清单和各项目的验证边界见
[`zero-api-capstone-matrix.md`](zero-api-capstone-matrix.md)。

## Trust Boundary

服务端只接受：

- 一个白名单 `templateId`；
- 课题名称；
- 假设陈述、机制解释和预期方向；
- 带版本号和 idempotency key 的状态转换命令。

服务端不接受用户代码、shell 命令、URL、包、数据集、variant、资源限制或
任意实验参数。模板目录负责选择完整实验配置，数据库再次校验模板和 variant
白名单。

生产 runner 必须满足：

- immutable image digest；
- `--network none`；
- read-only root filesystem；
- 1 CPU、1 GiB、120 秒；
- 最多两个并行 job，每个 variant 最多一次显式重试。

## Template Contract

`GET /v1/research-templates` 返回每个模板的：

- `id`、毕业项目编号、中英文名称和版本；
- 固定研究问题、数据集和 provenance；
- 自变量、主指标、预期方向和预先声明的 effect threshold；
- 3 到 4 个固定 variant 及其 pipeline；
- secondary metrics、限制和外部验证项；
- `$0` paid API 和 offline runtime 标记。

Capstone 02 使用冻结的 CodeSearchNet 公开快照。Capstone 07 使用 sklearn
打包数据。其余模板使用版本化、确定性的 fixture，且在 UI 和产物中明确标注，
不得表述为真实用户或生产系统效果。

## Generic Metrics

每个 variant 输出同一个通用结构：

```text
caseResults[]:
  caseId
  primaryScore in [0, 1]
  secondary: metric -> number
  latencyMs
  evidence[]

bootstrapResults[1000]:
  replicate
  primaryScore
  sampleSha256

aggregate:
  primaryScoreMean
  primaryScoreSampleStd
  secondaryMeans
  caseLatencyMsMean
  caseLatencyMsP95
  caseCount

resultSha256
```

`resultSha256` 覆盖 case ID、主指标、secondary metrics 和 evidence，不包含机器
相关 latency。复现器比较该哈希和主指标绝对差，容差为 `1e-12`。

## Analysis Rule

分析固定比较第一个 baseline variant 和最后一个 candidate variant，同时报告所有
中间 ablation 和实际最佳 variant。

- Bootstrap seed: `20260711`
- Replicates: `1000`
- Confidence interval: `95%`
- Increase: mean delta 达到模板阈值且 CI lower `> 0`
- Decrease: mean delta 小于负阈值且 CI upper `< 0`
- No change: absolute mean delta 小于阈值且 CI 包含 `0`
- 不报告 p-value；该区间是配对重采样不确定性，不宣称为通用显著性检验。

`partially_supported` 表示方向正确，但没有同时满足预注册效应阈值和区间规则。
系统保留负向 incremental delta，也允许中间 variant 成为最佳方案。

## State Machine

```text
draft
  -> hypothesis_ready
  -> queued
  -> running
  -> analysis_ready
  -> analyzing
  -> verification_required
  -> verifying
  -> completed

any active stage -> failed -> explicit retry
queued/running/analyzing/verifying -> cancelled
```

每个 accepted command 和后台状态转换增加 `run.version`。写命令必须携带
`expectedVersion`；版本不一致返回 `version_conflict`。创建、排队、分析、验证、
重试、取消、公开和撤回均使用 idempotency key。

## Persistence And Access

- `research_runs`: owner、模板、状态、revision 和预算。
- `research_hypotheses`: 每个 run 一个 accepted hypothesis。
- `research_experiment_jobs`: immutable attempts 和 measured metrics。
- `research_artifacts`: 版本化 topic、spec、raw metrics、analysis、verification bundle
  和 research card。
- `research_events`: append-only audit log 和 SSE cursor。

Supabase JWT 的 `sub` 是唯一用户身份来源。浏览器通过 RLS 只读自己的 run；Research
API 使用 server-only PostgreSQL credential 执行事务。匿名用户只能读取已通过验证且
显式公开的安全 JSON 产物，不能读取 `user_id`、日志或 Storage pointer。

## Publication

验证步骤重新运行全部 variant，重新计算 analysis hash，并生成 verification bundle。
只有完整性检查和复现都通过时才生成 research card。公开命令只允许：

- run 已完成；
- artifact 是 `verification_bundle` 或 `research_card`；
- `verificationStatus = passed`；
- 内容是裁剪后的 JSON；
- 没有 Storage pointer。

公开和撤回均可审计且可幂等重放。

## API Surface

Canonical schema: [`research-run-openapi.json`](research-run-openapi.json).

```text
GET    /health
GET    /v1/research-templates
GET    /v1/research-runs
POST   /v1/research-runs
GET    /v1/research-runs/{runId}
PUT    /v1/research-runs/{runId}/hypothesis
GET    /v1/research-runs/{runId}/experiment-jobs
POST   /v1/research-runs/{runId}/experiment-jobs
POST   /v1/research-runs/{runId}/analysis
POST   /v1/research-runs/{runId}/verify
POST   /v1/research-runs/{runId}/retry
POST   /v1/research-runs/{runId}/cancel
GET    /v1/research-runs/{runId}/events
GET    /v1/research-runs/{runId}/artifacts
POST   /v1/research-artifacts/{artifactId}/publish
DELETE /v1/research-artifacts/{artifactId}/publication
```

SSE 使用带 `Authorization` header 的 authenticated `fetch` stream。token 不放在 URL。

## Database Migration

1. `supabase/013-research-runs.sql` 创建基础表、RLS、索引和 publication RPC。
2. `supabase/014-zero-api-capstone-portfolio.sql` 加入 17 个模板白名单和 v3 通用配置。

014 保留历史 `2.0.0` Code RAG 行，但所有新 run 使用 `3.0.0`。先在非生产数据库
执行并通过 PostgreSQL/RLS 测试；本次开发不修改生产 Supabase。

## Verification Gates

```bash
cd server/research-api
uv run ruff check research_api tests
uv run pytest -q
TEST_DATABASE_URL=postgresql://... uv run pytest -q

cd ../..
npm run lint
npm run build
```

发布前还必须用 immutable Docker image 逐个执行 17 个模板，并在桌面与移动 viewport
检查模板选择、状态转换、结果显示、公开和 fallback 模式。
