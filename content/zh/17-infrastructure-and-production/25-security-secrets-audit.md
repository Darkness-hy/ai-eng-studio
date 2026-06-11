# 安全 — 密钥管理、API Key 轮换、审计日志与防护栏

> 用集中式保险库（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消除密钥散落（secret sprawl）。绝不要把凭证存放在配置文件、纳入版本控制的 env 文件或电子表格里。优先使用 IAM 角色而非静态密钥；CI/CD 使用 OIDC。AI 网关模式是 2026 年的标准方案：应用 → 网关 → 模型供应商，由网关在运行时从保险库拉取凭证。在保险库中轮换密钥，所有应用几分钟内自动生效——无需重新部署，也不用在 Slack 上发"谁有新密钥"的消息。轮换策略 ≤90 天；每次提交都用 TruffleHog / GitGuardian / Gitleaks 扫描。零信任：MFA、SSO、RBAC/ABAC、短时效令牌、设备状态检查。PII 清洗使用实体识别在转发前掩码 PHI/PII；一致性令牌化（Mesh 方案）把敏感值映射到稳定的占位符，使 LLM 保留代码和关系语义。网络出站：LLM 服务部署在专用 VPC/VNet 子网中，仅白名单放行 `api.openai.com`、`api.anthropic.com` 等；阻断其余所有出站流量。2026 年的标志性事件：Vercel 供应链攻击通过被攻陷的 CI/CD 凭证，在数千个客户部署中窃取了环境变量。

**Type:** Learn
**Languages:** Python (stdlib, toy PII-scrubber + audit-log writer)
**Prerequisites:** Phase 17 · 19 (AI Gateways), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## 学习目标

- 列举四种密钥管理反模式（配置文件入版本控制、硬编码环境变量、电子表格、静态密钥），并说出对应的替代方案。
- 解释"AI 网关从保险库拉取凭证"这一 2026 年生产标准模式。
- 实现一个带一致性令牌化的 PII 清洗器（相同的值 → 相同的占位符），使语义得以保留。
- 说出 2026 年 Vercel 供应链事件，以及它对 CI/CD 凭证卫生的教训。

## 问题背景

一名实习生把含有 API 密钥的 `.env` 提交了上去。他很快删掉了文件。但密钥已经留在 git 历史里——GitGuardian 扫描发现了它，而你们的轮换流程是"在 Slack 里通知团队，更新 40 个配置文件，重新部署所有服务"。8 小时后，一半服务已上线，另一半还在等部署窗口。

另一边，用户的提示词中包含"我的 SSN 是 123-45-6789"。这条提示词被发送到了 OpenAI。你们签了 BAA，但内部政策要求在转发前掩码 PII。而你们没做。

再另一边，你们 EKS 集群里的 LLM Pod 可以访问任意互联网主机。有人通过向攻击者控制的域名发起 DNS 查询来窃取数据。没有任何机制拦截它。

LLM 服务的安全必须同时覆盖这三个攻击向量。保险库托管的凭证。PII 清洗。网络出站过滤。审计日志。

## 核心概念

### 集中式保险库 + IAM 角色拉取

**保险库（Vault）**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。唯一的可信来源。

**IAM 角色**：应用/网关通过其 IAM 身份认证，而不是静态密钥。保险库返回的密钥仅在令牌有效期内可用。

**AI 网关模式**：网关在请求时从保险库拉取 `OPENAI_API_KEY`。在保险库中轮换；下一个请求就会拿到新密钥。无需重新部署。

### 轮换策略 ≤ 90 天

覆盖所有 API 密钥、保险库根令牌、CI/CD 凭证。能自动化的轮换都自动化。手动轮换必须记录并跟踪。

### 密钥扫描

- **TruffleHog** — 对提交做正则 + 熵值检测。
- **GitGuardian** — 商业产品，准确率高。
- **Gitleaks** — 开源，可在 CI 中运行。

每次提交都要扫描。检测到新密钥就阻断 PR。

### 零信任姿态

- 所有账户强制 MFA。
- 通过 SAML/OIDC 实现 SSO。
- 用 RBAC（基于角色）或 ABAC（基于属性）做细粒度访问控制。
- 短时效令牌（以小时计，而非天）。
- 设备状态检查——只允许启用了磁盘加密的公司设备。

### PII / PHI 清洗

在提示词离开你的基础设施之前：

1. 实体识别（spaCy NER、Presidio、商业方案）。
2. 掩码匹配到的实体：`"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`。
3. 一致性令牌化（Mesh 方案）：相同的值映射到相同的占位符，使 LLM 保留实体间的关系。
4. 可选：对 LLM 响应做反向映射。

静态正则过滤器能捕获基础模式；NER 能捕获更多。两者都要用。

### 输入 + 输出防护栏

输入侧：阻断已知越狱（jailbreak）和禁止话题；按用户做限流。

输出侧：用正则清洗泄露的密钥（API 密钥模式、拒答场景下的邮箱模式），用分类器检测违规内容。

### 网络出站白名单

LLM 服务部署在专用子网中：
- 白名单：`api.openai.com`、`api.anthropic.com`、向量数据库端点、保险库端点。
- 其余流量：全部丢弃。
- DNS 走仅允许白名单的解析器（防止 DNS 隧道窃取数据）。

### 审计日志

为每次 LLM 调用记录不可篡改的日志，包含：
- 时间戳。
- 用户 / 租户。
- 提示词哈希（出于隐私不存原始提示词）。
- 模型 + 版本。
- token 数量。
- 成本。
- 响应哈希。
- 触发的任何防护栏事件。

按监管要求保留（SOC 2 为 1 年，HIPAA 为 6 年）。

### 2026 年 Vercel 事件

供应链攻击：被攻陷的 CI/CD 凭证在数千个客户部署中窃取了环境变量。教训：CI/CD 凭证等同于生产凭证。存入保险库。最小化授权范围。激进地轮换。

### 必须记住的数字

- 轮换策略：≤ 90 天。
- 每次提交都扫描：TruffleHog / GitGuardian / Gitleaks。
- Vercel 2026：CI/CD 凭证被攻陷 → 数千个客户的环境变量泄露。
- 审计日志保留期：SOC 2 = 1 年，HIPAA = 6 年。

## 生产实践

`code/main.py` 实现了一个带一致性令牌化的玩具级 PII 清洗器，以及一个仅追加（append-only）的审计日志。

## 交付产物

本课产出 `outputs/skill-llm-security-plan.md`。给定监管范围与现状，规划保险库迁移、清洗器、出站策略和审计日志。

## 练习

1. 运行 `code/main.py`。发送两条引用同一个 SSN 的提示词，确认两者得到相同的占位符。
2. 为一个部署在 EKS 上的 vLLM 服务设计网络出站策略，该服务需要调用 OpenAI + Anthropic + Weaviate。
3. 你在 git 历史中发现了一个密钥（已存在 2 年）。正确的应对是什么——轮换密钥、清理历史，还是两者都做？给出理由。
4. 你的审计日志每天增长 10 GB。设计分层保留策略（热数据 30 天，温数据 12 个月，冷数据 6 年）。
5. 论证反向令牌化（把真实值替换回 LLM 响应）是否值得其复杂度，还是直接让占位符可见更好。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|------------------------|
| Vault | "密钥存储" | 集中式凭证管理服务 |
| IAM 角色 | "基于身份的认证" | 由应用扮演的角色；返回短时效凭证 |
| CI/CD 的 OIDC | "云端签发的令牌" | CI 中无静态密钥——通过 OIDC 确认身份 |
| TruffleHog / GitGuardian / Gitleaks | "密钥扫描器" | 提交时的密钥检测 |
| RBAC / ABAC | "访问控制" | 基于角色 vs 基于属性 |
| PII 清洗 | "数据脱敏" | 移除或令牌化敏感实体 |
| 一致性令牌化 | "稳定占位符" | 相同的值每次映射到相同的令牌 |
| Mesh 方案 | "Mesh 令牌化" | 保留语义的令牌化模式 |
| 出站白名单 | "出站允许列表" | 只有许可的域名可达 |
| 审计日志 | "不可篡改的历史记录" | 用于合规的仅追加记录 |

## 延伸阅读

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII 检测与匿名化。
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)
