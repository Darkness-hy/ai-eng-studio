from __future__ import annotations

from .models import Budget, ResearchTemplate, ResearchVariant

BOOTSTRAP_SEED = 20260711
BOOTSTRAP_REPLICATES = 1000
CONFIDENCE_LEVEL = 0.95
REPRODUCTION_TOLERANCE = 1e-12
TEMPLATE_VERSION = "3.0.0"


def _variant(variant_id: str, label: str, *pipeline: str) -> ResearchVariant:
    return ResearchVariant(id=variant_id, label=label, pipeline=list(pipeline))


def _template(
    *,
    number: int,
    template_id: str,
    name: str,
    name_zh: str,
    question: str,
    dataset: str,
    benchmark_kind: str,
    provenance: list[str],
    independent_variable: str,
    primary_metric: str,
    primary_metric_label: str,
    effect_threshold: float,
    variants: list[ResearchVariant],
    secondary_metrics: list[str],
    limitations: list[str],
    external_validation: list[str],
) -> ResearchTemplate:
    return ResearchTemplate(
        id=template_id,
        capstone_number=number,
        version=TEMPLATE_VERSION,
        name=name,
        name_zh=name_zh,
        research_question=question,
        dataset=dataset,
        benchmark_kind=benchmark_kind,
        provenance=provenance,
        independent_variable=independent_variable,
        primary_metric=primary_metric,
        primary_metric_label=primary_metric_label,
        effect_threshold=effect_threshold,
        variants=variants,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        secondary_metrics=secondary_metrics,
        limitations=limitations,
        external_validation=external_validation,
        budget=Budget(),
    )


TEMPLATES = (
    _template(
        number=1,
        template_id="terminal_coding_agent_v1",
        name="Capstone 01: Coding Harness Recovery",
        name_zh="终端编程智能体：Harness 恢复与安全",
        question=(
            "Do bounded observations, retry handling, lifecycle hooks, and checkpoints improve "
            "safe harness task completion over a naive tool loop?"
        ),
        dataset="coding-harness-fault-fixtures-v1",
        benchmark_kind="deterministic_fixture",
        provenance=["25 versioned tool-loop scenarios with known safe outcomes"],
        independent_variable="harness_control_surface",
        primary_metric="task_success_rate",
        primary_metric_label="安全任务完成率",
        effect_threshold=0.10,
        variants=[
            _variant("naive_loop", "Naive loop", "unbounded observation", "no recovery"),
            _variant("bounded_loop", "Bounded loop", "observation cap", "retry handling"),
            _variant("hooked_recovery", "Hooked recovery", "policy hooks", "checkpoint", "cancel"),
        ],
        secondary_metrics=["observation_bytes", "retries"],
        limitations=["Scripted tool outcomes test the harness, not model coding ability."],
        external_validation=["SWE-bench pass@1", "real repository PR quality"],
    ),
    _template(
        number=2,
        template_id="codebase_rag_retrieval_v1",
        name="Capstone 02: Real CodeSearchNet Python Retrieval",
        name_zh="代码库 RAG：CodeSearchNet 真实代码检索",
        question=(
            "Do AST-derived fields, local hybrid retrieval, and code-aware reranking improve "
            "nDCG@10 over a body-only BM25 baseline on the frozen CodeSearchNet Python human benchmark?"
        ),
        dataset="codesearchnet.python.human.v1",
        benchmark_kind="public_snapshot",
        provenance=[
            "github/CodeSearchNet@106e827405c968597da938f6b373d30183918869",
            "human annotation SHA-256 0340af32b551ceadb74fec147f97642b7fedf3ff039e38fb86baff49ee899846",
        ],
        independent_variable="retrieval_pipeline",
        primary_metric="ndcg_at_10",
        primary_metric_label="nDCG@10",
        effect_threshold=0.02,
        variants=[
            _variant("bm25_body", "BM25 Body", "source code", "BM25"),
            _variant("bm25_ast", "BM25 AST", "AST fields", "BM25"),
            _variant("hybrid", "Hybrid", "AST BM25", "local LSA", "RRF"),
            _variant("hybrid_rerank", "Hybrid + Reranker", "hybrid", "AST feature reranker"),
        ],
        secondary_metrics=["mrr_at_10", "recall_at_10", "citation_precision"],
        limitations=[
            "The judged-candidate metric ignores unjudged snippets.",
            "The frozen slice covers Python only and local LSA is not a production embedding model.",
        ],
        external_validation=["cross-language retrieval", "answer-generation quality"],
    ),
    _template(
        number=3,
        template_id="realtime_voice_vad_v1",
        name="Capstone 03: Offline VAD Robustness",
        name_zh="实时语音助手：离线 VAD 抗噪",
        question="Does adaptive hysteresis with hangover improve VAD frame F1 over a fixed energy threshold?",
        dataset="seeded-audio-signal-fixtures-v1",
        benchmark_kind="deterministic_fixture",
        provenance=["24 seeded 8 kHz speech/noise/dropout waveforms with exact frame labels"],
        independent_variable="vad_algorithm",
        primary_metric="vad_frame_f1",
        primary_metric_label="VAD Frame F1",
        effect_threshold=0.03,
        variants=[
            _variant("fixed_energy", "Fixed energy", "RMS", "fixed threshold"),
            _variant("adaptive_energy", "Adaptive energy", "noise floor", "dynamic threshold"),
            _variant("hysteresis_hangover", "Hysteresis + hangover", "dual threshold", "hangover"),
        ],
        secondary_metrics=["precision", "recall", "packet_loss"],
        limitations=["Generated waveforms do not represent conversational ASR or TTS quality."],
        external_validation=["real-call WER", "MOS", "barge-in latency"],
    ),
    _template(
        number=4,
        template_id="multimodal_document_retrieval_v1",
        name="Capstone 04: Layout Evidence Retrieval",
        name_zh="多模态文档问答：版面证据检索",
        question=(
            "Do layout fields and cross-page links improve evidence ranking over body-only "
            "lexical retrieval?"
        ),
        dataset="layout-page-evidence-fixtures-v1",
        benchmark_kind="deterministic_fixture",
        provenance=["24 versioned page/table/link questions with known evidence pages"],
        independent_variable="document_retrieval_pipeline",
        primary_metric="evidence_ndcg_at_3",
        primary_metric_label="Evidence nDCG@3",
        effect_threshold=0.05,
        variants=[
            _variant("body_lexical", "Body lexical", "body tokens"),
            _variant("layout_aware", "Layout-aware", "headings", "tables", "body"),
            _variant("cross_page", "Cross-page", "layout retrieval", "page links"),
        ],
        secondary_metrics=["reciprocal_rank", "cross_page"],
        limitations=["Text and layout fixtures do not measure OCR or VLM answer generation."],
        external_validation=["ViDoRe", "M3DocVQA", "human multi-page QA"],
    ),
    _template(
        number=5,
        template_id="autonomous_research_search_v1",
        name="Capstone 05: Bounded Experiment Search",
        name_zh="自主科研智能体：有界实验搜索",
        question="Does adaptive budget allocation find better configurations than a fixed sweep?",
        dataset="seeded-objective-landscapes-v1",
        benchmark_kind="deterministic_fixture",
        provenance=["24 deterministic non-convex objective landscapes with exhaustive optima"],
        independent_variable="experiment_scheduler",
        primary_metric="normalized_search_quality",
        primary_metric_label="归一化搜索质量",
        effect_threshold=0.01,
        variants=[
            _variant("fixed_sweep", "Fixed sweep", "fixed configurations"),
            _variant("random_search", "Random search", "seeded samples"),
            _variant("adaptive_search", "Adaptive search", "coarse probe", "local refinement"),
        ],
        secondary_metrics=["evaluations", "regret"],
        limitations=["Objective landscapes test scheduling, not hypothesis or paper-writing quality."],
        external_validation=["literature quality", "novelty review", "paper blind review"],
    ),
    _template(
        number=6,
        template_id="kubernetes_rca_v1",
        name="Capstone 06: Kubernetes Root-Cause Ranking",
        name_zh="Kubernetes 排障智能体：根因排序",
        question="Do dependency and temporal graph signals improve root-cause ranking over keyword severity?",
        dataset="kubernetes-incident-graph-fixtures-v1",
        benchmark_kind="deterministic_fixture",
        provenance=["30 dependency-graph incidents with injected roots and timestamped symptoms"],
        independent_variable="root_cause_ranker",
        primary_metric="root_cause_top1_utility",
        primary_metric_label="RCA Top-1 Utility",
        effect_threshold=0.03,
        variants=[
            _variant("keyword_rank", "Keyword severity", "local symptom severity"),
            _variant("dependency_rank", "Dependency graph", "upstream propagation"),
            _variant("temporal_graph_rank", "Temporal graph", "dependency", "first-failure weighting"),
        ],
        secondary_metrics=["root_rank", "observation_count"],
        limitations=["Injected incidents do not establish live-cluster remediation safety."],
        external_validation=["real Kubernetes incidents", "Slack approval", "ArgoCD rollback"],
    ),
    _template(
        number=7,
        template_id="local_finetuning_v1",
        name="Capstone 07: Local Fine-Tuning Pipeline",
        name_zh="端到端微调流水线：本地监督训练",
        question="Do scaling and tuned optimization improve held-out accuracy over one-pass SGD?",
        dataset="sklearn-digits-packaged-v1",
        benchmark_kind="packaged_dataset",
        provenance=["scikit-learn packaged handwritten digits dataset; eight fixed stratified splits"],
        independent_variable="training_pipeline",
        primary_metric="classification_accuracy",
        primary_metric_label="测试集准确率",
        effect_threshold=0.02,
        variants=[
            _variant("one_pass_sgd", "One-pass SGD", "raw pixels", "one pass"),
            _variant("scaled_sgd", "Scaled SGD", "standardization", "iterative SGD"),
            _variant("tuned_logreg", "Tuned logistic", "standardization", "regularized optimization"),
        ],
        secondary_metrics=["train_examples", "test_examples"],
        limitations=["This CPU classifier validates training mechanics, not LLM SFT or DPO quality."],
        external_validation=["LLM SFT", "DPO", "quantization", "serving throughput"],
    ),
    _template(
        number=8,
        template_id="regulated_rag_acl_v1",
        name="Capstone 08: ACL-Aware RAG Retrieval",
        name_zh="生产级 RAG：权限感知检索",
        question=(
            "Do ACL filtering and abstaining rerank improve secure answer utility over "
            "unfiltered retrieval?"
        ),
        dataset="regulated-document-acl-fixtures-v1",
        benchmark_kind="deterministic_fixture",
        provenance=["28 role/query/document cases with known authorization and answer outcomes"],
        independent_variable="access_controlled_retrieval",
        primary_metric="secure_answer_rate",
        primary_metric_label="安全回答正确率",
        effect_threshold=0.05,
        variants=[
            _variant("unfiltered_retrieval", "Unfiltered", "lexical retrieval"),
            _variant("acl_filter", "ACL filter", "role filter", "retrieval"),
            _variant("acl_rerank", "ACL + abstain", "role filter", "rerank", "abstention"),
        ],
        secondary_metrics=["authorized", "answered"],
        limitations=["Fixture ACLs do not constitute a regulatory compliance audit."],
        external_validation=["RAGAS", "jurisdiction policy review", "production drift"],
    ),
    _template(
        number=9,
        template_id="code_migration_v1",
        name="Capstone 09: Verified Code Migration",
        name_zh="代码迁移智能体：AST 迁移与验证",
        question=(
            "Do AST transforms and compile-time verification improve migration success over "
            "regex recipes?"
        ),
        dataset="python-typing-migration-fixtures-v1",
        benchmark_kind="deterministic_fixture",
        provenance=["25 Python typing-modernization fixtures with literal-preservation checks"],
        independent_variable="migration_strategy",
        primary_metric="migration_success_rate",
        primary_metric_label="迁移成功率",
        effect_threshold=0.05,
        variants=[
            _variant("regex_recipe", "Regex recipe", "text replacement"),
            _variant("ast_recipe", "AST recipe", "parsed transform"),
            _variant("verified_ast_recipe", "Verified AST", "AST transform", "compile and invariants"),
        ],
        secondary_metrics=["literal_preserved", "compiled"],
        limitations=["Python typing fixtures do not represent repository-scale dependency upgrades."],
        external_validation=["MigrationBench", "repository tests", "coverage preservation"],
    ),
    _template(
        number=10,
        template_id="multi_agent_coordination_v1",
        name="Capstone 10: Contract-Aware Coordination",
        name_zh="多智能体软件团队：契约感知协作",
        question=(
            "Does file-contract scheduling improve quality-adjusted throughput over sequential "
            "execution?"
        ),
        dataset="task-file-conflict-fixtures-v1",
        benchmark_kind="deterministic_fixture",
        provenance=["24 seeded task graphs with durations, file ownership, and exact conflict counts"],
        independent_variable="coordination_strategy",
        primary_metric="quality_adjusted_throughput",
        primary_metric_label="质量调整吞吐",
        effect_threshold=0.05,
        variants=[
            _variant("sequential", "Sequential", "single worker"),
            _variant("naive_parallel", "Naive parallel", "uncoordinated workers"),
            _variant("contract_parallel", "Contract parallel", "file ownership", "parallel schedule"),
        ],
        secondary_metrics=["speedup", "merge_conflicts", "elapsed"],
        limitations=["Deterministic workers measure coordination, not model-generated code quality."],
        external_validation=["SWE-bench", "human code review", "token efficiency"],
    ),
    _template(
        number=11,
        template_id="llm_observability_detection_v1",
        name="Capstone 11: Trace Regression Detection",
        name_zh="LLM 可观测面板：Trace 回归检测",
        question=(
            "Does robust multi-signal detection improve injected regression accuracy over a "
            "fixed threshold?"
        ),
        dataset="seeded-genai-trace-windows-v1",
        benchmark_kind="deterministic_fixture",
        provenance=["60 seeded latency/error/cost windows with injected regression labels"],
        independent_variable="regression_detector",
        primary_metric="regression_detection_accuracy",
        primary_metric_label="回归检测准确率",
        effect_threshold=0.05,
        variants=[
            _variant("fixed_threshold", "Fixed threshold", "latency threshold"),
            _variant("latency_zscore", "Latency z-score", "baseline distribution"),
            _variant("robust_multisignal", "Robust multi-signal", "latency", "errors", "cost"),
        ],
        secondary_metrics=["predicted_regression", "actual_regression", "latency_ms"],
        limitations=["Seeded windows test detector logic, not sustained ClickHouse ingest capacity."],
        external_validation=["1k spans/sec load", "Alertmanager routing", "human-labeled eval drift"],
    ),
    _template(
        number=12,
        template_id="video_temporal_grounding_v1",
        name="Capstone 12: Temporal Video Grounding",
        name_zh="视频理解流水线：时序定位",
        question="Does segment smoothing improve temporal IoU over global or single-frame matching?",
        dataset="seeded-temporal-feature-fixtures-v1",
        benchmark_kind="deterministic_fixture",
        provenance=["24 seeded frame-feature sequences with exact event intervals"],
        independent_variable="temporal_retrieval_strategy",
        primary_metric="temporal_iou",
        primary_metric_label="Temporal IoU",
        effect_threshold=0.05,
        variants=[
            _variant("global_pool", "Global pool", "whole-video interval"),
            _variant("frame_match", "Frame match", "peak frame"),
            _variant("segment_smoothing", "Segment smoothing", "frame similarity", "temporal smoothing"),
        ],
        secondary_metrics=["start_error", "segment_length"],
        limitations=["Feature fixtures do not measure visual encoder or natural-video QA quality."],
        external_validation=["ActivityNet-QA", "NeXT-GQA", "real video ingest"],
    ),
    _template(
        number=13,
        template_id="mcp_registry_governance_v1",
        name="Capstone 13: MCP Registry Governance",
        name_zh="MCP 服务器：注册与治理",
        question=(
            "Do scope, approval, and PII controls improve governance decision accuracy over "
            "schema checks alone?"
        ),
        dataset="mcp-call-governance-fixtures-v1",
        benchmark_kind="deterministic_fixture",
        provenance=["36 typed MCP calls with schema, scope, approval, and PII ground truth"],
        independent_variable="tool_governance_policy",
        primary_metric="governance_decision_accuracy",
        primary_metric_label="治理决策准确率",
        effect_threshold=0.05,
        variants=[
            _variant("schema_only", "Schema only", "argument validation"),
            _variant("schema_scope", "Schema + scope", "argument validation", "scope check"),
            _variant("full_governance", "Full governance", "scope", "approval", "PII policy"),
        ],
        secondary_metrics=["allowed", "expected_allowed"],
        limitations=[
            "In-process calls do not establish StreamableHTTP interoperability or horizontal scale."
        ],
        external_validation=["MCP conformance", "100-client load", "real identity provider"],
    ),
    _template(
        number=14,
        template_id="speculative_decoding_v1",
        name="Capstone 14: Exact Speculative Decoding",
        name_zh="投机解码服务器：输出等价与加速",
        question=(
            "Does an aligned adaptive draft improve exactness-adjusted speedup over target-only "
            "decoding?"
        ),
        dataset="deterministic-markov-decode-fixtures-v1",
        benchmark_kind="deterministic_fixture",
        provenance=["17 deterministic prompts with an exhaustive target decoder and exact output oracle"],
        independent_variable="draft_strategy",
        primary_metric="exactness_adjusted_speedup",
        primary_metric_label="等价性调整加速分",
        effect_threshold=0.05,
        variants=[
            _variant("target_only", "Target only", "one target call per token"),
            _variant("misaligned_draft", "Misaligned draft", "four-token draft", "target verify"),
            _variant("adaptive_draft", "Adaptive draft", "aligned draft", "target verify"),
        ],
        secondary_metrics=["speedup", "target_calls", "exact"],
        limitations=["Markov decoding validates the algorithm, not GPU kernels or large-model tail latency."],
        external_validation=["vLLM EAGLE", "GPU throughput", "batch-size p99"],
    ),
    _template(
        number=15,
        template_id="constitutional_safety_gate_v1",
        name="Capstone 15: Constitutional Safety Gate",
        name_zh="宪法式安全护栏：多语言规则闸门",
        question=(
            "Do normalization and context policy improve safety decision accuracy over keyword "
            "blocking?"
        ),
        dataset="bilingual-safety-policy-fixtures-v1",
        benchmark_kind="deterministic_fixture",
        provenance=["36 bilingual benign/unsafe policy fixtures with plain, zero-width, and base64 forms"],
        independent_variable="safety_gate_pipeline",
        primary_metric="safety_decision_accuracy",
        primary_metric_label="安全决策准确率",
        effect_threshold=0.05,
        variants=[
            _variant("keyword_gate", "Keyword gate", "surface terms"),
            _variant("normalized_gate", "Normalized gate", "Unicode", "base64", "surface terms"),
            _variant("context_policy_gate", "Context policy", "normalization", "benign-context exceptions"),
        ],
        secondary_metrics=["blocked", "unsafe"],
        limitations=["Authored policy fixtures do not measure open-ended jailbreak robustness."],
        external_validation=["XSTest", "garak", "multilingual red team", "human safety review"],
    ),
    _template(
        number=16,
        template_id="issue_patch_sandbox_v1",
        name="Capstone 16: Issue Patch Sandbox",
        name_zh="Issue 转 PR：Patch 沙箱策略",
        question=(
            "Do path, secret, force-push, and coverage gates improve patch acceptance accuracy "
            "over tests alone?"
        ),
        dataset="patch-policy-fixtures-v1",
        benchmark_kind="deterministic_fixture",
        provenance=["40 patch outcomes with test, path, force-push, secret, and coverage ground truth"],
        independent_variable="patch_acceptance_policy",
        primary_metric="patch_policy_accuracy",
        primary_metric_label="Patch 策略准确率",
        effect_threshold=0.05,
        variants=[
            _variant("tests_only", "Tests only", "test result"),
            _variant("path_policy", "Path policy", "tests", "allowed paths", "no force-push"),
            _variant("full_patch_policy", "Full policy", "paths", "secrets", "coverage", "tests"),
        ],
        secondary_metrics=["accepted", "expected_accepted"],
        limitations=["Patch metadata fixtures do not solve issues or create real pull requests."],
        external_validation=["GitHub App", "repository CI", "issue resolution pass@1"],
    ),
    _template(
        number=17,
        template_id="adaptive_tutor_bkt_v1",
        name="Capstone 17: Bayesian Adaptive Tutor",
        name_zh="个人 AI 导师：贝叶斯知识追踪",
        question="Does BKT with spacing improve post-intervention mastery over a linear curriculum?",
        dataset="seeded-learner-simulation-v1",
        benchmark_kind="deterministic_fixture",
        provenance=["24 seeded learners, six concepts, and explicit guess/slip/learning dynamics"],
        independent_variable="curriculum_policy",
        primary_metric="post_intervention_mastery",
        primary_metric_label="干预后掌握度",
        effect_threshold=0.02,
        variants=[
            _variant("linear_curriculum", "Linear curriculum", "round-robin concepts"),
            _variant("weakest_first", "Weakest first", "BKT estimate", "lowest mastery"),
            _variant("bkt_spaced", "BKT + spacing", "BKT estimate", "spacing urgency"),
        ],
        secondary_metrics=["mastery_gain", "initial_mastery"],
        limitations=["Simulated learners cannot establish real educational efficacy or safety."],
        external_validation=["two-week learner study", "Socratic fidelity", "privacy and child-safety audit"],
    ),
)

TEMPLATE_BY_ID = {template.id: template for template in TEMPLATES}


def get_template(template_id: str) -> ResearchTemplate:
    try:
        return TEMPLATE_BY_ID[template_id]
    except KeyError as exc:
        raise ValueError(f"unknown research template: {template_id}") from exc


def list_templates() -> list[ResearchTemplate]:
    return list(TEMPLATES)
