-- 014 - Expand ResearchRun from one Code RAG template to 17 fixed offline capstones.
-- This preserves read-only 2.0.0 Code RAG history while all new runs use 3.0.0.

begin;

alter table public.research_runs
  alter column template_version set default '3.0.0';
alter table public.research_experiment_jobs
  alter column template_version set default '3.0.0';

alter table public.research_runs
  drop constraint if exists research_runs_question_template_check,
  drop constraint if exists research_runs_template_check,
  drop constraint if exists research_runs_template_version_check;

alter table public.research_runs
  add constraint research_runs_template_check
    check (template_id in (
      'terminal_coding_agent_v1',
      'codebase_rag_retrieval_v1',
      'realtime_voice_vad_v1',
      'multimodal_document_retrieval_v1',
      'autonomous_research_search_v1',
      'kubernetes_rca_v1',
      'local_finetuning_v1',
      'regulated_rag_acl_v1',
      'code_migration_v1',
      'multi_agent_coordination_v1',
      'llm_observability_detection_v1',
      'video_temporal_grounding_v1',
      'mcp_registry_governance_v1',
      'speculative_decoding_v1',
      'constitutional_safety_gate_v1',
      'issue_patch_sandbox_v1',
      'adaptive_tutor_bkt_v1'
    )),
  add constraint research_runs_template_version_check
    check (
      template_version = '3.0.0'
      or (template_id = 'codebase_rag_retrieval_v1' and template_version = '2.0.0')
    );

alter table public.research_hypotheses
  drop constraint if exists research_hypotheses_independent_variable_check,
  drop constraint if exists research_hypotheses_dependent_metric_check;

alter table public.research_experiment_jobs
  drop constraint if exists research_jobs_variant_check,
  drop constraint if exists research_jobs_template_check,
  drop constraint if exists research_jobs_template_version_check,
  drop constraint if exists research_jobs_config_check;

alter table public.research_experiment_jobs
  add constraint research_jobs_template_check
    check (template_id in (
      'terminal_coding_agent_v1',
      'codebase_rag_retrieval_v1',
      'realtime_voice_vad_v1',
      'multimodal_document_retrieval_v1',
      'autonomous_research_search_v1',
      'kubernetes_rca_v1',
      'local_finetuning_v1',
      'regulated_rag_acl_v1',
      'code_migration_v1',
      'multi_agent_coordination_v1',
      'llm_observability_detection_v1',
      'video_temporal_grounding_v1',
      'mcp_registry_governance_v1',
      'speculative_decoding_v1',
      'constitutional_safety_gate_v1',
      'issue_patch_sandbox_v1',
      'adaptive_tutor_bkt_v1'
    )),
  add constraint research_jobs_template_version_check
    check (
      template_version = '3.0.0'
      or (template_id = 'codebase_rag_retrieval_v1' and template_version = '2.0.0')
    ),
  add constraint research_jobs_variant_check
    check (
      case template_id
        when 'terminal_coding_agent_v1' then variant in ('naive_loop', 'bounded_loop', 'hooked_recovery')
        when 'codebase_rag_retrieval_v1' then variant in ('bm25_body', 'bm25_ast', 'hybrid', 'hybrid_rerank')
        when 'realtime_voice_vad_v1' then variant in ('fixed_energy', 'adaptive_energy', 'hysteresis_hangover')
        when 'multimodal_document_retrieval_v1' then variant in ('body_lexical', 'layout_aware', 'cross_page')
        when 'autonomous_research_search_v1' then variant in ('fixed_sweep', 'random_search', 'adaptive_search')
        when 'kubernetes_rca_v1' then variant in ('keyword_rank', 'dependency_rank', 'temporal_graph_rank')
        when 'local_finetuning_v1' then variant in ('one_pass_sgd', 'scaled_sgd', 'tuned_logreg')
        when 'regulated_rag_acl_v1' then variant in ('unfiltered_retrieval', 'acl_filter', 'acl_rerank')
        when 'code_migration_v1' then variant in ('regex_recipe', 'ast_recipe', 'verified_ast_recipe')
        when 'multi_agent_coordination_v1' then variant in ('sequential', 'naive_parallel', 'contract_parallel')
        when 'llm_observability_detection_v1' then variant in ('fixed_threshold', 'latency_zscore', 'robust_multisignal')
        when 'video_temporal_grounding_v1' then variant in ('global_pool', 'frame_match', 'segment_smoothing')
        when 'mcp_registry_governance_v1' then variant in ('schema_only', 'schema_scope', 'full_governance')
        when 'speculative_decoding_v1' then variant in ('target_only', 'misaligned_draft', 'adaptive_draft')
        when 'constitutional_safety_gate_v1' then variant in ('keyword_gate', 'normalized_gate', 'context_policy_gate')
        when 'issue_patch_sandbox_v1' then variant in ('tests_only', 'path_policy', 'full_patch_policy')
        when 'adaptive_tutor_bkt_v1' then variant in ('linear_curriculum', 'weakest_first', 'bkt_spaced')
        else false
      end
    ),
  add constraint research_jobs_config_check
    check (
      (
        template_version = '2.0.0'
        and template_id = 'codebase_rag_retrieval_v1'
        and jsonb_typeof(config) = 'object'
      )
      or (
        template_version = '3.0.0'
        and jsonb_typeof(config) = 'object'
        and jsonb_typeof(config -> 'parameters') = 'object'
        and char_length(config ->> 'benchmarkVersion') between 1 and 100
        and config ->> 'method' = variant
        and config -> 'parameters' ->> 'offline' = 'true'
        and (config -> 'parameters' ->> 'capstoneNumber') ~ '^(1[0-7]|[1-9])$'
        and jsonb_typeof(config -> 'parameters' -> 'pipeline') = 'array'
      )
    );

comment on column public.research_runs.template_id is
  'Server-selected fixed offline template; arbitrary user code and configuration are not accepted.';

commit;
