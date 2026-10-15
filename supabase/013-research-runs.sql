-- 013 - ResearchRun P0 tables, RLS, indexes, and publication commands.
-- Apply in a non-production Supabase project first, then run the RLS and API
-- integration tests documented in server/research-api/README.md.
--
-- Trust boundary:
-- - authenticated and anon clients receive read-only access through RLS;
-- - the Research API owns state transitions and writes with service_role;
-- - publication is the only client-triggered write and goes through the
--   security-definer functions at the end of this file.

begin;

-- ---------------------------------------------------------------------------
-- Runs
-- ---------------------------------------------------------------------------

create table if not exists public.research_runs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.profiles (id) on delete cascade,
  idempotency_key text not null,
  create_request_sha256 text not null,
  topic text not null,
  research_question text not null,
  engine text not null default 'minimal',
  template_id text not null default 'codebase_rag_retrieval_v1',
  template_version text not null default '3.0.0',
  status text not null default 'draft',
  current_stage text not null default 'hypothesis',
  version integer not null default 1,
  budget jsonb not null default
    '{"cpuSecondsPerJob":120,"memoryMbPerJob":1024,"maxConcurrentJobs":2,"maxAttemptsPerVariant":2}'::jsonb,
  retry_count integer not null default 0,
  failure_code text,
  failure_message text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  completed_at timestamptz,
  cancelled_at timestamptz,

  constraint research_runs_idempotency_key_length
    check (char_length(idempotency_key) between 8 and 128),
  constraint research_runs_create_request_sha256_format
    check (create_request_sha256 ~ '^[0-9a-f]{64}$'),
  constraint research_runs_topic_length
    check (char_length(btrim(topic)) between 3 and 200),
  constraint research_runs_question_length
    check (char_length(btrim(research_question)) between 10 and 1000),
  constraint research_runs_engine_check
    check (engine in ('minimal')),
  constraint research_runs_template_check
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
  constraint research_runs_template_version_check
    check (template_version = '3.0.0'),
  constraint research_runs_status_check
    check (status in (
      'draft',
      'hypothesis_ready',
      'queued',
      'running',
      'analysis_ready',
      'analyzing',
      'verification_required',
      'verifying',
      'completed',
      'failed',
      'cancelled'
    )),
  constraint research_runs_stage_check
    check (current_stage in (
      'topic',
      'hypothesis',
      'experiment',
      'analysis',
      'verification',
      'publication'
    )),
  constraint research_runs_version_check check (version >= 1),
  constraint research_runs_budget_check
    check (
      budget =
        '{"cpuSecondsPerJob":120,"memoryMbPerJob":1024,"maxConcurrentJobs":2,"maxAttemptsPerVariant":2}'::jsonb
    ),
  constraint research_runs_retry_count_check check (retry_count >= 0),
  constraint research_runs_failure_pair_check
    check ((failure_code is null) = (failure_message is null)),
  constraint research_runs_terminal_time_check
    check (
      (status = 'completed' and completed_at is not null and cancelled_at is null)
      or (status = 'cancelled' and cancelled_at is not null and completed_at is null)
      or (status not in ('completed', 'cancelled') and completed_at is null and cancelled_at is null)
    ),
  constraint research_runs_user_idempotency_unique
    unique (user_id, idempotency_key),
  constraint research_runs_id_user_unique unique (id, user_id)
);

create index if not exists research_runs_user_created_idx
  on public.research_runs (user_id, created_at desc);

create index if not exists research_runs_status_updated_idx
  on public.research_runs (status, updated_at)
  where status not in ('completed', 'failed', 'cancelled');

comment on column public.research_runs.version is
  'Monotonic revision incremented per accepted command and run-state transition.';

comment on column public.research_runs.failure_message is
  'Safe user-facing summary only; never store stack traces or secrets.';

-- ---------------------------------------------------------------------------
-- One hypothesis per P0 run
-- ---------------------------------------------------------------------------

create table if not exists public.research_hypotheses (
  id uuid primary key default gen_random_uuid(),
  run_id uuid not null,
  user_id uuid not null,
  statement text not null,
  independent_variable text not null,
  dependent_metric text not null,
  expected_direction text not null,
  rationale text not null,
  status text not null default 'draft',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  accepted_at timestamptz,

  constraint research_hypotheses_run_owner_fk
    foreign key (run_id, user_id)
    references public.research_runs (id, user_id)
    on delete cascade,
  constraint research_hypotheses_statement_length
    check (char_length(btrim(statement)) between 10 and 1000),
  constraint research_hypotheses_independent_variable_length
    check (char_length(btrim(independent_variable)) between 1 and 200),
  constraint research_hypotheses_dependent_metric_length
    check (char_length(btrim(dependent_metric)) between 1 and 200),
  constraint research_hypotheses_direction_check
    check (expected_direction in ('increase', 'decrease', 'no_change')),
  constraint research_hypotheses_rationale_length
    check (char_length(btrim(rationale)) between 10 and 2000),
  constraint research_hypotheses_status_check
    check (status in ('draft', 'submitted', 'accepted', 'rejected')),
  constraint research_hypotheses_accepted_time_check
    check ((status = 'accepted') = (accepted_at is not null)),
  constraint research_hypotheses_one_per_run unique (run_id),
  constraint research_hypotheses_id_run_user_unique unique (id, run_id, user_id)
);

-- ---------------------------------------------------------------------------
-- Immutable experiment attempts
-- ---------------------------------------------------------------------------

create table if not exists public.research_experiment_jobs (
  id uuid primary key default gen_random_uuid(),
  run_id uuid not null,
  user_id uuid not null,
  hypothesis_id uuid not null,
  suite_id uuid not null,
  variant text not null,
  template_id text not null default 'codebase_rag_retrieval_v1',
  template_version text not null default '3.0.0',
  attempt integer not null default 1,
  status text not null default 'queued',
  bootstrap_seed integer not null default 20260711,
  bootstrap_replicates integer not null default 1000,
  config jsonb not null,
  resource_limits jsonb not null default
    '{"cpuCount":1,"memoryMb":1024,"timeoutSeconds":120,"network":false}'::jsonb,
  metrics jsonb,
  error_code text,
  error_message text,
  queued_at timestamptz not null default now(),
  started_at timestamptz,
  finished_at timestamptz,
  duration_ms integer,

  constraint research_jobs_run_owner_fk
    foreign key (run_id, user_id)
    references public.research_runs (id, user_id)
    on delete cascade,
  constraint research_jobs_hypothesis_owner_fk
    foreign key (hypothesis_id, run_id, user_id)
    references public.research_hypotheses (id, run_id, user_id)
    on delete restrict,
  constraint research_jobs_variant_check
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
  constraint research_jobs_template_check
    check (template_id in (
      'terminal_coding_agent_v1', 'codebase_rag_retrieval_v1', 'realtime_voice_vad_v1',
      'multimodal_document_retrieval_v1', 'autonomous_research_search_v1',
      'kubernetes_rca_v1', 'local_finetuning_v1', 'regulated_rag_acl_v1',
      'code_migration_v1', 'multi_agent_coordination_v1',
      'llm_observability_detection_v1', 'video_temporal_grounding_v1',
      'mcp_registry_governance_v1', 'speculative_decoding_v1',
      'constitutional_safety_gate_v1', 'issue_patch_sandbox_v1',
      'adaptive_tutor_bkt_v1'
    )),
  constraint research_jobs_template_version_check
    check (template_version = '3.0.0'),
  constraint research_jobs_attempt_check check (attempt between 1 and 2),
  constraint research_jobs_status_check
    check (status in (
      'queued', 'running', 'succeeded', 'failed', 'cancelled', 'timed_out'
    )),
  constraint research_jobs_bootstrap_check
    check (bootstrap_seed = 20260711 and bootstrap_replicates = 1000),
  constraint research_jobs_config_check
    check (
      jsonb_typeof(config) = 'object'
      and jsonb_typeof(config -> 'parameters') = 'object'
      and char_length(config ->> 'benchmarkVersion') between 1 and 100
      and config ->> 'method' = variant
      and config -> 'parameters' ->> 'offline' = 'true'
      and (config -> 'parameters' ->> 'capstoneNumber') ~ '^(1[0-7]|[1-9])$'
      and jsonb_typeof(config -> 'parameters' -> 'pipeline') = 'array'
    ),
  constraint research_jobs_resource_limits_check
    check (
      resource_limits =
        '{"cpuCount":1,"memoryMb":1024,"timeoutSeconds":120,"network":false}'::jsonb
    ),
  constraint research_jobs_metrics_object_check
    check (metrics is null or jsonb_typeof(metrics) = 'object'),
  constraint research_jobs_success_metrics_check
    check (status <> 'succeeded' or metrics is not null),
  constraint research_jobs_error_pair_check
    check ((error_code is null) = (error_message is null)),
  constraint research_jobs_error_status_check
    check (
      error_code is null
      or status in ('failed', 'cancelled', 'timed_out')
    ),
  constraint research_jobs_time_check
    check (
      (status = 'queued' and started_at is null and finished_at is null)
      or (status = 'running' and started_at is not null and finished_at is null)
      or (status in ('succeeded', 'failed', 'cancelled', 'timed_out') and finished_at is not null)
    ),
  constraint research_jobs_duration_check
    check (duration_ms is null or duration_ms >= 0),
  constraint research_jobs_run_variant_attempt_unique
    unique (run_id, variant, attempt),
  constraint research_jobs_suite_variant_unique unique (suite_id, variant),
  constraint research_jobs_id_run_user_unique unique (id, run_id, user_id)
);

create index if not exists research_jobs_queue_idx
  on public.research_experiment_jobs (queued_at, id)
  where status = 'queued';

create index if not exists research_jobs_run_created_idx
  on public.research_experiment_jobs (run_id, queued_at, id);

comment on column public.research_experiment_jobs.metrics is
  'Validated measured values only; deterministic analysis reads this field.';

-- ---------------------------------------------------------------------------
-- Immutable, versioned outputs
-- ---------------------------------------------------------------------------

create table if not exists public.research_artifacts (
  id uuid primary key default gen_random_uuid(),
  run_id uuid not null,
  user_id uuid not null,
  job_id uuid,
  stage text not null,
  kind text not null,
  version integer not null default 1,
  content jsonb,
  storage_bucket text,
  storage_path text,
  filename text,
  mime_type text,
  byte_size bigint,
  sha256 text,
  verification_status text not null default 'unverified',
  visibility text not null default 'private',
  created_at timestamptz not null default now(),
  published_at timestamptz,

  constraint research_artifacts_run_owner_fk
    foreign key (run_id, user_id)
    references public.research_runs (id, user_id)
    on delete cascade,
  constraint research_artifacts_job_owner_fk
    foreign key (job_id, run_id, user_id)
    references public.research_experiment_jobs (id, run_id, user_id)
    on delete restrict,
  constraint research_artifacts_stage_check
    check (stage in (
      'topic',
      'hypothesis',
      'experiment',
      'analysis',
      'verification',
      'publication'
    )),
  constraint research_artifacts_kind_check
    check (kind in (
      'topic_brief',
      'hypothesis',
      'experiment_spec',
      'source_bundle',
      'experiment_log',
      'raw_metrics',
      'analysis',
      'verification_bundle',
      'research_card'
    )),
  constraint research_artifacts_version_check check (version >= 1),
  constraint research_artifacts_content_object_check
    check (content is null or jsonb_typeof(content) = 'object'),
  constraint research_artifacts_storage_pair_check
    check ((storage_bucket is null) = (storage_path is null)),
  constraint research_artifacts_payload_check
    check (content is not null or storage_path is not null),
  constraint research_artifacts_byte_size_check
    check (byte_size is null or byte_size >= 0),
  constraint research_artifacts_sha256_format
    check (sha256 is null or sha256 ~ '^[0-9a-f]{64}$'),
  constraint research_artifacts_file_sha256_check
    check (storage_path is null or sha256 is not null),
  constraint research_artifacts_verification_check
    check (verification_status in ('unverified', 'passed', 'failed')),
  constraint research_artifacts_visibility_check
    check (visibility in ('private', 'public')),
  constraint research_artifacts_publication_time_check
    check (
      (visibility = 'private' and published_at is null)
      or (visibility = 'public' and published_at is not null)
    ),
  constraint research_artifacts_public_safety_check
    check (
      visibility = 'private'
      or (
        kind in ('verification_bundle', 'research_card')
        and verification_status = 'passed'
        and content is not null
        and storage_bucket is null
        and storage_path is null
      )
    )
);

create unique index if not exists research_artifacts_run_kind_version_unique
  on public.research_artifacts (run_id, kind, version)
  where job_id is null;

create unique index if not exists research_artifacts_job_kind_version_unique
  on public.research_artifacts (job_id, kind, version)
  where job_id is not null;

create index if not exists research_artifacts_run_created_idx
  on public.research_artifacts (run_id, created_at, id);

create index if not exists research_artifacts_public_idx
  on public.research_artifacts (published_at desc, id)
  where visibility = 'public' and verification_status = 'passed';

comment on constraint research_artifacts_public_safety_check
  on public.research_artifacts is
  'Public rows must be verified, publication-safe JSON with no Storage pointer.';

-- ---------------------------------------------------------------------------
-- Append-only audit and SSE stream
-- ---------------------------------------------------------------------------

create table if not exists public.research_events (
  id bigint generated always as identity primary key,
  run_id uuid not null,
  user_id uuid not null,
  actor text not null,
  event_type text not null,
  stage text not null,
  command_name text,
  idempotency_key text,
  request_sha256 text,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),

  constraint research_events_run_owner_fk
    foreign key (run_id, user_id)
    references public.research_runs (id, user_id)
    on delete cascade,
  constraint research_events_actor_check
    check (actor in ('user', 'runner', 'verifier', 'system')),
  constraint research_events_type_length
    check (char_length(btrim(event_type)) between 3 and 100),
  constraint research_events_stage_check
    check (stage in (
      'topic',
      'hypothesis',
      'experiment',
      'analysis',
      'verification',
      'publication'
    )),
  constraint research_events_command_triplet_check
    check (
      (command_name is null and idempotency_key is null and request_sha256 is null)
      or (
        command_name is not null
        and char_length(btrim(command_name)) between 3 and 100
        and char_length(idempotency_key) between 8 and 128
        and request_sha256 ~ '^[0-9a-f]{64}$'
      )
    ),
  constraint research_events_payload_object_check
    check (jsonb_typeof(payload) = 'object')
);

create unique index if not exists research_events_command_idempotency_unique
  on public.research_events (user_id, command_name, idempotency_key)
  where idempotency_key is not null;

create index if not exists research_events_run_cursor_idx
  on public.research_events (run_id, id);

comment on column public.research_events.id is
  'Monotonic per-database SSE cursor returned as a JSON string.';

comment on column public.research_events.payload is
  'Safe event projection only; never store secrets, stack traces, or raw logs.';

-- ---------------------------------------------------------------------------
-- updated_at triggers
-- ---------------------------------------------------------------------------

create or replace function public.set_research_updated_at()
returns trigger
language plpgsql
set search_path = public
as $$
begin
  new.updated_at := now();
  return new;
end;
$$;

drop trigger if exists research_runs_set_updated_at on public.research_runs;
create trigger research_runs_set_updated_at
  before update on public.research_runs
  for each row execute function public.set_research_updated_at();

drop trigger if exists research_hypotheses_set_updated_at on public.research_hypotheses;
create trigger research_hypotheses_set_updated_at
  before update on public.research_hypotheses
  for each row execute function public.set_research_updated_at();

-- ---------------------------------------------------------------------------
-- RLS: clients read; Research API writes with service_role
-- ---------------------------------------------------------------------------

alter table public.research_runs enable row level security;
alter table public.research_hypotheses enable row level security;
alter table public.research_experiment_jobs enable row level security;
alter table public.research_artifacts enable row level security;
alter table public.research_events enable row level security;

drop policy if exists research_runs_select_owner_admin on public.research_runs;
create policy research_runs_select_owner_admin
  on public.research_runs
  for select
  using (auth.uid() = user_id or public.is_admin());

drop policy if exists research_hypotheses_select_owner_admin on public.research_hypotheses;
create policy research_hypotheses_select_owner_admin
  on public.research_hypotheses
  for select
  using (auth.uid() = user_id or public.is_admin());

drop policy if exists research_jobs_select_owner_admin on public.research_experiment_jobs;
create policy research_jobs_select_owner_admin
  on public.research_experiment_jobs
  for select
  using (auth.uid() = user_id or public.is_admin());

drop policy if exists research_artifacts_select_owner_admin_or_public on public.research_artifacts;
create policy research_artifacts_select_owner_admin_or_public
  on public.research_artifacts
  for select
  using (
    auth.uid() = user_id
    or public.is_admin()
    or (
      visibility = 'public'
      and verification_status = 'passed'
      and kind in ('verification_bundle', 'research_card')
      and storage_path is null
    )
  );

drop policy if exists research_events_select_owner_admin on public.research_events;
create policy research_events_select_owner_admin
  on public.research_events
  for select
  using (auth.uid() = user_id or public.is_admin());

-- No INSERT, UPDATE, or DELETE policies are intentionally defined.
revoke all on table public.research_runs from anon, authenticated;
revoke all on table public.research_hypotheses from anon, authenticated;
revoke all on table public.research_experiment_jobs from anon, authenticated;
revoke all on table public.research_artifacts from anon, authenticated;
revoke all on table public.research_events from anon, authenticated;

grant select on table public.research_runs to authenticated;
grant select on table public.research_hypotheses to authenticated;
grant select on table public.research_experiment_jobs to authenticated;
grant select on table public.research_artifacts to authenticated;
grant select (
  id,
  kind,
  version,
  content,
  sha256,
  verification_status,
  visibility,
  created_at,
  published_at
) on table public.research_artifacts to anon;
grant select on table public.research_events to authenticated;

grant select, insert, update, delete on table public.research_runs to service_role;
grant select, insert, update, delete on table public.research_hypotheses to service_role;
grant select, insert, update, delete on table public.research_experiment_jobs to service_role;
grant select, insert, update, delete on table public.research_artifacts to service_role;
grant select, insert, update, delete on table public.research_events to service_role;
grant usage, select on sequence public.research_events_id_seq to service_role;

-- ---------------------------------------------------------------------------
-- Explicit publication commands
--
-- These functions are API-internal atomic commands. The API derives p_user_id
-- from the validated JWT subject. Direct browser execution is revoked.
-- ---------------------------------------------------------------------------

create or replace function public.publish_research_artifact(
  p_user_id uuid,
  p_artifact_id uuid,
  p_expected_run_version integer,
  p_idempotency_key text,
  p_request_sha256 text
)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
  existing_event public.research_events;
  target public.research_artifacts;
  current_run public.research_runs;
  next_run_version integer;
  inserted_event_id bigint;
  event_payload jsonb;
begin
  if p_user_id is null
     or p_artifact_id is null
     or p_expected_run_version is null
     or p_expected_run_version < 1
     or p_idempotency_key is null
     or char_length(p_idempotency_key) not between 8 and 128
     or p_request_sha256 is null
     or p_request_sha256 !~ '^[0-9a-f]{64}$' then
    raise exception 'invalid publication command' using errcode = '22023';
  end if;

  perform pg_advisory_xact_lock(
    hashtextextended(
      p_user_id::text || ':artifact.publish:' || p_idempotency_key,
      0
    )
  );

  select * into existing_event
  from public.research_events
  where user_id = p_user_id
    and command_name = 'artifact.publish'
    and idempotency_key = p_idempotency_key;

  if found then
    if existing_event.request_sha256 <> p_request_sha256 then
      raise exception 'idempotency conflict' using errcode = '23505';
    end if;
    return existing_event.payload
      || jsonb_build_object('eventId', existing_event.id::text);
  end if;

  select * into target
  from public.research_artifacts
  where id = p_artifact_id and user_id = p_user_id
  for update;

  if not found then
    raise exception 'artifact not found' using errcode = 'P0002';
  end if;

  if target.visibility <> 'private'
     or target.kind not in ('verification_bundle', 'research_card')
     or target.verification_status <> 'passed'
     or target.content is null
     or target.storage_bucket is not null
     or target.storage_path is not null then
    raise exception 'artifact is not publishable' using errcode = '22023';
  end if;

  select * into current_run
  from public.research_runs
  where id = target.run_id and user_id = p_user_id
  for update;

  if current_run.status <> 'completed' then
    raise exception 'run is not completed' using errcode = '22023';
  end if;

  if current_run.version <> p_expected_run_version then
    raise exception 'version conflict' using errcode = '40001';
  end if;

  next_run_version := current_run.version + 1;

  update public.research_artifacts
  set visibility = 'public', published_at = now()
  where id = target.id;

  update public.research_runs
  set version = next_run_version
  where id = target.run_id and user_id = p_user_id;

  event_payload := jsonb_build_object(
    'artifactId', target.id,
    'visibility', 'public',
    'runVersion', next_run_version
  );

  insert into public.research_events (
    run_id,
    user_id,
    actor,
    event_type,
    stage,
    command_name,
    idempotency_key,
    request_sha256,
    payload
  ) values (
    target.run_id,
    p_user_id,
    'user',
    'artifact.published',
    'publication',
    'artifact.publish',
    p_idempotency_key,
    p_request_sha256,
    event_payload
  )
  returning id into inserted_event_id;

  return event_payload || jsonb_build_object('eventId', inserted_event_id::text);
end;
$$;

create or replace function public.unpublish_research_artifact(
  p_user_id uuid,
  p_artifact_id uuid,
  p_expected_run_version integer,
  p_idempotency_key text,
  p_request_sha256 text
)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
  existing_event public.research_events;
  target public.research_artifacts;
  current_run_version integer;
  next_run_version integer;
  inserted_event_id bigint;
  event_payload jsonb;
begin
  if p_user_id is null
     or p_artifact_id is null
     or p_expected_run_version is null
     or p_expected_run_version < 1
     or p_idempotency_key is null
     or char_length(p_idempotency_key) not between 8 and 128
     or p_request_sha256 is null
     or p_request_sha256 !~ '^[0-9a-f]{64}$' then
    raise exception 'invalid publication command' using errcode = '22023';
  end if;

  perform pg_advisory_xact_lock(
    hashtextextended(
      p_user_id::text || ':artifact.unpublish:' || p_idempotency_key,
      0
    )
  );

  select * into existing_event
  from public.research_events
  where user_id = p_user_id
    and command_name = 'artifact.unpublish'
    and idempotency_key = p_idempotency_key;

  if found then
    if existing_event.request_sha256 <> p_request_sha256 then
      raise exception 'idempotency conflict' using errcode = '23505';
    end if;
    return existing_event.payload
      || jsonb_build_object('eventId', existing_event.id::text);
  end if;

  select * into target
  from public.research_artifacts
  where id = p_artifact_id and user_id = p_user_id
  for update;

  if not found then
    raise exception 'artifact not found' using errcode = 'P0002';
  end if;

  if target.visibility <> 'public' then
    raise exception 'artifact is not published' using errcode = '22023';
  end if;

  select version into current_run_version
  from public.research_runs
  where id = target.run_id and user_id = p_user_id
  for update;

  if current_run_version <> p_expected_run_version then
    raise exception 'version conflict' using errcode = '40001';
  end if;

  next_run_version := current_run_version + 1;

  update public.research_artifacts
  set visibility = 'private', published_at = null
  where id = target.id;

  update public.research_runs
  set version = next_run_version
  where id = target.run_id and user_id = p_user_id;

  event_payload := jsonb_build_object(
    'artifactId', target.id,
    'visibility', 'private',
    'runVersion', next_run_version
  );

  insert into public.research_events (
    run_id,
    user_id,
    actor,
    event_type,
    stage,
    command_name,
    idempotency_key,
    request_sha256,
    payload
  ) values (
    target.run_id,
    p_user_id,
    'user',
    'artifact.unpublished',
    'publication',
    'artifact.unpublish',
    p_idempotency_key,
    p_request_sha256,
    event_payload
  )
  returning id into inserted_event_id;

  return event_payload || jsonb_build_object('eventId', inserted_event_id::text);
end;
$$;

revoke all on function public.publish_research_artifact(uuid, uuid, integer, text, text)
  from public, anon, authenticated;
revoke all on function public.unpublish_research_artifact(uuid, uuid, integer, text, text)
  from public, anon, authenticated;
grant execute on function public.publish_research_artifact(uuid, uuid, integer, text, text)
  to service_role;
grant execute on function public.unpublish_research_artifact(uuid, uuid, integer, text, text)
  to service_role;

-- ---------------------------------------------------------------------------
-- Private Storage bucket
-- ---------------------------------------------------------------------------

insert into storage.buckets (id, name, public)
values ('research-artifacts', 'research-artifacts', false)
on conflict (id) do update set public = false;

-- Do not add authenticated/anon policies for this bucket. The Research API
-- writes objects with service_role and returns short-lived signed URLs only
-- after an owner/admin authorization check.

commit;
