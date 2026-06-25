-- ════════════════════════════════════════════════════════════════════════════
-- 011 — 允许学员撤回自己「失败/已撤销」的 Spark 申请(以便重试)
--   009 的 spark_delete 只允许删 status='requested' 的行,导致开户失败后学员无法
--   「撤回并重试」。放宽为:本人可删自己 requested/failed/revoked 的行;provisioning/
--   ready 仍只有管理员能删(已开出来的账户不应被学员随手清掉)。幂等。
-- ════════════════════════════════════════════════════════════════════════════

drop policy if exists spark_delete on public.spark_accounts;
create policy spark_delete on public.spark_accounts for delete
  using (
    (user_id = auth.uid() and status in ('requested', 'failed', 'revoked'))
    or public.is_admin()
  );
