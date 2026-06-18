-- ════════════════════════════════════════════════════════════════════════════
-- 008 — 放开班级创建:任意登录用户都可独立创建自己的班级
--   背景:f1be36a 曾把建班限制为 admin(004-classes.sql 的 classes_insert
--   要求 public.is_admin())。现恢复为「人人可建」——创建者即该班 owner/老师。
--   幂等:可在 Supabase Dashboard → SQL Editor 重复执行。
--   关联:004-classes.sql(classes 表 + owns_class_of 看板策略仍按 owner 归属,
--   与 admin 无关,无需改动);入班仍走 join_class() RPC,不受影响。
-- ════════════════════════════════════════════════════════════════════════════

drop policy if exists classes_insert on public.classes;
create policy classes_insert on public.classes for insert
  with check (owner_id = auth.uid());

-- 校验(可选):应仅剩 owner_id = auth.uid() 这一条 with check,不再含 is_admin()。
--   select polname, pg_get_expr(polwithcheck, polrelid) as with_check
--   from pg_policy where polname = 'classes_insert';
