-- 安全加固。在 Supabase Dashboard → SQL Editor 整段执行一次。幂等:可重复执行。
-- 修复 2026-06-14 安全审计发现的提权(P0)与越权(P1)缺口。
-- 关联:schema.sql(profiles/is_admin)、004-classes.sql(class_members/join_class)。

-- ════════════════════════════════════════════════════════════════════════════
-- P0 [Critical] profiles.role 自我提权
--   旧策略 profiles_update_own 的 with check (auth.uid()=id and role = role) 里,
--   role = role 是恒真式 —— Postgres RLS 表达式只能引用待写入的 NEW 行,无法引用 OLD,
--   故 role = role 实为 NEW.role = NEW.role,且 role 为 NOT NULL,恒为 TRUE。
--   结果:任意登录学生可 update 自己的 role='admin',is_admin() 随即放行,
--   越权读取全站所有用户的 profiles/progress/activity/placement/tutor_messages。
--   列不可变性 RLS 表达不了,必须用触发器 + 列级权限。
-- ════════════════════════════════════════════════════════════════════════════

-- (1) 触发器:阻止用户自助修改自己的 role。
--     特权上下文(Dashboard SQL Editor / service_role 后台)的 auth.uid() 为 null,
--     或操作的不是本人行,均不受影响 —— 管理员仍可正常授/撤权。
create or replace function public.lock_profile_role()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  if new.role is distinct from old.role
     and auth.uid() is not null
     and auth.uid() = old.id then
    new.role := old.role;          -- 静默忽略自我提权尝试
  end if;
  return new;
end;
$$;

drop trigger if exists profiles_lock_role on public.profiles;
create trigger profiles_lock_role
  before update on public.profiles
  for each row execute function public.lock_profile_role();

-- (2) 去掉无效的恒真式,把策略意图写清楚(role 不变性已由上面的触发器保证)。
drop policy if exists profiles_update_own on public.profiles;
create policy profiles_update_own on public.profiles
  for update using (auth.uid() = id) with check (auth.uid() = id);

-- (3) 纵深防御:数据库层撤销普通角色对 profiles 的整表 UPDATE,只重授 display_name。
--     前端从不直接 update profiles(已核实:仅 select),故此操作不影响任何现有功能;
--     profiles 行由 handle_new_user() 触发器创建(security definer,不受此限)。
revoke update on public.profiles from anon, authenticated;
grant  update (display_name) on public.profiles to authenticated;

-- (4) 存量审计:执行后人工核对下面这条查询的结果。
--     只应出现 handle_new_user() 白名单里的三个邮箱;其余任何 admin 都是被旧漏洞
--     提权出来的,需立即 `update public.profiles set role='student' where id = '<uid>'` 降权。
--   select id, email, role from public.profiles where role = 'admin';


-- ════════════════════════════════════════════════════════════════════════════
-- P1 [Medium] class_members 绕过邀请码自插任意班级
--   class_members_insert 的 with check 只校验 user_id = auth.uid(),不校验 class_id,
--   学生拿到任意 class_id 即可直接 INSERT 成为成员,绕过 join_class() 的邀请码校验。
--   修复:入班统一走 join_class()(security definer,以函数属主身份写入,不受 RLS/表权限限制),
--   撤掉对普通角色的直接 INSERT 能力。前端从不直接 insert class_members(已核实:仅 rpc 加入、select、delete)。
-- ════════════════════════════════════════════════════════════════════════════

drop policy if exists class_members_insert on public.class_members;
revoke insert on public.class_members from anon, authenticated;
-- 说明:RLS 开启后无 INSERT 策略即拒绝直接写入;revoke 为纵深防御,亦防未来误加宽松策略。
-- join_class(code)(004-classes.sql)与退课 class_members_delete 均不受影响。


-- ════════════════════════════════════════════════════════════════════════════
-- 需人工处理(非本脚本可完成)
-- ════════════════════════════════════════════════════════════════════════════
-- [需确认] schema.sql 注释要求关闭邮箱验证(免验证注册),而 handle_new_user() 按硬编码邮箱
--   自动授予 admin。若三个 admin 邮箱中任一尚未被本人注册占用,攻击者可用该邮箱免验证注册
--   即得 admin。请确认:
--     a) handle_new_user 白名单里你自己的管理员邮箱均已被本人注册并锁定;且
--     b) 在 Authentication → Providers → Email 重新开启 "Confirm email",或将 admin 引导
--        改为手动 update / 独立 admins 表(见下)。
--
-- [可选根治] 把管理员判定的信任根与用户可写数据物理隔离(防止 role 列任何残余写路径再次成为提权点):
--   create table if not exists public.admins (
--     user_id uuid primary key references auth.users (id) on delete cascade
--   );
--   alter table public.admins enable row level security;     -- 不给 authenticated/anon 任何写策略
--   create or replace function public.is_admin()
--   returns boolean language sql stable security definer set search_path = public as $$
--     select exists (select 1 from public.admins where user_id = auth.uid());
--   $$;
--   -- 并把 handle_new_user() 中按邮箱授 admin 的分支改为 insert into public.admins(user_id);
--   -- 迁移现有 admin:insert into public.admins (user_id) select id from public.profiles where role='admin';
