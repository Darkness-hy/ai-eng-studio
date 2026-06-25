-- ════════════════════════════════════════════════════════════════════════════
-- 009 — 毕业设计 Spark 账户开户队列
--   学员在校园网内的 Spark 机器上做毕业设计,需要 Linux 系统账户。助教后端在校园
--   网外、连不到 Spark,所以开户不能由它直接执行。
--   架构:学员先加入「Spark 使用班级」→ 对 AI 助教说「申请spark账号」→ 落一条申请到本表 →
--   Spark 上的 spark-agent 守护进程「出站」轮询本表 → 校验该学员确在 Spark 班级 →
--   跑固定脚本 useradd → 回写凭据(host/port/用户名/临时密码)。
--   资格闸门 = Spark 班级成员身份(由 agent 用 service-role 校验 class_members);'approved'
--   状态保留给管理员手动放行/复核,日常不需要。
--   LLM 不碰特权路径;真正 useradd 由 Spark 内、只认固定脚本的 agent 执行。
--   在 Supabase Dashboard → SQL Editor 整段执行(幂等)。关联 schema.sql(is_admin)。
-- ════════════════════════════════════════════════════════════════════════════

create table if not exists public.spark_accounts (
  user_id        uuid primary key references auth.users (id) on delete cascade,  -- 一人一号
  status         text not null default 'requested'
                   check (status in ('requested','approved','provisioning','ready','failed','revoked')),
  ssh_username   text,        -- 由 agent 派生分配
  temp_password  text,        -- 一次性临时密码;首次登录强制改密,窗口天然受限
  password_seen  boolean not null default false,
  host           text,        -- SSH 主机/IP,如 114.212.174.38
  ssh_port       integer,     -- SSH 端口,如 10148
  requested_username text     -- 学员自报的小写全拼用户名(如 dinghongyu);agent 用它做账户名
                   check (requested_username is null or requested_username ~ '^[a-z][a-z0-9]{1,31}$'),
  note           text,        -- 学员可选备注
  error          text,        -- 开户失败原因(给学员看的通用文案,细节落 agent 日志)
  requested_at   timestamptz not null default now(),
  approved_by    uuid references auth.users (id),
  approved_at    timestamptz,
  provisioned_at timestamptz
);

alter table public.spark_accounts enable row level security;

-- 学员:只能为自己申请,且 status 强制为 requested、不能预填任何特权字段。
drop policy if exists spark_insert on public.spark_accounts;
create policy spark_insert on public.spark_accounts for insert
  with check (
    user_id = auth.uid()
    and status = 'requested'
    and ssh_username is null and temp_password is null
    and approved_by is null and approved_at is null and provisioned_at is null
  );

-- 读:本人读自己,管理员读全部。
drop policy if exists spark_select on public.spark_accounts;
create policy spark_select on public.spark_accounts for select
  using (user_id = auth.uid() or public.is_admin());

-- 写:仅管理员(审批/撤销)。状态机的 provisioning/ready/failed 由 agent 走 service-role
--     越过 RLS 写,不开放给任何前端角色。
drop policy if exists spark_update_admin on public.spark_accounts;
create policy spark_update_admin on public.spark_accounts for update
  using (public.is_admin()) with check (public.is_admin());

-- 删:本人可撤回尚未批准的申请;管理员可删任意。
drop policy if exists spark_delete on public.spark_accounts;
create policy spark_delete on public.spark_accounts for delete
  using ((user_id = auth.uid() and status = 'requested') or public.is_admin());

-- 说明:学员没有任何 UPDATE 策略 → 改不了自己行的任何字段(无法自我审批/改用户名);
-- 仅 is_admin() 可 UPDATE(审批);provisioning/ready/failed 由 agent 走 service-role 写。
-- (不做列级 revoke:authenticated 含管理员,会连带挡掉 admin 审批。)

-- 管理员审批示例(在 Dashboard 手动,或由 admin 前端调用):
--   update public.spark_accounts set status='approved', approved_by=auth.uid(), approved_at=now()
--   where user_id = '<学员 uid>' and status = 'requested';
