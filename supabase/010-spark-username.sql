-- ════════════════════════════════════════════════════════════════════════════
-- 010 — 学员自报用户名(小写全拼)作为 Spark 账户名
--   流程:学员对 AI 助教说「申请spark账号」→ 助教问他名字的小写全拼(如 dinghongyu)→
--   该拼音作为开户用户名写入 requested_username;Spark 上的 agent 用它做 useradd 的账户名。
--   给「已经执行过 009」的库补这一列;新库的 009 已自带本列。幂等。
-- ════════════════════════════════════════════════════════════════════════════

alter table public.spark_accounts
  add column if not exists requested_username text;

-- 格式约束:小写字母开头、仅小写字母/数字、2–32 位(挡住 shell 特殊字符与注入)。
alter table public.spark_accounts
  drop constraint if exists spark_username_fmt;
alter table public.spark_accounts
  add constraint spark_username_fmt
  check (requested_username is null or requested_username ~ '^[a-z][a-z0-9]{1,31}$');

-- 说明:学员 insert 时可自带 requested_username(非特权列,spark_insert 策略不禁止);
--   ssh_username(最终分配的账户名,可能因重名加后缀)仍由 agent 走 service-role 回写。
