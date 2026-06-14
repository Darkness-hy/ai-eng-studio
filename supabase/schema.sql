-- ai-eng-studio cloud sync schema
-- 在 Supabase Dashboard → SQL Editor 里整段执行一次即可。
-- 另外请在 Authentication → Sign In / Providers → Email 中关闭 "Confirm email"（免验证注册）。

-- ── 用户档案 ─────────────────────────────────────────────────────────
create table if not exists public.profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  email text not null,
  display_name text,
  role text not null default 'student' check (role in ('student', 'admin')),
  created_at timestamptz not null default now()
);

-- ── 学习进度（每人每课一行）──────────────────────────────────────────
create table if not exists public.progress (
  user_id uuid not null references public.profiles (id) on delete cascade,
  lesson_id text not null,
  done boolean not null default false,
  pre_score int,
  pre_total int,
  post_score int,
  post_total int,
  completed_at timestamptz,
  updated_at timestamptz not null default now(),
  primary key (user_id, lesson_id)
);

-- ── 每日活跃（streak 与活跃度图表）───────────────────────────────────
create table if not exists public.activity (
  user_id uuid not null references public.profiles (id) on delete cascade,
  day date not null,
  primary key (user_id, day)
);

-- ── 管理员判定（security definer 避免 RLS 自递归）────────────────────
create or replace function public.is_admin()
returns boolean
language sql stable security definer
set search_path = public
as $$
  select exists (
    select 1 from public.profiles
    where id = auth.uid() and role = 'admin'
  );
$$;

-- ── 注册时自动建档；指定邮箱自动成为管理员 ───────────────────────────
create or replace function public.handle_new_user()
returns trigger
language plpgsql security definer
set search_path = public
as $$
begin
  insert into public.profiles (id, email, display_name, role)
  values (
    new.id,
    new.email,
    coalesce(new.raw_user_meta_data ->> 'display_name', split_part(new.email, '@', 1)),
    -- 替换为你自己的管理员邮箱(可多个);生产环境建议保持邮箱验证开启
    case when new.email in ('admin@example.com') then 'admin' else 'student' end
  );
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- ── 行级权限：学生只碰自己的数据，admin 全量只读 ─────────────────────
alter table public.profiles enable row level security;
alter table public.progress enable row level security;
alter table public.activity enable row level security;

drop policy if exists profiles_select on public.profiles;
create policy profiles_select on public.profiles
  for select using (auth.uid() = id or public.is_admin());

drop policy if exists profiles_update_own on public.profiles;
create policy profiles_update_own on public.profiles
  for update using (auth.uid() = id) with check (auth.uid() = id and role = role);

drop policy if exists progress_select on public.progress;
create policy progress_select on public.progress
  for select using (auth.uid() = user_id or public.is_admin());

drop policy if exists progress_write on public.progress;
create policy progress_write on public.progress
  for insert with check (auth.uid() = user_id);

drop policy if exists progress_update on public.progress;
create policy progress_update on public.progress
  for update using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists activity_select on public.activity;
create policy activity_select on public.activity
  for select using (auth.uid() = user_id or public.is_admin());

drop policy if exists activity_write on public.activity;
create policy activity_write on public.activity
  for insert with check (auth.uid() = user_id);

-- ── 后续手动添加管理员 ──────────────────────────────────────────────
-- update public.profiles set role = 'admin' where email = 'someone@example.com';
-- 定级测试结果云同步表：在 Supabase Dashboard → SQL Editor 执行一次。
create table if not exists public.placement (
  user_id uuid primary key references public.profiles (id) on delete cascade,
  answers int[] not null,
  area_scores jsonb not null,
  total int not null,
  entry int not null,
  taken_at timestamptz not null default now()
);
alter table public.placement enable row level security;
drop policy if exists placement_select on public.placement;
create policy placement_select on public.placement for select using (auth.uid() = user_id or public.is_admin());
drop policy if exists placement_write on public.placement;
create policy placement_write on public.placement for insert with check (auth.uid() = user_id);
drop policy if exists placement_update on public.placement;
create policy placement_update on public.placement for update using (auth.uid() = user_id) with check (auth.uid() = user_id);
drop policy if exists placement_delete on public.placement;
create policy placement_delete on public.placement for delete using (auth.uid() = user_id);
