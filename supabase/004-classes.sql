-- 班级 / 小组（教师端）。在 Supabase Dashboard → SQL Editor 整段执行一次。
-- 任何登录用户都可创建班级（成为该班老师），学生用邀请码加入；
-- 班级老师可查看本班成员的学习进度、定级与答题数据。
-- 幂等：可重复执行（修正了早期版本的 RLS 无限递归）。

-- ── 班级 + 成员 ─────────────────────────────────────────────────────
create table if not exists public.classes (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  owner_id uuid not null references public.profiles (id) on delete cascade,
  invite_code text not null unique,
  created_at timestamptz not null default now()
);

create table if not exists public.class_members (
  class_id uuid not null references public.classes (id) on delete cascade,
  user_id uuid not null references public.profiles (id) on delete cascade,
  joined_at timestamptz not null default now(),
  primary key (class_id, user_id)
);

-- ── security-definer helpers (bypass the OTHER table's RLS → no recursion) ──
create or replace function public.is_member_of(cid uuid)
returns boolean language sql stable security definer set search_path = public as $$
  select exists (select 1 from public.class_members where class_id = cid and user_id = auth.uid());
$$;

create or replace function public.is_owner_of(cid uuid)
returns boolean language sql stable security definer set search_path = public as $$
  select exists (select 1 from public.classes where id = cid and owner_id = auth.uid());
$$;

create or replace function public.owns_class_of(student uuid)
returns boolean language sql stable security definer set search_path = public as $$
  select exists (
    select 1 from public.class_members cm
    join public.classes c on c.id = cm.class_id
    where cm.user_id = student and c.owner_id = auth.uid()
  );
$$;

-- ── 凭邀请码加入班级 ────────────────────────────────────────────────
create or replace function public.join_class(code text)
returns uuid language plpgsql security definer set search_path = public as $$
declare cid uuid;
begin
  select id into cid from public.classes where invite_code = code;
  if cid is null then raise exception 'invalid invite code'; end if;
  insert into public.class_members (class_id, user_id) values (cid, auth.uid()) on conflict do nothing;
  return cid;
end;
$$;

-- ── RLS ─────────────────────────────────────────────────────────────
alter table public.classes enable row level security;
alter table public.class_members enable row level security;

drop policy if exists classes_select on public.classes;
create policy classes_select on public.classes for select
  using (owner_id = auth.uid() or public.is_member_of(id) or public.is_admin());

-- Any authenticated user may create their own class (they become its owner/teacher).
-- (For DBs that already ran the admin-only version, see 008-open-class-creation.sql.)
drop policy if exists classes_insert on public.classes;
create policy classes_insert on public.classes for insert
  with check (owner_id = auth.uid());

drop policy if exists classes_delete on public.classes;
create policy classes_delete on public.classes for delete using (owner_id = auth.uid());

drop policy if exists class_members_select on public.class_members;
create policy class_members_select on public.class_members for select
  using (user_id = auth.uid() or public.is_owner_of(class_id) or public.is_admin());

drop policy if exists class_members_insert on public.class_members;
create policy class_members_insert on public.class_members for insert with check (user_id = auth.uid());

drop policy if exists class_members_delete on public.class_members;
create policy class_members_delete on public.class_members for delete
  using (user_id = auth.uid() or public.is_owner_of(class_id));

-- ── 让班级老师能读到本班成员的学习数据 ──────────────────────────────
drop policy if exists profiles_select on public.profiles;
create policy profiles_select on public.profiles for select
  using (auth.uid() = id or public.is_admin() or public.owns_class_of(id));

drop policy if exists progress_select on public.progress;
create policy progress_select on public.progress for select
  using (auth.uid() = user_id or public.is_admin() or public.owns_class_of(user_id));

drop policy if exists activity_select on public.activity;
create policy activity_select on public.activity for select
  using (auth.uid() = user_id or public.is_admin() or public.owns_class_of(user_id));

drop policy if exists placement_select on public.placement;
create policy placement_select on public.placement for select
  using (auth.uid() = user_id or public.is_admin() or public.owns_class_of(user_id));
