-- 班级作业。在 Supabase Dashboard → SQL Editor 整段执行一次。
-- 老师(班级 owner)给班级布置作业(一组课 + 截止日);完成判定复用 progress
-- (学生完成作业指定的全部课 = 完成该作业),无需单独的提交表。

create table if not exists public.assignments (
  id uuid primary key default gen_random_uuid(),
  class_id uuid not null references public.classes (id) on delete cascade,
  title text not null,
  lesson_ids text[] not null default '{}',
  due_date date,
  created_at timestamptz not null default now()
);

alter table public.assignments enable row level security;

-- 班级成员与老师可见;只有老师可增删。
drop policy if exists assignments_select on public.assignments;
create policy assignments_select on public.assignments for select
  using (public.is_member_of(class_id) or public.is_owner_of(class_id) or public.is_admin());

drop policy if exists assignments_insert on public.assignments;
create policy assignments_insert on public.assignments for insert
  with check (public.is_owner_of(class_id));

drop policy if exists assignments_delete on public.assignments;
create policy assignments_delete on public.assignments for delete
  using (public.is_owner_of(class_id));
