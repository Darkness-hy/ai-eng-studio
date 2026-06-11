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
