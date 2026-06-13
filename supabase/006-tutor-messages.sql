-- AI 助教对话留存。在 Supabase Dashboard → SQL Editor 整段执行一次。
-- 每一轮问答(用户问 + 助教答)各存一行,按用户隔离;管理员可读全部(用于分析)。

create table if not exists public.tutor_messages (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  lesson_id text,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  created_at timestamptz not null default now()
);

create index if not exists tutor_messages_user_idx on public.tutor_messages (user_id, created_at desc);
create index if not exists tutor_messages_lesson_idx on public.tutor_messages (lesson_id);

alter table public.tutor_messages enable row level security;

-- 本人可读自己的;管理员可读全部。
drop policy if exists tutor_messages_select on public.tutor_messages;
create policy tutor_messages_select on public.tutor_messages
  for select using (auth.uid() = user_id or public.is_admin());

-- 只能写自己的。
drop policy if exists tutor_messages_insert on public.tutor_messages;
create policy tutor_messages_insert on public.tutor_messages
  for insert with check (auth.uid() = user_id);

-- 本人可删自己的(清空对话历史)。
drop policy if exists tutor_messages_delete on public.tutor_messages;
create policy tutor_messages_delete on public.tutor_messages
  for delete using (auth.uid() = user_id);
