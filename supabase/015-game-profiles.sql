-- 015 - Persistent player identity for the Research Island P0.
-- Apply in a non-production project first. Realtime movement remains ephemeral.

begin;

create table if not exists public.game_profiles (
  user_id uuid primary key references public.profiles (id) on delete cascade,
  avatar_preset text not null,
  current_scene text not null default 'campus',
  last_x double precision not null default 800,
  last_y double precision not null default 570,
  tutorial_step integer not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  constraint game_profiles_avatar_check
    check (avatar_preset in (
      'theory-blue', 'systems-teal', 'vision-green',
      'robotics-gold', 'agents-red', 'safety-violet'
    )),
  constraint game_profiles_scene_check
    check (current_scene in ('campus', 'academy', 'research', 'compute', 'paper', 'plaza')),
  constraint game_profiles_position_check
    check (last_x between 0 and 1600 and last_y between 0 and 900),
  constraint game_profiles_tutorial_check check (tutorial_step >= 0)
);

create or replace function public.set_game_profile_updated_at()
returns trigger
language plpgsql
set search_path = public
as $$
begin
  new.updated_at := now();
  return new;
end;
$$;

drop trigger if exists game_profiles_set_updated_at on public.game_profiles;
create trigger game_profiles_set_updated_at
  before update on public.game_profiles
  for each row execute function public.set_game_profile_updated_at();

alter table public.game_profiles enable row level security;

drop policy if exists game_profiles_select_own_admin on public.game_profiles;
create policy game_profiles_select_own_admin
  on public.game_profiles for select
  using (auth.uid() = user_id or public.is_admin());

drop policy if exists game_profiles_insert_own on public.game_profiles;
create policy game_profiles_insert_own
  on public.game_profiles for insert
  with check (auth.uid() = user_id);

drop policy if exists game_profiles_update_own on public.game_profiles;
create policy game_profiles_update_own
  on public.game_profiles for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

revoke all on table public.game_profiles from anon, authenticated;
grant select, insert, update on table public.game_profiles to authenticated;

commit;
