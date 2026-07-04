-- 012 — 三类排行榜摘要 RPC
-- 在 Supabase Dashboard → SQL Editor 整段执行一次。幂等。
-- 学生端只通过本 RPC 获取允许公开的前三名、自己的百分位区间和上榜差距。
-- 不返回总用户数、完整排名、email 或非前三名用户。

drop function if exists public.get_leaderboard_summary(integer);

create or replace function public.get_leaderboard_summary(lesson_ids text[])
returns jsonb
language plpgsql
stable
security definer
set search_path = public
as $$
declare
  requester uuid := auth.uid();
  result jsonb;
begin
  if requester is null then
    raise exception 'not authenticated' using errcode = '28000';
  end if;

  with
  board_defs(board_id, sort_order) as (
    values
      ('progress', 1),
      ('placement', 2),
      ('unit_quiz', 3)
  ),
  catalog as (
    select distinct btrim(id) as lesson_id
    from unnest(coalesce(lesson_ids, array[]::text[])) as ids(id)
    where btrim(id) <> ''
  ),
  lesson_total as (
    select count(*)::numeric as total from catalog
  ),
  settings as (
    select 10::int as min_quiz_attempts
  ),
  students as (
    select
      id as user_id,
      coalesce(nullif(btrim(display_name), ''), '未命名学习者') as display_name
    from public.profiles
    where role = 'student'
  ),
  progress_scores as (
    select
      'progress'::text as board_id,
      s.user_id,
      s.display_name,
      count(distinct p.lesson_id)::numeric as score,
      true as eligible,
      null::int as attempts
    from students s
    left join public.progress p
      on p.user_id = s.user_id
     and p.done = true
     and exists (select 1 from catalog c where c.lesson_id = p.lesson_id)
    group by s.user_id, s.display_name
  ),
  placement_scores as (
    select
      'placement'::text as board_id,
      s.user_id,
      s.display_name,
      pl.total::numeric as score,
      pl.user_id is not null as eligible,
      null::int as attempts
    from students s
    left join public.placement pl on pl.user_id = s.user_id
  ),
  quiz_raw as (
    select
      s.user_id,
      s.display_name,
      count(p.lesson_id)::int as attempts,
      avg((p.post_score::numeric / nullif(p.post_total, 0)) * 100) as avg_pct
    from students s
    left join public.progress p
      on p.user_id = s.user_id
     and p.post_total is not null
     and p.post_total > 0
     and p.post_score is not null
     and exists (select 1 from catalog c where c.lesson_id = p.lesson_id)
    group by s.user_id, s.display_name
  ),
  quiz_scores as (
    select
      'unit_quiz'::text as board_id,
      q.user_id,
      q.display_name,
      q.avg_pct as score,
      q.attempts >= (select min_quiz_attempts from settings) as eligible,
      q.attempts
    from quiz_raw q
  ),
  all_scores as (
    select * from progress_scores
    union all
    select * from placement_scores
    union all
    select * from quiz_scores
  ),
  eligible_counts as (
    select board_id, count(*)::int as eligible_count
    from all_scores
    where eligible
    group by board_id
  ),
  ranked as (
    select
      a.*,
      row_number() over (
        partition by a.board_id
        order by a.score desc nulls last, a.display_name asc, a.user_id::text asc
      ) as position
    from all_scores a
    where a.eligible
  ),
  top_entries as (
    select
      r.board_id,
      jsonb_agg(
        jsonb_build_object(
          'position', r.position,
          'displayName', r.display_name,
          'score', r.score
        )
        order by r.position
      ) as entries
    from ranked r
    join eligible_counts ec on ec.board_id = r.board_id
    where r.position <= 3
      and ec.eligible_count >= 3
    group by r.board_id
  ),
  third_scores as (
    select r.board_id, r.score as podium_score
    from ranked r
    join eligible_counts ec on ec.board_id = r.board_id
    where r.position = 3
      and ec.eligible_count >= 3
  ),
  mine_rows as (
    select
      bd.board_id,
      a.score,
      coalesce(a.eligible, false) as eligible,
      a.attempts
    from board_defs bd
    left join all_scores a
      on a.board_id = bd.board_id
     and a.user_id = requester
  ),
  mine_stats as (
    select
      m.board_id,
      m.score,
      m.eligible,
      m.attempts,
      coalesce(ec.eligible_count, 0) >= 3 as ready,
      case
        when m.eligible and coalesce(ec.eligible_count, 0) >= 3 then
          least(
            100,
            greatest(
              10,
              (
                ceil(
                  (
                    (
                      1 + (
                        select count(*)::numeric
                        from all_scores b
                        where b.board_id = m.board_id
                          and b.eligible
                          and b.score > m.score
                      )
                    ) / nullif(ec.eligible_count::numeric, 0)
                  ) * 10
                ) * 10
              )::int
            )
          )
        else null
      end as percentile,
      exists (
        select 1
        from ranked r
        join eligible_counts ec2 on ec2.board_id = r.board_id
        where r.board_id = m.board_id
          and r.user_id = requester
          and r.position <= 3
          and ec2.eligible_count >= 3
      ) as on_podium,
      case
        when m.eligible
         and coalesce(ec.eligible_count, 0) >= 3
         and not exists (
           select 1
           from ranked r
           join eligible_counts ec2 on ec2.board_id = r.board_id
           where r.board_id = m.board_id
             and r.user_id = requester
             and r.position <= 3
             and ec2.eligible_count >= 3
         )
        then greatest((select ts.podium_score from third_scores ts where ts.board_id = m.board_id) - m.score, 0)
        else null
      end as gap
    from mine_rows m
    left join eligible_counts ec on ec.board_id = m.board_id
  )
  select jsonb_build_object(
    'lessonTotal', (select total from lesson_total),
    'boards',
    jsonb_agg(
      jsonb_build_object(
        'id', bd.board_id,
        'top', coalesce(te.entries, '[]'::jsonb),
        'mine', jsonb_build_object(
          'eligible', coalesce(ms.eligible, false),
          'ready', coalesce(ms.ready, false),
          'score', ms.score,
          'attempts', ms.attempts,
          'percentile', ms.percentile,
          'onPodium', coalesce(ms.on_podium, false),
          'gap', ms.gap
        )
      )
      order by bd.sort_order
    )
  )
  into result
  from board_defs bd
  left join top_entries te on te.board_id = bd.board_id
  left join mine_stats ms on ms.board_id = bd.board_id;

  return result;
end;
$$;

revoke all on function public.get_leaderboard_summary(text[]) from public;
grant execute on function public.get_leaderboard_summary(text[]) to authenticated;
