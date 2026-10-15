import { cloudEnabled, getSupabase } from '../lib/supabase';
import { gameScenes, isGameSceneId } from './worldContent';
import type { AvatarPresetId, GameSceneId, PlayerGameProfile, WorldPoint } from './types';

const STORAGE_KEY = 'research-game:player-profile-v1';

type GameProfileRow = {
  user_id: string;
  avatar_preset: AvatarPresetId;
  current_scene: GameSceneId;
  last_x: number;
  last_y: number;
  tutorial_step: number;
};

export function loadLocalGameProfile(userId: string | null = null): PlayerGameProfile | null {
  try {
    const raw = window.localStorage.getItem(localStorageKey(userId));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<PlayerGameProfile>;
    if (!parsed.avatarPreset || !isGameSceneId(parsed.currentScene)) return null;
    const fallback = gameScenes[parsed.currentScene].spawn;
    return {
      avatarPreset: parsed.avatarPreset,
      currentScene: parsed.currentScene,
      lastPosition: validPoint(parsed.lastPosition) ? parsed.lastPosition : fallback,
      tutorialStep: Math.max(0, parsed.tutorialStep ?? 0),
    };
  } catch {
    return null;
  }
}

export function saveLocalGameProfile(
  profile: PlayerGameProfile,
  userId: string | null = null,
): void {
  window.localStorage.setItem(localStorageKey(userId), JSON.stringify(profile));
}

export async function loadCloudGameProfile(userId: string): Promise<PlayerGameProfile | null> {
  if (!cloudEnabled) return null;
  const { data, error } = await getSupabase()
    .from('game_profiles')
    .select('user_id, avatar_preset, current_scene, last_x, last_y, tutorial_step')
    .eq('user_id', userId)
    .maybeSingle<GameProfileRow>();
  if (error) {
    if (error.code === '42P01') return null;
    throw error;
  }
  return data ? fromRow(data) : null;
}

export async function saveCloudGameProfile(
  userId: string,
  profile: PlayerGameProfile,
): Promise<void> {
  if (!cloudEnabled) return;
  const { error } = await getSupabase().from('game_profiles').upsert(
    {
      user_id: userId,
      avatar_preset: profile.avatarPreset,
      current_scene: profile.currentScene,
      last_x: profile.lastPosition.x,
      last_y: profile.lastPosition.y,
      tutorial_step: profile.tutorialStep,
    },
    { onConflict: 'user_id' },
  );
  if (error && error.code !== '42P01') throw error;
}

function fromRow(row: GameProfileRow): PlayerGameProfile {
  const scene = isGameSceneId(row.current_scene) ? row.current_scene : 'campus';
  const point = { x: row.last_x, y: row.last_y };
  return {
    avatarPreset: row.avatar_preset,
    currentScene: scene,
    lastPosition: validPoint(point) ? point : gameScenes[scene].spawn,
    tutorialStep: Math.max(0, row.tutorial_step),
  };
}

function validPoint(point: WorldPoint | undefined): point is WorldPoint {
  return Boolean(
    point &&
      Number.isFinite(point.x) &&
      Number.isFinite(point.y) &&
      point.x >= 0 &&
      point.x <= 1600 &&
      point.y >= 0 &&
      point.y <= 900,
  );
}

function localStorageKey(userId: string | null): string {
  return `${STORAGE_KEY}:${userId ?? 'guest'}`;
}
