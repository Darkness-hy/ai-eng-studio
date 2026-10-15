import { gameScenes } from '../game/worldContent';
import type { GameSceneId, WorldPoint } from '../game/types';
import type { Game3DSceneDefinition, Vec3 } from './types3d';
import { gameScenes3D, isThreeSceneId } from './world3d';

const LEGACY_WIDTH = 1600;
const LEGACY_HEIGHT = 900;

export function legacyToWorld3D(sceneId: GameSceneId, point: WorldPoint): Vec3 {
  if (!isThreeSceneId(sceneId)) return [0, 1.15, 0];
  const scene = gameScenes3D[sceneId];
  const legacySpawn = gameScenes[sceneId].spawn;
  if (near(point.x, legacySpawn.x) && near(point.y, legacySpawn.y)) {
    return [...scene.spawn];
  }
  const x = scene.bounds.minX + clamp01(point.x / LEGACY_WIDTH) * sceneWidth(scene);
  const z = scene.bounds.minZ + clamp01(point.y / LEGACY_HEIGHT) * sceneDepth(scene);
  return [x, scene.spawn[1], z];
}

export function world3DToLegacy(
  sceneId: Game3DSceneDefinition['id'],
  point: Pick<Vec3Point, 'x' | 'z'>,
): WorldPoint {
  const scene = gameScenes3D[sceneId];
  return {
    x: clamp01((point.x - scene.bounds.minX) / sceneWidth(scene)) * LEGACY_WIDTH,
    y: clamp01((point.z - scene.bounds.minZ) / sceneDepth(scene)) * LEGACY_HEIGHT,
  };
}

type Vec3Point = { x: number; y: number; z: number };

function sceneWidth(scene: Game3DSceneDefinition): number {
  return scene.bounds.maxX - scene.bounds.minX;
}

function sceneDepth(scene: Game3DSceneDefinition): number {
  return scene.bounds.maxZ - scene.bounds.minZ;
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function near(a: number, b: number): boolean {
  return Math.abs(a - b) < 0.01;
}
