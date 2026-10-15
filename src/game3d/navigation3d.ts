import PF from 'pathfinding';
import type {
  Game3DInteraction,
  Game3DSceneDefinition,
  NavigationCommand3D,
  Vec3,
} from './types3d';

const PLAYER_CLEARANCE = 0.42;

export function createNavigationCommand(
  id: number,
  scene: Game3DSceneDefinition,
  start: Vec3,
  requestedTarget: Vec3,
  interaction: Game3DInteraction | null = null,
): NavigationCommand3D | null {
  const approachTarget = interaction
    ? interactionApproachPoint(start, interaction)
    : requestedTarget;
  const route = findNavigationPath(scene, start, approachTarget);
  if (route.length === 0) return null;
  const destination = route.at(-1)!;
  return {
    id,
    points: route,
    destination,
    interaction,
    stopDistance: interaction ? 0.34 : 0.24,
  };
}

export function findNavigationPath(
  scene: Game3DSceneDefinition,
  start: Vec3,
  target: Vec3,
): Vec3[] {
  const { matrix, width, height } = buildMatrix(scene);
  const startCell = nearestWalkableCell(scene, matrix, worldToCell(scene, start), width, height);
  const targetCell = nearestWalkableCell(scene, matrix, worldToCell(scene, target), width, height);
  if (!startCell || !targetCell) return [];

  const grid = new PF.Grid(matrix);
  const finder = new PF.AStarFinder({
    allowDiagonal: true,
    dontCrossCorners: true,
    heuristic: PF.Heuristic.octile,
  });
  const found = finder.findPath(
    startCell[0],
    startCell[1],
    targetCell[0],
    targetCell[1],
    grid,
  );
  if (found.length === 0) return [];
  const compressed = PF.Util.compressPath(found);
  const route = compressed.slice(1).map(([cellX, cellZ]) => cellToWorld(scene, cellX, cellZ));
  return route.length > 0 ? route : [cellToWorld(scene, targetCell[0], targetCell[1])];
}

function buildMatrix(scene: Game3DSceneDefinition) {
  const width = Math.ceil((scene.bounds.maxX - scene.bounds.minX) / scene.navigation.cellSize) + 1;
  const height = Math.ceil((scene.bounds.maxZ - scene.bounds.minZ) / scene.navigation.cellSize) + 1;
  const matrix = Array.from({ length: height }, (_, z) =>
    Array.from({ length: width }, (_, x) => (isWalkable(scene, cellToWorld(scene, x, z)) ? 0 : 1)),
  );
  return { matrix, width, height };
}

function isWalkable(scene: Game3DSceneDefinition, point: Vec3): boolean {
  const [x, , z] = point;
  const { padding, shape, obstacles } = scene.navigation;
  if (shape === 'ellipse') {
    const centerX = (scene.bounds.minX + scene.bounds.maxX) / 2;
    const centerZ = (scene.bounds.minZ + scene.bounds.maxZ) / 2;
    const radiusX = (scene.bounds.maxX - scene.bounds.minX) / 2 - padding;
    const radiusZ = (scene.bounds.maxZ - scene.bounds.minZ) / 2 - padding;
    const normalized = ((x - centerX) / radiusX) ** 2 + ((z - centerZ) / radiusZ) ** 2;
    if (normalized > 1) return false;
  } else if (
    x < scene.bounds.minX + padding ||
    x > scene.bounds.maxX - padding ||
    z < scene.bounds.minZ + padding ||
    z > scene.bounds.maxZ - padding
  ) {
    return false;
  }

  const hasAuthoredWalkways = Boolean(scene.navigation.paths?.length || scene.navigation.zones?.length);
  if (hasAuthoredWalkways && !isOnAuthoredWalkway(scene, x, z)) return false;

  return !obstacles.some(
    (obstacle) =>
      x >= obstacle.minX - PLAYER_CLEARANCE &&
      x <= obstacle.maxX + PLAYER_CLEARANCE &&
      z >= obstacle.minZ - PLAYER_CLEARANCE &&
      z <= obstacle.maxZ + PLAYER_CLEARANCE,
  );
}

function isOnAuthoredWalkway(scene: Game3DSceneDefinition, x: number, z: number): boolean {
  const inZone = scene.navigation.zones?.some((zone) => {
    const normalized = ((x - zone.x) / zone.radiusX) ** 2 + ((z - zone.z) / zone.radiusZ) ** 2;
    return normalized <= 1;
  });
  if (inZone) return true;
  return Boolean(scene.navigation.paths?.some((path) => {
    for (let index = 1; index < path.points.length; index += 1) {
      if (distanceToSegment(x, z, path.points[index - 1], path.points[index]) <= path.width / 2) {
        return true;
      }
    }
    return false;
  }));
}

function distanceToSegment(
  x: number,
  z: number,
  start: [number, number],
  end: [number, number],
): number {
  const dx = end[0] - start[0];
  const dz = end[1] - start[1];
  const lengthSquared = dx * dx + dz * dz;
  if (lengthSquared === 0) return Math.hypot(x - start[0], z - start[1]);
  const t = clamp(((x - start[0]) * dx + (z - start[1]) * dz) / lengthSquared, 0, 1);
  return Math.hypot(x - (start[0] + t * dx), z - (start[1] + t * dz));
}

function nearestWalkableCell(
  scene: Game3DSceneDefinition,
  matrix: number[][],
  requested: [number, number],
  width: number,
  height: number,
): [number, number] | null {
  const [requestedX, requestedZ] = requested;
  if (matrix[requestedZ]?.[requestedX] === 0) return requested;
  let best: [number, number] | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let z = 0; z < height; z += 1) {
    for (let x = 0; x < width; x += 1) {
      if (matrix[z][x] !== 0) continue;
      const distance = (x - requestedX) ** 2 + (z - requestedZ) ** 2;
      if (distance >= bestDistance) continue;
      best = [x, z];
      bestDistance = distance;
    }
  }
  if (!best || !isWalkable(scene, cellToWorld(scene, best[0], best[1]))) return null;
  return best;
}

function worldToCell(scene: Game3DSceneDefinition, point: Vec3): [number, number] {
  const { cellSize } = scene.navigation;
  const x = Math.round((point[0] - scene.bounds.minX) / cellSize);
  const z = Math.round((point[2] - scene.bounds.minZ) / cellSize);
  const maxX = Math.ceil((scene.bounds.maxX - scene.bounds.minX) / cellSize);
  const maxZ = Math.ceil((scene.bounds.maxZ - scene.bounds.minZ) / cellSize);
  return [clamp(x, 0, maxX), clamp(z, 0, maxZ)];
}

function cellToWorld(scene: Game3DSceneDefinition, x: number, z: number): Vec3 {
  return [
    scene.bounds.minX + x * scene.navigation.cellSize,
    0.04,
    scene.bounds.minZ + z * scene.navigation.cellSize,
  ];
}

function interactionApproachPoint(start: Vec3, interaction: Game3DInteraction): Vec3 {
  if (interaction.kind === 'portal') return [...interaction.position];
  const dx = start[0] - interaction.position[0];
  const dz = start[2] - interaction.position[2];
  const length = Math.hypot(dx, dz) || 1;
  const distance = Math.min(1.45, interaction.radius * 0.58);
  return [
    interaction.position[0] + (dx / length) * distance,
    0.04,
    interaction.position[2] + (dz / length) * distance,
  ];
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
