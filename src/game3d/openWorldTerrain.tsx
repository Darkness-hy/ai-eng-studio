import { useFrame } from '@react-three/fiber';
import { CuboidCollider, HeightfieldCollider, RigidBody } from '@react-three/rapier';
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type MutableRefObject,
} from 'react';
import * as THREE from 'three';

/**
 * Chunked procedural open world (Minicraft-like exploration).
 * - Biomes: meadow / shore / ridge
 * - Streamed chunks around the player
 * - Per-chunk heightfield colliders for walkable relief
 * - Ridge caves with walkable interiors
 * Original art; not affiliated with commercial block-game brands.
 */

export const CELL = 2;
/** Half-extent of the playable continent (world units). */
export const WORLD_HALF = 108;
const CHUNK_CELLS = 12;
const CHUNK_WORLD = CHUNK_CELLS * CELL;
/** Chunks loaded around the player → (2*r+1)^2 window. */
const LOAD_RADIUS = 3;
const MAX_CACHE = 80;

export type BiomeId = 'meadow' | 'shore' | 'ridge';
export type TerrainKind = 'grass' | 'sand' | 'dirt' | 'stone' | 'water' | 'path' | 'snow';

function hash2(x: number, z: number): number {
  const n = Math.sin(x * 127.1 + z * 311.7) * 43758.5453;
  return n - Math.floor(n);
}

function noise2(x: number, z: number): number {
  const x0 = Math.floor(x);
  const z0 = Math.floor(z);
  const fx = x - x0;
  const fz = z - z0;
  const u = fx * fx * (3 - 2 * fx);
  const v = fz * fz * (3 - 2 * fz);
  const a = hash2(x0, z0);
  const b = hash2(x0 + 1, z0);
  const c = hash2(x0, z0 + 1);
  const d = hash2(x0 + 1, z0 + 1);
  return a * (1 - u) * (1 - v) + b * u * (1 - v) + c * (1 - u) * v + d * u * v;
}

function fbm(x: number, z: number): number {
  return (
    noise2(x * 0.07, z * 0.07) * 0.5 +
    noise2(x * 0.16, z * 0.16) * 0.32 +
    noise2(x * 0.38, z * 0.38) * 0.18
  );
}

export function sampleBiome(wx: number, wz: number): BiomeId {
  // Larger continent: shore on outer ring + south; ridge north and high plateaus.
  if (wz > 36 || Math.hypot(wx, wz) > 78) return 'shore';
  if (wz < -34 || fbm(wx * 0.45, wz * 0.45) > 0.7) return 'ridge';
  return 'meadow';
}

export function sampleHeight(wx: number, wz: number): number {
  const hubDist = Math.hypot(wx, wz - 8);
  const island = 1 - Math.min(1, Math.hypot(wx * 0.008, wz * 0.008));
  let h = fbm(wx, wz) * 5.6 * Math.max(0.12, island);
  if (hubDist < 20) h *= hubDist / 20;
  const biome = sampleBiome(wx, wz);
  if (biome === 'shore') h = Math.min(h, 0.35 + fbm(wx + 9, wz) * 0.45);
  if (biome === 'ridge') h += 1.35 + fbm(wx - 11, wz + 5) * 2.6;
  if (Math.abs(wx) < 2.2 || Math.abs(wz - 8) < 2) h = Math.min(h, 0.35);
  // Cave mouths: depress ridge surface into a ramp so heightfield stays walkable into the mouth.
  for (const cave of CAVE_SITES) {
    const d = Math.hypot(wx - cave.x, wz - cave.z);
    if (d < cave.radius) {
      const t = 1 - d / cave.radius;
      h = Math.min(h, cave.floorY + (1 - t) * 1.2);
    }
  }
  return Math.max(0.05, h);
}

/** Fixed cave mouth sites (ridge). Floor is walkable via heightfield depression + interior tunnel. */
export const CAVE_SITES: Array<{
  id: string;
  x: number;
  z: number;
  radius: number;
  floorY: number;
  facing: number; // yaw radians: tunnel goes inward
  label: string;
}> = [
  { id: 'cave-north', x: -8, z: -52, radius: 5.5, floorY: 0.45, facing: Math.PI, label: '北脊主洞' },
  { id: 'cave-ne', x: 42, z: -38, radius: 4.8, floorY: 0.5, facing: -2.2, label: '东北矿洞' },
  { id: 'cave-nw', x: -46, z: -34, radius: 4.8, floorY: 0.5, facing: 2.2, label: '西北苔洞' },
  { id: 'cave-mid', x: 18, z: -58, radius: 4.2, floorY: 0.4, facing: Math.PI * 0.9, label: '脊线裂隙' },
];

export function sampleTerrain(wx: number, wz: number): {
  h: number;
  kind: TerrainKind;
  biome: BiomeId;
} {
  const biome = sampleBiome(wx, wz);
  const h = sampleHeight(wx, wz);
  let kind: TerrainKind = 'grass';
  if (biome === 'shore') {
    kind = h < 0.25 && fbm(wx + 40, wz - 20) < 0.38 ? 'water' : 'sand';
  } else if (biome === 'ridge') {
    kind = h > 3.2 ? 'snow' : h > 2.0 ? 'stone' : 'dirt';
  } else {
    kind = h > 2.4 ? 'dirt' : 'grass';
    if (h < 0.2 && fbm(wx + 40, wz - 20) < 0.3) kind = 'water';
  }
  if (
    (Math.abs(wx) < 1.8 || Math.abs(wz - 8) < 1.6) &&
    kind !== 'water' &&
    kind !== 'stone' &&
    kind !== 'snow'
  ) {
    kind = 'path';
  }
  return { h, kind, biome };
}

/** Top of stacked voxel columns (must match createChunk visual layers). */
export function surfaceTopY(wx: number, wz: number): number {
  const h = sampleHeight(wx, wz);
  const layers = Math.max(1, Math.min(6, Math.round(h / 0.55) + 1));
  // Top face of the uppermost block (center = (ly+0.5)*0.55, half = 0.275 → top = (ly+1)*0.55).
  return layers * 0.55;
}

/** Collider surface slightly ABOVE visual top so feet never clip into grass blocks. */
export function walkSurfaceY(wx: number, wz: number): number {
  return surfaceTopY(wx, wz) + 0.12;
}

const KIND_COLOR: Record<TerrainKind, string> = {
  grass: '#6faf5a',
  sand: '#e2d3a2',
  dirt: '#9a7348',
  stone: '#8a9096',
  water: '#4eb3d4',
  path: '#c4b48a',
  snow: '#eef6ff',
};

type ChunkKey = string;
function chunkKey(cx: number, cz: number): ChunkKey {
  return `${cx},${cz}`;
}

function worldToChunk(wx: number, wz: number): [number, number] {
  return [Math.floor(wx / CHUNK_WORLD), Math.floor(wz / CHUNK_WORLD)];
}

type ChunkData = {
  key: ChunkKey;
  cx: number;
  cz: number;
  originX: number;
  originZ: number;
  heights: number[];
  byKind: Record<TerrainKind, THREE.Matrix4[]>;
  trunk: THREE.Matrix4[];
  leaf: THREE.Matrix4[];
  decor: THREE.Matrix4[];
  decorColor: string;
};

function createChunk(cx: number, cz: number): ChunkData {
  const originX = cx * CHUNK_WORLD;
  const originZ = cz * CHUNK_WORLD;
  const n = CHUNK_CELLS + 1;
  const heights: number[] = new Array(n * n);
  const byKind: Record<TerrainKind, THREE.Matrix4[]> = {
    grass: [],
    sand: [],
    dirt: [],
    stone: [],
    water: [],
    path: [],
    snow: [],
  };
  const trunk: THREE.Matrix4[] = [];
  const leaf: THREE.Matrix4[] = [];
  const decor: THREE.Matrix4[] = [];
  let decorColor = '#f2a0c0';
  const dummy = new THREE.Object3D();

  for (let iz = 0; iz < n; iz += 1) {
    for (let ix = 0; ix < n; ix += 1) {
      heights[iz * n + ix] = walkSurfaceY(originX + ix * CELL, originZ + iz * CELL);
    }
  }

  for (let iz = 0; iz < CHUNK_CELLS; iz += 1) {
    for (let ix = 0; ix < CHUNK_CELLS; ix += 1) {
      const wx = originX + (ix + 0.5) * CELL;
      const wz = originZ + (iz + 0.5) * CELL;
      if (Math.abs(wx) > WORLD_HALF + 4 || Math.abs(wz) > WORLD_HALF + 4) continue;
      const { h, kind, biome } = sampleTerrain(wx, wz);
      const layers = Math.max(1, Math.min(6, Math.round(h / 0.55) + 1));
      for (let ly = 0; ly < layers; ly += 1) {
        dummy.position.set(wx, ly * 0.55 + 0.275, wz);
        dummy.scale.set(CELL * 0.96, 0.55, CELL * 0.96);
        dummy.updateMatrix();
        const k = ly === layers - 1 ? kind : ly === 0 ? 'stone' : 'dirt';
        byKind[k].push(dummy.matrix.clone());
      }

      const hv = hash2(cx * 97 + ix, cz * 53 + iz);
      if (biome === 'meadow' && kind === 'grass' && hv > 0.88) {
        const trunkH = 1.1 + hv * 1.5;
        dummy.position.set(wx, layers * 0.55 + trunkH / 2, wz);
        dummy.scale.set(0.4, trunkH, 0.4);
        dummy.updateMatrix();
        trunk.push(dummy.matrix.clone());
        for (let ti = 0; ti < 3; ti += 1) {
          dummy.position.set(wx, layers * 0.55 + trunkH + ti * 0.5, wz);
          const s = 1.5 - ti * 0.32;
          dummy.scale.set(s, 0.5, s);
          dummy.updateMatrix();
          leaf.push(dummy.matrix.clone());
        }
      } else if (biome === 'meadow' && kind === 'grass' && hv > 0.8) {
        dummy.position.set(wx, layers * 0.55 + 0.2, wz);
        dummy.scale.set(0.32, 0.32, 0.32);
        dummy.updateMatrix();
        decor.push(dummy.matrix.clone());
        decorColor = '#f2a0c0';
      } else if (biome === 'ridge' && hv > 0.9) {
        dummy.position.set(wx, layers * 0.55 + 0.45, wz);
        dummy.scale.set(0.9 + hv, 0.7, 0.9 + hv);
        dummy.updateMatrix();
        decor.push(dummy.matrix.clone());
        decorColor = '#9aa3aa';
      } else if (biome === 'shore' && kind === 'sand' && hv > 0.91) {
        dummy.position.set(wx, layers * 0.55 + 0.25, wz);
        dummy.scale.set(0.5, 0.35, 0.5);
        dummy.updateMatrix();
        decor.push(dummy.matrix.clone());
        decorColor = '#d9c27a';
      }
    }
  }

  return {
    key: chunkKey(cx, cz),
    cx,
    cz,
    originX,
    originZ,
    heights,
    byKind,
    trunk,
    leaf,
    decor,
    decorColor,
  };
}

function InstancedBoxes({
  matrices,
  color,
  castShadow = false,
}: {
  matrices: THREE.Matrix4[];
  color: string;
  castShadow?: boolean;
}) {
  const ref = useRef<THREE.InstancedMesh>(null);
  useEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    for (let i = 0; i < matrices.length; i += 1) mesh.setMatrixAt(i, matrices[i]!);
    mesh.count = matrices.length;
    mesh.instanceMatrix.needsUpdate = true;
  }, [matrices]);

  if (matrices.length === 0) return null;
  return (
    <instancedMesh
      ref={ref}
      args={[undefined, undefined, matrices.length]}
      castShadow={castShadow}
      receiveShadow
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color={color} roughness={0.92} flatShading />
    </instancedMesh>
  );
}

function ChunkView({
  chunk,
  reducedMotion,
}: {
  chunk: ChunkData;
  reducedMotion: boolean;
}) {
  // Rapier heightfield is XZ-aligned, centered on the rigid body.
  // scale.y multiplies height samples; we store absolute meters so y=1.
  const scale = useMemo(() => ({ x: CHUNK_WORLD, y: 1, z: CHUNK_WORLD }), []);
  // Sit collider flush with top of voxel stack (tiny epsilon under top face).
  const bodyPos: [number, number, number] = [
    chunk.originX + CHUNK_WORLD / 2,
    0,
    chunk.originZ + CHUNK_WORLD / 2,
  ];

  return (
    <group>
      {(Object.keys(chunk.byKind) as TerrainKind[]).map((kind) => (
        <InstancedBoxes
          key={kind}
          matrices={chunk.byKind[kind]}
          color={KIND_COLOR[kind]}
          castShadow={kind === 'stone' || kind === 'snow'}
        />
      ))}
      {!reducedMotion && (
        <>
          <InstancedBoxes matrices={chunk.trunk} color="#8b5a2b" castShadow />
          <InstancedBoxes matrices={chunk.leaf} color="#3f9b4a" castShadow />
          <InstancedBoxes matrices={chunk.decor} color={chunk.decorColor} />
        </>
      )}
      <RigidBody type="fixed" colliders={false} position={bodyPos} friction={0.9} restitution={0}>
        <HeightfieldCollider
          args={[CHUNK_CELLS, CHUNK_CELLS, chunk.heights, scale]}
          friction={0.9}
          restitution={0}
        />
      </RigidBody>
    </group>
  );
}

function VoxelOutpost({
  position,
  palette,
}: {
  position: [number, number, number];
  palette: string;
}) {
  const [x, y, z] = position;
  const blocks: Array<{ p: [number, number, number]; s: [number, number, number]; c: string }> = [];
  for (let ix = -2; ix <= 2; ix += 1) {
    for (let iz = -2; iz <= 2; iz += 1) {
      blocks.push({
        p: [x + ix * 0.6, y + 0.25, z + iz * 0.6],
        s: [0.58, 0.5, 0.58],
        c: '#b8a882',
      });
      blocks.push({
        p: [x + ix * 0.6, y + 1.7, z + iz * 0.6],
        s: [0.58, 0.4, 0.58],
        c: '#6d8f6a',
      });
    }
  }
  for (let i = -2; i <= 2; i += 1) {
    blocks.push({ p: [x + i * 0.6, y + 1.0, z - 1.2], s: [0.58, 1.0, 0.4], c: palette });
    blocks.push({ p: [x + i * 0.6, y + 1.0, z + 1.2], s: [0.58, 1.0, 0.4], c: palette });
  }
  blocks.push({ p: [x, y + 2.4, z], s: [0.45, 1.2, 0.45], c: '#e8c45a' });
  return (
    <group>
      {blocks.map((b, i) => (
        <mesh key={i} castShadow receiveShadow position={b.p}>
          <boxGeometry args={b.s} />
          <meshStandardMaterial color={b.c} roughness={0.9} flatShading />
        </mesh>
      ))}
      <RigidBody type="fixed" colliders={false}>
        <CuboidCollider args={[1.5, 1.2, 1.5]} position={[x, y + 1.0, z]} />
      </RigidBody>
    </group>
  );
}

/** Walkable ridge cave: mouth ramp (via sampleHeight) + tunnel floor/walls/ceiling. */
function RidgeCave({
  x,
  z,
  floorY,
  facing,
  label,
}: {
  x: number;
  z: number;
  floorY: number;
  facing: number;
  label: string;
}) {
  const dirX = Math.sin(facing);
  const dirZ = Math.cos(facing);
  const segments = 7;
  const spacing = 2.2;
  const floorMeshes: Array<{ p: [number, number, number]; s: [number, number, number] }> = [];
  const wallMeshes: Array<{ p: [number, number, number]; s: [number, number, number] }> = [];
  for (let i = 0; i < segments; i += 1) {
    const px = x + dirX * (i * spacing);
    const pz = z + dirZ * (i * spacing);
    const fy = floorY + i * 0.04;
    floorMeshes.push({ p: [px, fy, pz], s: [3.2, 0.4, 2.4] });
    // side walls
    wallMeshes.push({ p: [px + dirZ * 1.7, fy + 1.2, pz - dirX * 1.7], s: [0.5, 2.4, 2.2] });
    wallMeshes.push({ p: [px - dirZ * 1.7, fy + 1.2, pz + dirX * 1.7], s: [0.5, 2.4, 2.2] });
    // ceiling
    wallMeshes.push({ p: [px, fy + 2.5, pz], s: [3.4, 0.45, 2.4] });
  }
  // crystal cache at the end
  const endX = x + dirX * (segments * spacing * 0.85);
  const endZ = z + dirZ * (segments * spacing * 0.85);
  const endY = floorY + segments * 0.04 + 0.9;

  return (
    <group>
      {floorMeshes.map((b, i) => (
        <mesh key={`cf-${i}`} receiveShadow position={b.p}>
          <boxGeometry args={b.s} />
          <meshStandardMaterial color="#4a4540" roughness={0.95} flatShading />
        </mesh>
      ))}
      {wallMeshes.map((b, i) => (
        <mesh key={`cw-${i}`} castShadow receiveShadow position={b.p}>
          <boxGeometry args={b.s} />
          <meshStandardMaterial color="#5c5854" roughness={0.92} flatShading />
        </mesh>
      ))}
      {/* floor colliders along tunnel */}
      {floorMeshes.map((b, i) => (
        <RigidBody key={`cfc-${i}`} type="fixed" colliders={false} position={b.p} friction={1.15}>
          <CuboidCollider args={[b.s[0] / 2, b.s[1] / 2, b.s[2] / 2]} />
        </RigidBody>
      ))}
      {/* wall colliders (sides only) */}
      {wallMeshes.slice(0, segments * 2).map((b, i) => (
        <RigidBody key={`cwc-${i}`} type="fixed" colliders={false} position={b.p}>
          <CuboidCollider args={[b.s[0] / 2, b.s[1] / 2, b.s[2] / 2]} />
        </RigidBody>
      ))}
      <mesh position={[endX, endY, endZ]} castShadow>
        <boxGeometry args={[0.7, 1.1, 0.7]} />
        <meshStandardMaterial color="#7ec8ff" emissive="#2a6a9a" emissiveIntensity={0.45} flatShading />
      </mesh>
      <pointLight position={[endX, endY + 0.6, endZ]} intensity={1.1} distance={10} color="#9fd4ff" />
      <pointLight
        position={[x + dirX * 1.2, floorY + 1.6, z + dirZ * 1.2]}
        intensity={0.55}
        distance={8}
        color="#ffd9a0"
      />
      {/* mouth marker (visual only) */}
      <mesh position={[x, floorY + 2.8, z]}>
        <boxGeometry args={[0.35, 0.35, 0.35]} />
        <meshStandardMaterial color="#f0d78d" emissive="#b0892a" emissiveIntensity={0.35} flatShading />
      </mesh>
      <group visible={false} userData={{ caveLabel: label }} />
    </group>
  );
}

export function OpenWorldTerrain({
  reducedMotion,
  playerPosRef,
}: {
  reducedMotion: boolean;
  playerPosRef?: MutableRefObject<THREE.Vector3>;
}) {
  const [loaded, setLoaded] = useState<ChunkData[]>([]);
  const cacheRef = useRef(new Map<ChunkKey, ChunkData>());
  const lastCenterRef = useRef('__none__');

  useEffect(() => {
    const seeds = [createChunk(0, 0), createChunk(0, 1), createChunk(-1, 0), createChunk(1, 0)];
    for (const s of seeds) cacheRef.current.set(s.key, s);
    setLoaded(seeds);
  }, []);

  useFrame(() => {
    const px = playerPosRef?.current.x ?? 0;
    const pz = playerPosRef?.current.z ?? 22;
    const [pcx, pcz] = worldToChunk(px, pz);
    const centerKey = `${pcx},${pcz}`;
    if (centerKey === lastCenterRef.current) return;
    lastCenterRef.current = centerKey;

    const needed: Array<[number, number]> = [];
    for (let dz = -LOAD_RADIUS; dz <= LOAD_RADIUS; dz += 1) {
      for (let dx = -LOAD_RADIUS; dx <= LOAD_RADIUS; dx += 1) {
        needed.push([pcx + dx, pcz + dz]);
      }
    }

    const next: ChunkData[] = [];
    const neededKeys = new Set<string>();
    for (const [cx, cz] of needed) {
      const ox = cx * CHUNK_WORLD + CHUNK_WORLD / 2;
      const oz = cz * CHUNK_WORLD + CHUNK_WORLD / 2;
      if (Math.abs(ox) > WORLD_HALF + CHUNK_WORLD || Math.abs(oz) > WORLD_HALF + CHUNK_WORLD) {
        continue;
      }
      const key = chunkKey(cx, cz);
      neededKeys.add(key);
      let chunk = cacheRef.current.get(key);
      if (!chunk) {
        chunk = createChunk(cx, cz);
        cacheRef.current.set(key, chunk);
      }
      next.push(chunk);
    }

    // prune cache
    if (cacheRef.current.size > MAX_CACHE) {
      for (const key of [...cacheRef.current.keys()]) {
        if (!neededKeys.has(key)) cacheRef.current.delete(key);
        if (cacheRef.current.size <= MAX_CACHE) break;
      }
    }

    setLoaded(next);
  });

  return (
    <group>
      {/* Hub safety pad */}
      <RigidBody type="fixed" colliders={false} friction={0.95} position={[0, -0.5, 8]}>
        <CuboidCollider args={[14, 0.2, 14]} restitution={0} friction={0.95} />
      </RigidBody>

      {loaded.map((chunk) => (
        <ChunkView key={chunk.key} chunk={chunk} reducedMotion={reducedMotion} />
      ))}

      <VoxelOutpost position={[-36, 0, -28]} palette="#7d8a9a" />
      <VoxelOutpost position={[34, 0, -22]} palette="#9a7d6a" />
      <VoxelOutpost position={[-40, 0, 24]} palette="#6a8f7a" />
      <VoxelOutpost position={[38, 0, 30]} palette="#8a7a9a" />
      <VoxelOutpost position={[0, 0, -48]} palette="#6f7f8a" />
      <VoxelOutpost position={[-22, 0, 42]} palette="#8a9060" />
      <VoxelOutpost position={[20, 0, 44]} palette="#906a60" />
      {/* Far-ring outposts for larger continent */}
      <VoxelOutpost position={[-70, 0, -10]} palette="#6a7a88" />
      <VoxelOutpost position={[70, 0, -8]} palette="#887a6a" />
      <VoxelOutpost position={[12, 0, 70]} palette="#6a886a" />
      <VoxelOutpost position={[-14, 0, 68]} palette="#886a7a" />

      {CAVE_SITES.map((cave) => (
        <RidgeCave
          key={cave.id}
          x={cave.x}
          z={cave.z}
          floorY={cave.floorY}
          facing={cave.facing}
          label={cave.label}
        />
      ))}
    </group>
  );
}

export const OPEN_WORLD_HALF = WORLD_HALF;
