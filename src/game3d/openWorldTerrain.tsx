import { useMemo } from 'react';
import * as THREE from 'three';
import { CuboidCollider, RigidBody } from '@react-three/rapier';

/**
 * Procedural voxel open-world layer (Minicraft-like exploration scale).
 * Original art direction — not affiliated with any commercial block game brand.
 */

const CELL = 2;
const HALF = 36; // world half-extent in cells → ~144 world units across

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
    noise2(x * 0.08, z * 0.08) * 0.55 +
    noise2(x * 0.18, z * 0.18) * 0.3 +
    noise2(x * 0.4, z * 0.4) * 0.15
  );
}

export type TerrainKind = 'grass' | 'sand' | 'dirt' | 'stone' | 'water' | 'path';

export function sampleTerrain(wx: number, wz: number): { h: number; kind: TerrainKind } {
  // Station hub stays relatively flat.
  const hubDist = Math.hypot(wx, wz - 8);
  const island = 1 - Math.min(1, Math.hypot(wx * 0.012, wz * 0.012));
  let h = fbm(wx, wz) * 4.2 * island;
  if (hubDist < 18) h *= hubDist / 18;
  // Southern pier / beach
  if (wz > 28) h = Math.min(h, 0.15);
  // Northern cliffs
  if (wz < -40) h += (1 - (wz + 70) / 30) * 2.2;

  let kind: TerrainKind = 'grass';
  if (wz > 32 || (wz > 20 && Math.abs(wx) > 22)) kind = 'sand';
  else if (h > 2.6) kind = 'stone';
  else if (h > 1.4) kind = 'dirt';
  else if (h < 0.12 && fbm(wx + 40, wz - 20) < 0.32) kind = 'water';

  // Soft path toward major axes
  if (Math.abs(wx) < 1.6 || Math.abs(wz - 8) < 1.4) {
    if (kind !== 'water' && kind !== 'stone') kind = 'path';
    h = Math.min(h, 0.25);
  }

  return { h: Math.max(0, h), kind };
}

const KIND_COLOR: Record<TerrainKind, string> = {
  grass: '#6faf5a',
  sand: '#e2d3a2',
  dirt: '#9a7348',
  stone: '#8a9096',
  water: '#4eb3d4',
  path: '#c4b48a',
};

function buildInstances() {
  const grass: THREE.Matrix4[] = [];
  const sand: THREE.Matrix4[] = [];
  const dirt: THREE.Matrix4[] = [];
  const stone: THREE.Matrix4[] = [];
  const water: THREE.Matrix4[] = [];
  const path: THREE.Matrix4[] = [];
  const treeTrunk: THREE.Matrix4[] = [];
  const treeLeaf: THREE.Matrix4[] = [];
  const rocks: THREE.Matrix4[] = [];
  const flowers: THREE.Matrix4[] = [];

  const dummy = new THREE.Object3D();
  const buckets: Record<TerrainKind, THREE.Matrix4[]> = {
    grass,
    sand,
    dirt,
    stone,
    water,
    path,
  };

  for (let ix = -HALF; ix <= HALF; ix += 1) {
    for (let iz = -HALF; iz <= HALF; iz += 1) {
      const wx = ix * CELL;
      const wz = iz * CELL;
      // Skip dense fill under far ocean skirt
      if (Math.hypot(wx, wz) > HALF * CELL * 1.02) continue;

      const { h, kind } = sampleTerrain(wx, wz);
      const layers = Math.max(1, Math.min(5, Math.round(h / 0.55) + 1));
      for (let ly = 0; ly < layers; ly += 1) {
        const y = ly * 0.55 + 0.275;
        dummy.position.set(wx, y, wz);
        dummy.scale.set(CELL * 0.98, 0.55, CELL * 0.98);
        dummy.updateMatrix();
        const k = ly === layers - 1 ? kind : ly === 0 ? 'stone' : 'dirt';
        buckets[k].push(dummy.matrix.clone());
      }

      // Vegetation
      const hv = hash2(ix + 3, iz - 7);
      if (kind === 'grass' && h < 2.2 && hv > 0.86) {
        const trunkH = 1.2 + hv * 1.6;
        dummy.position.set(wx, layers * 0.55 + trunkH / 2, wz);
        dummy.scale.set(0.45, trunkH, 0.45);
        dummy.updateMatrix();
        treeTrunk.push(dummy.matrix.clone());
        for (let t = 0; t < 3; t += 1) {
          dummy.position.set(wx, layers * 0.55 + trunkH + t * 0.55, wz);
          const s = 1.6 - t * 0.35;
          dummy.scale.set(s, 0.55, s);
          dummy.updateMatrix();
          treeLeaf.push(dummy.matrix.clone());
        }
      } else if (kind === 'grass' && hv > 0.78 && hv <= 0.86) {
        dummy.position.set(wx + (hv - 0.8) * 4, layers * 0.55 + 0.2, wz);
        dummy.scale.set(0.35, 0.35, 0.35);
        dummy.updateMatrix();
        flowers.push(dummy.matrix.clone());
      } else if ((kind === 'stone' || kind === 'dirt') && hv > 0.9) {
        dummy.position.set(wx, layers * 0.55 + 0.4, wz);
        dummy.scale.set(0.9 + hv, 0.7 + hv, 0.9 + hv);
        dummy.updateMatrix();
        rocks.push(dummy.matrix.clone());
      }
    }
  }

  return { grass, sand, dirt, stone, water, path, treeTrunk, treeLeaf, rocks, flowers };
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
  const mesh = useMemo(() => {
    if (matrices.length === 0) return null;
    const m = new THREE.InstancedMesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshStandardMaterial({ color, roughness: 0.92, flatShading: true }),
      matrices.length,
    );
    matrices.forEach((mat, i) => m.setMatrixAt(i, mat));
    m.instanceMatrix.needsUpdate = true;
    m.castShadow = castShadow;
    m.receiveShadow = true;
    return m;
  }, [matrices, color, castShadow]);

  if (!mesh) return null;
  return <primitive object={mesh} />;
}

/** Outposts / ruins for open-world exploration density. */
function VoxelOutpost({ position, palette }: { position: [number, number, number]; palette: string }) {
  const [x, y, z] = position;
  const blocks: Array<{ p: [number, number, number]; s: [number, number, number]; c: string }> = [];
  // floor
  for (let ix = -2; ix <= 2; ix += 1) {
    for (let iz = -2; iz <= 2; iz += 1) {
      blocks.push({ p: [x + ix * 0.6, y + 0.25, z + iz * 0.6], s: [0.58, 0.5, 0.58], c: '#b8a882' });
    }
  }
  // walls
  for (let i = -2; i <= 2; i += 1) {
    blocks.push({ p: [x + i * 0.6, y + 1.0, z - 1.2], s: [0.58, 1.0, 0.4], c: palette });
    blocks.push({ p: [x + i * 0.6, y + 1.0, z + 1.2], s: [0.58, 1.0, 0.4], c: palette });
    blocks.push({ p: [x - 1.2, y + 1.0, z + i * 0.6], s: [0.4, 1.0, 0.58], c: palette });
    blocks.push({ p: [x + 1.2, y + 1.0, z + i * 0.6], s: [0.4, 1.0, 0.58], c: palette });
  }
  // roof slab
  for (let ix = -2; ix <= 2; ix += 1) {
    for (let iz = -2; iz <= 2; iz += 1) {
      blocks.push({ p: [x + ix * 0.6, y + 1.7, z + iz * 0.6], s: [0.58, 0.4, 0.58], c: '#6d8f6a' });
    }
  }
  // beacon
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

export function OpenWorldTerrain({ reducedMotion }: { reducedMotion: boolean }) {
  const data = useMemo(() => buildInstances(), []);
  const halfWorld = HALF * CELL;

  return (
    <group>
      {/* Walkable mega-collider (flat playable plane). Visual height is decorative. */}
      <RigidBody type="fixed" colliders={false} friction={1.2}>
        <CuboidCollider args={[halfWorld + 4, 0.6, halfWorld + 4]} position={[0, -0.6, 0]} />
        <mesh position={[0, -0.05, 0]} receiveShadow>
          <boxGeometry args={[(halfWorld + 4) * 2, 0.2, (halfWorld + 4) * 2]} />
          <meshStandardMaterial color="#6faf5a" roughness={0.98} flatShading />
        </mesh>
      </RigidBody>

      <InstancedBoxes matrices={data.grass} color={KIND_COLOR.grass} />
      <InstancedBoxes matrices={data.sand} color={KIND_COLOR.sand} />
      <InstancedBoxes matrices={data.dirt} color={KIND_COLOR.dirt} />
      <InstancedBoxes matrices={data.stone} color={KIND_COLOR.stone} castShadow />
      <InstancedBoxes matrices={data.water} color={KIND_COLOR.water} />
      <InstancedBoxes matrices={data.path} color={KIND_COLOR.path} />
      {!reducedMotion && (
        <>
          <InstancedBoxes matrices={data.treeTrunk} color="#8b5a2b" castShadow />
          <InstancedBoxes matrices={data.treeLeaf} color="#3f9b4a" castShadow />
          <InstancedBoxes matrices={data.rocks} color="#7a8086" castShadow />
          <InstancedBoxes matrices={data.flowers} color="#f2a0c0" />
        </>
      )}

      {/* Exploration outposts across the open map */}
      <VoxelOutpost position={[-36, 0, -28]} palette="#7d8a9a" />
      <VoxelOutpost position={[34, 0, -22]} palette="#9a7d6a" />
      <VoxelOutpost position={[-40, 0, 24]} palette="#6a8f7a" />
      <VoxelOutpost position={[38, 0, 30]} palette="#8a7a9a" />
      <VoxelOutpost position={[0, 0, -48]} palette="#6f7f8a" />
      <VoxelOutpost position={[-22, 0, 42]} palette="#8a9060" />
      <VoxelOutpost position={[20, 0, 44]} palette="#906a60" />

      {/* Distant ridge wall (complexity silhouette) */}
      {Array.from({ length: 40 }).map((_, i) => {
        const x = -HALF * CELL + i * (CELL * 1.8);
        const z = -HALF * CELL + 2;
        const h = 2 + hash2(i, 9) * 5;
        return (
          <mesh key={`ridge-${i}`} position={[x, h / 2, z]} castShadow receiveShadow>
            <boxGeometry args={[CELL * 1.6, h, CELL]} />
            <meshStandardMaterial color="#8a9096" roughness={0.92} flatShading />
          </mesh>
        );
      })}
    </group>
  );
}

export const OPEN_WORLD_HALF = HALF * CELL;
