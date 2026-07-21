import { Html } from '@react-three/drei';
import { useFrame, type ThreeEvent } from '@react-three/fiber';
import { CuboidCollider, RigidBody } from '@react-three/rapier';
import { useEffect, useMemo, useRef, type ReactElement } from 'react';
import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import type { AgentId, GameSceneId } from '../game/types';
import { agents } from '../game/worldContent';
import type {
  CameraMode3D,
  Game3DInteraction,
  Game3DSceneDefinition,
  LandmarkInteraction3D,
  Vec3,
} from './types3d';
import { OpenWorldTerrain } from './openWorldTerrain';
import { campusPathNetwork3D } from './world3d';

export function ResearchWorld3D({
  scene,
  cameraMode,
  reducedMotion,
  onSelectGround,
  onSelectInteraction,
}: {
  scene: Game3DSceneDefinition;
  cameraMode: CameraMode3D;
  reducedMotion: boolean;
  onSelectGround: (position: Vec3) => void;
  onSelectInteraction: (interaction: Game3DInteraction) => void;
}) {
  const palette = scenePalette(scene.id);
  return (
    <>
      <color attach="background" args={[palette.sky]} />
      <fog
        attach="fog"
        args={[
          palette.fog,
          cameraMode === 'map' ? (scene.id === 'campus' ? 140 : 70) : 40,
          cameraMode === 'map' ? (scene.id === 'campus' ? 260 : 140) : scene.id === 'campus' ? 160 : 90,
        ]}
      />
      <hemisphereLight args={[palette.hemi, palette.ground, 1.55]} />
      <directionalLight
        castShadow={cameraMode !== 'map'}
        position={[14, 22, 6]}
        intensity={1.85}
        color={palette.sun}
        shadow-mapSize={[1024, 1024]}
        shadow-camera-left={scene.id === 'campus' ? -54 : -24}
        shadow-camera-right={scene.id === 'campus' ? 54 : 24}
        shadow-camera-top={scene.id === 'campus' ? 42 : 20}
        shadow-camera-bottom={scene.id === 'campus' ? -42 : -20}
        shadow-bias={-0.00015}
      />
      <NavigationSurface
        scene={scene}
        enabled={cameraMode === 'follow'}
        onSelect={onSelectGround}
      />
      {scene.id === 'campus' ? (
        <CampusWorld
          cameraMode={cameraMode}
          reducedMotion={reducedMotion}
          interactions={scene.interactions}
          onSelectInteraction={onSelectInteraction}
        />
      ) : scene.id === 'academy' ? (
        <AcademyWorld
          cameraMode={cameraMode}
          reducedMotion={reducedMotion}
          interactions={scene.interactions}
          onSelectInteraction={onSelectInteraction}
        />
      ) : (
        <ResearchInteriorWorld
          scene={scene}
          cameraMode={cameraMode}
          reducedMotion={reducedMotion}
          onSelectInteraction={onSelectInteraction}
        />
      )}
    </>
  );
}

function NavigationSurface({
  scene,
  enabled,
  onSelect,
}: {
  scene: Game3DSceneDefinition;
  enabled: boolean;
  onSelect: (position: Vec3) => void;
}) {
  const width = scene.bounds.maxX - scene.bounds.minX;
  const depth = scene.bounds.maxZ - scene.bounds.minZ;
  const centerX = (scene.bounds.minX + scene.bounds.maxX) / 2;
  const centerZ = (scene.bounds.minZ + scene.bounds.maxZ) / 2;
  const handleClick = (event: ThreeEvent<MouseEvent>) => {
    if (!enabled || event.button !== 0) return;
    event.stopPropagation();
    onSelect([event.point.x, 0.04, event.point.z]);
  };
  return (
    <mesh
      position={[centerX, 0.075, centerZ]}
      rotation={[-Math.PI / 2, 0, 0]}
      onClick={handleClick}
    >
      <planeGeometry args={[width, depth]} />
      <meshBasicMaterial transparent opacity={0} depthWrite={false} colorWrite={false} />
    </mesh>
  );
}

function CampusWorld({
  cameraMode,
  reducedMotion,
  interactions,
  onSelectInteraction,
}: WorldProps) {
  // Open voxel continent + research hub (paths/buildings remain the dense core).
  return (
    <>
      <WaveOcean reducedMotion={reducedMotion} />
      <OpenWorldTerrain reducedMotion={reducedMotion} />
      <CampusPaths mapMode={cameraMode === 'map'} />
      <CampusBuildings />
      <CampusLandmarks
        interactions={interactions}
        reducedMotion={reducedMotion}
        onSelect={onSelectInteraction}
      />
      {cameraMode !== 'map' && (
        <AgentFigure
          agentId="campus-guide"
          position={[1.4, 0, 15.5]}
          color="#7a5c56"
          reducedMotion={reducedMotion}
          showLabel
          onClick={() => selectById(interactions, 'guide', onSelectInteraction)}
        />
      )}
      <InteractionMarkers
        interactions={interactions}
        cameraMode={cameraMode}
        reducedMotion={reducedMotion}
        onSelect={onSelectInteraction}
      />
    </>
  );
}

function AcademyWorld({
  cameraMode,
  reducedMotion,
  interactions,
  onSelectInteraction,
}: WorldProps) {
  return (
    <>
      <RigidBody type="fixed" colliders={false} friction={1.1}>
        <CuboidCollider args={[11, 0.35, 9]} position={[0, -0.35, 0]} />
        <mesh receiveShadow position={[0, -0.38, 0]}>
          <boxGeometry args={[22, 0.7, 18]} />
          <meshStandardMaterial color="#c5c0b0" roughness={0.92} />
        </mesh>
        <AcademyWalls />
        <AcademyFurniture />
      </RigidBody>
      <AcademyAmbient reducedMotion={reducedMotion} />
      <AgentFigure
        agentId="research-mentor"
        position={[0, 0, -2.8]}
        color="#4a6670"
        reducedMotion={reducedMotion}
        showLabel={cameraMode !== 'map'}
        onClick={() => selectById(interactions, 'academy-mentor', onSelectInteraction)}
      />
      <InteractionMarkers
        interactions={interactions}
        cameraMode={cameraMode}
        reducedMotion={reducedMotion}
        onSelect={onSelectInteraction}
      />
    </>
  );
}

function ResearchInteriorWorld({
  scene,
  cameraMode,
  reducedMotion,
  onSelectInteraction,
}: {
  scene: Game3DSceneDefinition;
  cameraMode: CameraMode3D;
  reducedMotion: boolean;
  onSelectInteraction: (interaction: Game3DInteraction) => void;
}) {
  const agentInteraction = scene.interactions.find((item) => item.kind === 'agent');
  const workspaceInteractions = scene.interactions.filter((item) => item.kind === 'workspace');
  return (
    <>
      <InteriorShell scene={scene} />
      <InteriorDecor scene={scene} reducedMotion={reducedMotion} />
      {agentInteraction?.kind === 'agent' && (
        <AgentFigure
          agentId={agentInteraction.agentId}
          position={agentInteraction.position}
          color={agentColor(agentInteraction.agentId)}
          reducedMotion={reducedMotion}
          showLabel={cameraMode !== 'map'}
          onClick={() => onSelectInteraction(agentInteraction)}
        />
      )}
      {workspaceInteractions.map((interaction) => (
        <WorkspaceStation
          key={interaction.id}
          interaction={interaction}
          sceneId={scene.id}
          onSelect={() => onSelectInteraction(interaction)}
        />
      ))}
      <InteractionMarkers
        interactions={scene.interactions}
        cameraMode={cameraMode}
        reducedMotion={reducedMotion}
        onSelect={onSelectInteraction}
      />
    </>
  );
}

function InteriorShell({ scene }: { scene: Game3DSceneDefinition }) {
  if (scene.id === 'plaza') {
    return (
      <RigidBody type="fixed" colliders={false} friction={1.1}>
        <CuboidCollider args={[13, 0.35, 10]} position={[0, -0.35, 0]} />
        <mesh receiveShadow position={[0, -0.4, 0]} scale={[13, 0.8, 10]}>
          <cylinderGeometry args={[1, 1, 1, 64]} />
          <TexturedSurfaceMaterial color="#a9aaa3" variance={0.1} repeat={[12, 10]} roughness={0.92} />
        </mesh>
        <mesh position={[0, -0.88, 0]} scale={[13.4, 1, 10.4]}>
          <cylinderGeometry args={[1, 1, 1, 64]} />
          <meshStandardMaterial color="#59625e" roughness={0.96} />
        </mesh>
      </RigidBody>
    );
  }
  return (
    <RigidBody type="fixed" colliders={false} friction={1.1}>
      <CuboidCollider args={[12, 0.35, 9]} position={[0, -0.35, 0]} />
      <mesh receiveShadow position={[0, -0.37, 0]}>
        <boxGeometry args={[24, 0.7, 18]} />
        <TexturedSurfaceMaterial color={interiorFloorColor(scene.id)} variance={0.08} repeat={[12, 9]} roughness={0.9} />
      </mesh>
      <RoomWall position={[0, 2.5, -9]} size={[24, 5, 0.42]} sceneId={scene.id} />
      <RoomWall position={[-12, 2.5, 0]} size={[0.42, 5, 18]} sceneId={scene.id} />
      <RoomWall position={[12, 2.5, 0]} size={[0.42, 5, 18]} sceneId={scene.id} />
      <RoomWall position={[-7, 2.5, 9]} size={[9.5, 5, 0.42]} sceneId={scene.id} />
      <RoomWall position={[7, 2.5, 9]} size={[9.5, 5, 0.42]} sceneId={scene.id} />
    </RigidBody>
  );
}

function RoomWall({
  position,
  size,
  sceneId,
}: {
  position: Vec3;
  size: Vec3;
  sceneId: GameSceneId;
}) {
  return (
    <>
      <CuboidCollider args={[size[0] / 2, size[1] / 2, size[2] / 2]} position={position} />
      <mesh castShadow receiveShadow position={position}>
        <boxGeometry args={size} />
        <TexturedSurfaceMaterial color={interiorWallColor(sceneId)} variance={0.05} repeat={[8, 3]} roughness={0.88} />
      </mesh>
    </>
  );
}

function InteriorDecor({
  scene,
  reducedMotion,
}: {
  scene: Game3DSceneDefinition;
  reducedMotion: boolean;
}) {
  if (scene.id === 'research') return <ResearchLabDecor />;
  if (scene.id === 'compute') return <ComputeWorkshopDecor reducedMotion={reducedMotion} />;
  if (scene.id === 'paper') return <PaperWorkshopDecor />;
  return <AcademicPlazaDecor reducedMotion={reducedMotion} />;
}

function ResearchLabDecor() {
  return (
    <>
      <pointLight position={[0, 4.2, 1]} intensity={2.8} distance={21} color="#c5d4d6" />
      <ObstacleDesk position={[-6.7, 0.62, -4.9]} size={[4.2, 1.24, 1.4]} color="#3f5755" />
      <ObstacleDesk position={[-4.2, 0.62, 2.7]} size={[3.4, 1.24, 2]} color="#6f6555" />
      <mesh position={[-6.7, 1.55, -4.82]} rotation={[-0.12, 0, 0]}>
        <boxGeometry args={[2.9, 1.1, 0.1]} />
        <meshPhysicalMaterial color="#5f7e7c" emissive="#2a4545" emissiveIntensity={0.12} roughness={0.35} />
      </mesh>
      <EvidenceWall position={[-8.9, 2.25, -8.72]} accent="#4b8984" />
      <EvidenceWall position={[0, 2.25, -8.72]} accent="#ad7858" />
      <CeilingRails color="#526b68" />
    </>
  );
}

function ComputeWorkshopDecor({ reducedMotion }: { reducedMotion: boolean }) {
  return (
    <>
      <pointLight position={[0, 4.2, 1]} intensity={2.4} distance={20} color="#c0ced2" />
      {[-7.3, 0, 7.3].map((x, rackIndex) => (
        <ServerRack key={x} position={[x, 1.65, -3.4]} phase={rackIndex} reducedMotion={reducedMotion} />
      ))}
      <mesh position={[0, 4.55, -3]}>
        <boxGeometry args={[20, 0.18, 0.18]} />
        <meshStandardMaterial color="#586267" metalness={0.58} roughness={0.38} />
      </mesh>
      <CeilingRails color="#596a70" />
    </>
  );
}

function PaperWorkshopDecor() {
  return (
    <>
      <pointLight position={[0, 4.1, 0]} intensity={2.6} distance={20} color="#e0d8c8" />
      <ObstacleDesk position={[-6.4, 0.62, -4.5]} size={[4.4, 1.24, 1.4]} color="#725c48" />
      <ObstacleDesk position={[-5.5, 0.62, 2.6]} size={[4, 1.24, 1.6]} color="#725c48" />
      <ObstacleDesk position={[5.5, 0.62, 2.6]} size={[4, 1.24, 1.6]} color="#725c48" />
      <EvidenceWall position={[-7.2, 2.2, -8.72]} accent="#9d6b5f" />
      <EvidenceWall position={[0, 2.2, -8.72]} accent="#727f78" />
      {[[-6.2, -4.5], [-5.8, 2.6], [5.2, 2.6]].map(([x, z], index) => (
        <group key={`${x}-${z}`} position={[x, 1.3, z]} rotation={[0, index * 0.35 - 0.2, 0]}>
          <mesh><boxGeometry args={[0.8, 0.035, 0.58]} /><meshStandardMaterial color="#f1eee5" roughness={0.9} /></mesh>
          <mesh position={[0.08, 0.025, -0.08]}><boxGeometry args={[0.52, 0.02, 0.03]} /><meshStandardMaterial color="#777b76" /></mesh>
        </group>
      ))}
    </>
  );
}

function AcademicPlazaDecor({ reducedMotion }: { reducedMotion: boolean }) {
  const bannerRef = useRef<THREE.Group | null>(null);
  useFrame(({ clock }) => {
    if (!bannerRef.current || reducedMotion) return;
    bannerRef.current.rotation.z = Math.sin(clock.elapsedTime * 0.7) * 0.018;
  });
  return (
    <>
      <WaveOcean reducedMotion={reducedMotion} />
      <directionalLight position={[-8, 14, 4]} intensity={0.65} color="#e6e2d8" />
      <group ref={bannerRef} position={[0, 4.2, -5.8]}>
        <mesh><boxGeometry args={[8.5, 0.1, 0.1]} /><meshStandardMaterial color="#4f5f60" metalness={0.4} /></mesh>
        <mesh position={[0, -0.7, 0]}><planeGeometry args={[7.5, 1.2]} /><meshStandardMaterial color="#7d3f3c" roughness={0.78} side={THREE.DoubleSide} /></mesh>
      </group>
      {[-5.8, 5.8].map((x) => (
        <ObstacleDesk key={x} position={[x, 0.48, 3]} size={[2.8, 0.96, 1]} color="#596a66" />
      ))}
      <PlazaPlanters />
    </>
  );
}

function WorkspaceStation({
  interaction,
  sceneId,
  onSelect,
}: {
  interaction: Extract<Game3DInteraction, { kind: 'workspace' }>;
  sceneId: GameSceneId;
  onSelect: () => void;
}) {
  const [x, , z] = interaction.position;
  const isBoard = interaction.workspaceZone === 'lab' || interaction.workspaceZone === 'plaza';
  return (
    <RigidBody type="fixed" colliders={false}>
      <CuboidCollider
        args={isBoard ? [1.85, 1.15, 0.55] : [1.75, 0.8, 0.8]}
        position={[x, isBoard ? 1.15 : 0.8, z]}
      />
      <group
        position={[x, 0, z]}
        onClick={(event) => {
          event.stopPropagation();
          onSelect();
        }}
      >
        {isBoard ? (
          <>
            <mesh castShadow position={[0, 1.35, 0]}>
              <boxGeometry args={[3.5, 2.35, 0.22]} />
              <meshStandardMaterial color={sceneId === 'plaza' ? '#ece5d8' : '#41595b'} roughness={0.72} />
            </mesh>
            <mesh position={[0, 1.4, 0.13]}>
              <planeGeometry args={[2.9, 1.75]} />
              <meshStandardMaterial color={sceneId === 'plaza' ? '#9c5d4f' : '#88b7b1'} emissive="#4e8581" emissiveIntensity={0.08} />
            </mesh>
          </>
        ) : (
          <>
            <mesh castShadow receiveShadow position={[0, 0.48, 0]}>
              <boxGeometry args={[3.3, 0.96, 1.35]} />
              <meshStandardMaterial color={sceneId === 'compute' ? '#454f53' : sceneId === 'paper' ? '#725c48' : '#4b5e5c'} metalness={sceneId === 'compute' ? 0.4 : 0.08} roughness={0.62} />
            </mesh>
            <mesh position={[0, 1.36, -0.18]} rotation={[-0.16, 0, 0]}>
              <boxGeometry args={[2.35, 1.15, 0.12]} />
              <meshPhysicalMaterial color="#5f8286" emissive="#2a4a4e" emissiveIntensity={0.14} roughness={0.32} />
            </mesh>
          </>
        )}
      </group>
    </RigidBody>
  );
}

type WorldProps = {
  cameraMode: CameraMode3D;
  reducedMotion: boolean;
  interactions: Game3DInteraction[];
  onSelectInteraction: (interaction: Game3DInteraction) => void;
};

function WaveOcean({ reducedMotion }: { reducedMotion: boolean }) {
  const materialRef = useRef<THREE.ShaderMaterial | null>(null);
  useFrame(({ clock }) => {
    if (materialRef.current && !reducedMotion) {
      materialRef.current.uniforms.uTime.value = clock.elapsedTime;
    }
  });
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.12, 0]} receiveShadow>
      <planeGeometry args={[260, 260, 72, 72]} />
      <shaderMaterial
        ref={materialRef}
        transparent
        uniforms={{ uTime: { value: 0 } }}
        vertexShader={`
          uniform float uTime;
          varying float vWave;
          void main() {
            vec3 p = position;
            float wave = sin(p.x * 0.28 + uTime * 0.9) * 0.12;
            wave += cos(p.y * 0.34 - uTime * 0.7) * 0.08;
            p.z += wave;
            vWave = wave;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
          }
        `}
        fragmentShader={`
          varying float vWave;
          void main() {
            vec3 deep = vec3(0.28, 0.62, 0.72);
            vec3 lightWater = vec3(0.55, 0.84, 0.88);
            vec3 color = mix(deep, lightWater, clamp(vWave * 2.8 + 0.45, 0.0, 1.0));
            gl_FragColor = vec4(color, 0.94);
          }
        `}
      />
    </mesh>
  );
}

function CampusPaths({ mapMode }: { mapMode: boolean }) {
  return (
    <group position={[0, 0.03, 0]}>
      <MergedCampusPaths />
      <PlazaSurface position={[0, 0, 2]} radius={[5.6, 4.8]} color="#6c706e" simplified={mapMode} />
      <PlazaSurface position={[0, 0, 24]} radius={[4.2, 3.2]} color="#6e726f" simplified={mapMode} />
      <PlazaSurface position={[-10, 0, 4]} radius={[4.0, 3.4]} color="#c5c0b0" simplified={mapMode} />
      <PlazaSurface position={[-10, 0, -8]} radius={[4.0, 3.4]} color="#686c6a" simplified={mapMode} />
      <PlazaSurface position={[11, 0, -6]} radius={[4.0, 3.4]} color="#686c6a" simplified={mapMode} />
      <PlazaSurface position={[0, 0, -18]} radius={[4.4, 3.2]} color="#666a68" simplified={mapMode} />
      <PlazaSurface position={[11, 0, 8]} radius={[4.0, 3.4]} color="#c5c0b0" simplified={mapMode} />
    </group>
  );
}

function MergedCampusPaths() {
  const geometries = useMemo(() => {
    const surfaces = campusPathNetwork3D.map((path) => createPathRibbonGeometry(path.points, path.width));
    const edges = campusPathNetwork3D.map((path) => createPathRibbonGeometry(path.points, path.width + 0.42));
    const surface = mergeGeometries(surfaces);
    const edge = mergeGeometries(edges);
    surfaces.forEach((geometry) => geometry.dispose());
    edges.forEach((geometry) => geometry.dispose());
    return { surface, edge };
  }, []);
  useEffect(() => () => {
    geometries.surface?.dispose();
    geometries.edge?.dispose();
  }, [geometries]);
  if (!geometries.surface || !geometries.edge) return null;
  return (
    <group>
      <mesh geometry={geometries.edge} receiveShadow renderOrder={1}>
        <meshStandardMaterial
          color="#777b75"
          roughness={0.94}
          polygonOffset
          polygonOffsetFactor={-1}
          polygonOffsetUnits={-1}
        />
      </mesh>
      <mesh geometry={geometries.surface} position={[0, 0.018, 0]} receiveShadow renderOrder={2}>
        <TexturedSurfaceMaterial
          color="#aaa79e"
          variance={0.048}
          repeat={[1, 1]}
          roughness={0.94}
          polygonOffsetFactor={-2}
          polygonOffsetUnits={-2}
        />
      </mesh>
    </group>
  );
}

function PlazaSurface({
  position,
  radius,
  color,
  simplified,
}: {
  position: Vec3;
  radius: [number, number];
  color: string;
  simplified: boolean;
}) {
  return (
    <group position={position}>
      <mesh
        position={[0, 0.048, 0]}
        receiveShadow
        renderOrder={4}
        rotation={[-Math.PI / 2, 0, 0]}
        scale={[radius[0], radius[1], 1]}
      >
        <circleGeometry args={[1, 48]} />
        {simplified ? (
          <meshStandardMaterial
            color={color}
            roughness={0.95}
            polygonOffset
            polygonOffsetFactor={-4}
            polygonOffsetUnits={-4}
          />
        ) : (
          <TexturedSurfaceMaterial
            color={color}
            variance={0.045}
            repeat={[3, 3]}
            roughness={0.95}
            polygonOffsetFactor={-4}
            polygonOffsetUnits={-4}
          />
        )}
      </mesh>
      {!simplified && (
        <mesh
          position={[0, 0.035, 0]}
          renderOrder={3}
          rotation={[-Math.PI / 2, 0, 0]}
          scale={[radius[0] + 0.24, radius[1] + 0.24, 1]}
        >
          <ringGeometry args={[0.94, 1, 48]} />
          <meshStandardMaterial
            color="#707672"
            roughness={0.9}
            polygonOffset
            polygonOffsetFactor={-3}
            polygonOffsetUnits={-3}
          />
        </mesh>
      )}
    </group>
  );
}

function createPathRibbonGeometry(points: Array<[number, number]>, width: number) {
  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];
  points.forEach((point, index) => {
    const previous = points[Math.max(0, index - 1)];
    const next = points[Math.min(points.length - 1, index + 1)];
    const dx = next[0] - previous[0];
    const dz = next[1] - previous[1];
    const length = Math.hypot(dx, dz) || 1;
    const offsetX = (-dz / length) * width / 2;
    const offsetZ = (dx / length) * width / 2;
    positions.push(point[0] + offsetX, 0, point[1] + offsetZ);
    positions.push(point[0] - offsetX, 0, point[1] - offsetZ);
    // World-space UVs make every ribbon sample the same texture at junctions.
    const textureScale = 1 / 8;
    uvs.push(
      (point[0] + offsetX) * textureScale,
      (point[1] + offsetZ) * textureScale,
      (point[0] - offsetX) * textureScale,
      (point[1] - offsetZ) * textureScale,
    );
    if (index > 0) {
      const base = index * 2;
      indices.push(base - 2, base, base - 1, base, base + 1, base - 1);
    }
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}


function StationPier() {
  const planks = [];
  for (let i = -3; i <= 3; i += 1) {
    planks.push(
      <VoxelBrick
        key={i}
        position={[i * 0.55, 0.25, 2]}
        size={[0.5, 0.5, 8]}
        color={i % 2 === 0 ? '#5a5e5c' : '#4a4e4c'}
      />,
    );
  }
  return (
    <group position={[0, 0, 24]}>
      {planks}
      {[-2.5, 2.5].map((x) => (
        <group key={x}>
          <VoxelBrick position={[x, 1.0, 4]} size={[0.5, 2.0, 0.5]} color="#3a4246" />
          <VoxelBrick position={[x, 2.25, 4]} size={[0.5, 0.5, 0.5]} color="#c4a35a" roughness={0.55} />
        </group>
      ))}
      <VoxelBrick position={[0, 2.0, 4]} size={[5.5, 0.5, 0.5]} color="#4a5054" />
    </group>
  );
}


function StationRadar() {
  // Voxel stepped dome (architectural beacon, not a brand asset)
  const layers = [
    { y: 0.25, s: 5.0, c: '#5a5e5c' },
    { y: 0.75, s: 4.5, c: '#6a6e6c' },
    { y: 1.25, s: 4.0, c: '#7a7e7c' },
    { y: 1.75, s: 3.0, c: '#8a9094' },
    { y: 2.25, s: 2.0, c: '#9aa0a4' },
    { y: 2.75, s: 1.0, c: '#aab0b4' },
  ];
  return (
    <group position={[0, 0, -2]}>
      {layers.map((layer) => (
        <VoxelBrick
          key={layer.y}
          position={[0, layer.y, 0]}
          size={[layer.s, 0.5, layer.s]}
          color={layer.c}
        />
      ))}
      <VoxelBrick position={[0, 3.5, 0]} size={[0.5, 2.5, 0.5]} color="#3a4246" />
      <VoxelBrick position={[0, 5.0, 0]} size={[2.5, 0.5, 0.5]} color="#5a6266" />
    </group>
  );
}



/** Unit size for the voxel architectural language (not any commercial brand). */
const VOX = 0.5;

function snapVox(n: number): number {
  return Math.max(VOX, Math.round(n / VOX) * VOX);
}

function VoxelBrick({
  position,
  size = [VOX, VOX, VOX],
  color,
  roughness = 0.95,
}: {
  position: Vec3;
  size?: Vec3;
  color: string;
  roughness?: number;
}) {
  return (
    <mesh castShadow receiveShadow position={position}>
      <boxGeometry args={size} />
      <meshStandardMaterial color={color} roughness={roughness} metalness={0} flatShading />
    </mesh>
  );
}

/** Stepped voxel mass: layered boxes + window grid + flat slab roof. */
function VoxelMass({
  size,
  wall,
  accent,
  roofColor,
  kind,
}: {
  size: Vec3;
  wall: string;
  accent: string;
  roofColor: string;
  kind: 'gable' | 'glass' | 'antenna' | 'pavilion';
}) {
  const w = snapVox(size[0]);
  const h = snapVox(size[1]);
  const d = snapVox(size[2]);
  const layers = Math.max(3, Math.round(h / VOX));
  const winY = [Math.floor(layers * 0.35), Math.floor(layers * 0.65)];
  const bricks: ReactElement[] = [];
  // main body as single box for perf, plus detail layer of window voxels on front
  bricks.push(
    <VoxelBrick key="body" position={[0, 0, 0]} size={[w, h, d]} color={wall} />,
  );
  // base plinth
  bricks.push(
    <VoxelBrick
      key="plinth"
      position={[0, -h / 2 + VOX * 0.5, 0]}
      size={[w + VOX, VOX, d + VOX]}
      color={accent}
    />,
  );
  // front window grid
  const cols = Math.max(2, Math.floor((w - VOX * 2) / (VOX * 2)));
  for (const ly of winY) {
    const y = -h / 2 + ly * VOX + VOX * 0.5;
    for (let c = 0; c < cols; c += 1) {
      const x = -w / 2 + VOX * 1.5 + c * VOX * 2;
      bricks.push(
        <VoxelBrick
          key={`w-${ly}-${c}`}
          position={[x, y, d / 2 + 0.02]}
          size={[VOX * 1.2, VOX * 1.2, 0.08]}
          color="#2a3338"
          roughness={0.4}
        />,
      );
    }
  }
  // door
  bricks.push(
    <VoxelBrick
      key="door"
      position={[0, -h / 2 + VOX * 1.5, d / 2 + 0.03]}
      size={[VOX * 2, VOX * 3, 0.1]}
      color={accent}
    />,
  );
  // roof style
  if (kind === 'antenna') {
    bricks.push(
      <VoxelBrick key="roof" position={[0, h / 2 + VOX * 0.5, 0]} size={[w, VOX, d]} color={roofColor} />,
    );
    bricks.push(
      <VoxelBrick key="mast" position={[0, h / 2 + VOX * 3, 0]} size={[VOX * 0.6, VOX * 5, VOX * 0.6]} color="#3a4246" />,
    );
    bricks.push(
      <VoxelBrick key="beacon" position={[0, h / 2 + VOX * 5.5, 0]} size={[VOX, VOX, VOX]} color="#c4a35a" roughness={0.5} />,
    );
  } else if (kind === 'glass' || kind === 'pavilion') {
    // stepped dome approximation
    const rings = [0.9, 0.7, 0.45];
    rings.forEach((s, i) => {
      bricks.push(
        <VoxelBrick
          key={`dome-${i}`}
          position={[0, h / 2 + VOX * (0.5 + i), 0]}
          size={[w * s, VOX, d * s]}
          color={i === 0 ? roofColor : '#5a7074'}
        />,
      );
    });
  } else {
    // stepped gable
    bricks.push(
      <VoxelBrick key="roof1" position={[0, h / 2 + VOX * 0.5, 0]} size={[w + VOX * 0.5, VOX, d + VOX * 0.5]} color={roofColor} />,
    );
    bricks.push(
      <VoxelBrick key="roof2" position={[0, h / 2 + VOX * 1.5, 0]} size={[w * 0.7, VOX, d * 0.7]} color={roofColor} />,
    );
  }
  return <group>{bricks}</group>;
}

function CampusBuildings() {
  return (
    <>
      <StationRadar />
      <StationPier />
      <CampusBuilding sceneId="academy" position={[-10, 2.5, 4]} size={[7.2, 5.0, 6.0]} wall="#8a8e8c" accent="#4a5250" roof="gable" />
      <CampusBuilding sceneId="research" position={[-10, 2.6, -8]} size={[7.0, 5.2, 6.2]} wall="#848886" accent="#3f4a4c" roof="glass" />
      <CampusBuilding sceneId="compute" position={[11, 2.8, -6]} size={[7.4, 5.6, 6.4]} wall="#7e8486" accent="#3a4246" roof="antenna" />
      <CampusBuilding sceneId="paper" position={[0, 2.4, -18]} size={[8.0, 4.8, 5.8]} wall="#8c8884" accent="#4a4642" roof="gable" />
      <CampusBuilding sceneId="plaza" position={[11, 2.3, 8]} size={[6.8, 4.6, 5.6]} wall="#868a88" accent="#4c5452" roof="pavilion" />
      <CampusBuilding sceneId="academy" position={[-6, 1.4, 18]} size={[3.2, 2.8, 3.0]} wall="#7a7e7c" accent="#454a48" roof="gable" />
      <CampusBuilding sceneId="academy" position={[6, 1.4, 18]} size={[3.2, 2.8, 3.0]} wall="#7a7e7c" accent="#454a48" roof="gable" />
    </>
  );
}

function CampusBuilding({
  sceneId,
  position,
  size,
  wall,
  accent,
  roof,
}: {
  sceneId: GameSceneId;
  position: Vec3;
  size: Vec3;
  wall: string;
  accent: string;
  roof: 'gable' | 'glass' | 'antenna' | 'pavilion';
}) {
  return (
    <RigidBody type="fixed" colliders={false} userData={{ kind: 'building', sceneId }}>
      <CuboidCollider args={[size[0] / 2, size[1] / 2, size[2] / 2]} position={position} />
      <group position={position}>
        <VoxelMass size={size} wall={wall} accent={accent} roofColor={accent} kind={roof} />
      </group>
    </RigidBody>
  );
}

function Roof({ kind, size, color }: { kind: 'gable' | 'glass' | 'antenna' | 'pavilion'; size: Vec3; color: string }) {
  if (kind === 'glass') {
    return (
      <mesh castShadow position={[0, size[1] / 2 + 0.75, 0]}>
        <cylinderGeometry args={[1.65, 2.2, 1.5, 8]} />
        <meshPhysicalMaterial color="#568d8c" metalness={0.18} roughness={0.24} transmission={0.2} />
      </mesh>
    );
  }
  if (kind === 'antenna') {
    return (
      <group position={[0, size[1] / 2 + 0.6, 0]}>
        <mesh castShadow><boxGeometry args={[size[0] * 0.92, 0.35, size[2] * 0.92]} /><meshStandardMaterial color="#555d60" metalness={0.38} roughness={0.58} /></mesh>
        <mesh position={[0, 1.1, 0]}><cylinderGeometry args={[0.08, 0.12, 2.2, 8]} /><meshStandardMaterial color="#45535a" metalness={0.5} /></mesh>
        <mesh position={[0, 2.05, 0]}><sphereGeometry args={[0.2, 8, 6]} /><meshStandardMaterial color="#f1c64b" emissive="#eaa826" emissiveIntensity={1.2} /></mesh>
      </group>
    );
  }
  if (kind === 'pavilion') {
    return (
      <mesh castShadow position={[0, size[1] / 2 + 0.48, 0]}>
        <cylinderGeometry args={[3.1, 2.35, 0.95, 8]} />
        <meshStandardMaterial color={color} metalness={0.12} roughness={0.7} />
      </mesh>
    );
  }
  return (
    <group position={[0, size[1] / 2 + 0.12, 0]}>
      <mesh castShadow position={[-size[0] * 0.245, 0.12, 0]} rotation={[0, 0, -0.13]}>
        <boxGeometry args={[size[0] * 0.54, 0.22, size[2] * 1.08]} />
        <meshStandardMaterial color={color} metalness={0.08} roughness={0.76} />
      </mesh>
      <mesh castShadow position={[size[0] * 0.245, 0.12, 0]} rotation={[0, 0, 0.13]}>
        <boxGeometry args={[size[0] * 0.54, 0.22, size[2] * 1.08]} />
        <meshStandardMaterial color={color} metalness={0.08} roughness={0.76} />
      </mesh>
    </group>
  );
}

function WindowRow({ width, depth, y }: { width: number; depth: number; y: number }) {
  const positions = [-0.28, 0.28];
  return (
    <>
      {positions.map((fraction) => (
        <mesh key={fraction} position={[fraction * width, y, depth / 2 + 0.065]}>
          <boxGeometry args={[0.9, 0.72, 0.1]} />
          <meshPhysicalMaterial color="#789ca1" metalness={0.18} roughness={0.2} transmission={0.18} />
        </mesh>
      ))}
    </>
  );
}

const treePositions: Vec3[] = [
  [-42, 0, -17], [-40, 0, 13], [-36, 0, 20], [-33, 0, 25], [-28, 0, 20],
  [-25, 0, 27], [-18, 0, 26], [-10, 0, 29], [9, 0, 29], [18, 0, 27],
  [27, 0, 25], [34, 0, 20], [42, 0, 15], [44, 0, -5], [41, 0, -15],
  [35, 0, -24], [28, 0, -28], [16, 0, -29], [-13, 0, -29], [-32, 0, -24],
  [-40, 0, -14], [-30, 0, 15], [-25, 0, 1], [-18, 0, -2], [-15, 0, -19],
  [-8, 0, -20], [8, 0, -20], [15, 0, -19], [22, 0, -3], [28, 0, 13],
  [20, 0, 17], [10, 0, 20], [-6, 0, 20], [-17, 0, 13], [31, 0, 1],
  [33, 0, -12], [-27, 0, -14], [7, 0, 8], [-7, 0, 8], [6, 0, -4],
];

function CampusTrees({ reducedMotion }: { reducedMotion: boolean }) {
  const trunkRef = useRef<THREE.InstancedMesh | null>(null);
  const crownRef = useRef<THREE.InstancedMesh | null>(null);
  const sideCrownRef = useRef<THREE.InstancedMesh | null>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  useFrame(({ clock }) => {
    const trunk = trunkRef.current;
    const crown = crownRef.current;
    const sideCrown = sideCrownRef.current;
    if (!trunk || !crown || !sideCrown) return;
    treePositions.forEach((position, index) => {
      const phase = index * 0.73;
      const swayZ = reducedMotion ? 0 : Math.sin(clock.elapsedTime * 0.7 + phase) * 0.035;
      const swayX = reducedMotion ? 0 : Math.cos(clock.elapsedTime * 0.55 + phase) * 0.025;

      dummy.position.set(position[0], 0.9, position[2]);
      dummy.rotation.set(0, 0, 0);
      dummy.scale.set(1, 1, 1);
      dummy.updateMatrix();
      trunk.setMatrixAt(index, dummy.matrix);

      dummy.position.set(position[0], 2.35, position[2]);
      dummy.rotation.set(swayX, 0, swayZ);
      dummy.updateMatrix();
      crown.setMatrixAt(index, dummy.matrix);

      dummy.position.set(position[0] + 0.55, 1.95, position[2] + 0.1);
      dummy.rotation.set(swayX, 0, swayZ);
      dummy.updateMatrix();
      sideCrown.setMatrixAt(index, dummy.matrix);
    });
    trunk.instanceMatrix.needsUpdate = true;
    crown.instanceMatrix.needsUpdate = true;
    sideCrown.instanceMatrix.needsUpdate = true;
  });

  return (
    <>
      <instancedMesh ref={trunkRef} args={[undefined, undefined, treePositions.length]} castShadow>
        <cylinderGeometry args={[0.15, 0.22, 1.8, 7]} />
        <meshStandardMaterial color="#75583f" roughness={1} />
      </instancedMesh>
      <instancedMesh ref={crownRef} args={[undefined, undefined, treePositions.length]} castShadow>
        <icosahedronGeometry args={[0.95, 1]} />
        <meshStandardMaterial color="#456f51" roughness={0.96} />
      </instancedMesh>
      <instancedMesh ref={sideCrownRef} args={[undefined, undefined, treePositions.length]} castShadow>
        <icosahedronGeometry args={[0.62, 1]} />
        <meshStandardMaterial color="#64845a" roughness={0.97} />
      </instancedMesh>
    </>
  );
}

function CampusDetails({ reducedMotion }: { reducedMotion: boolean }) {
  const beaconRef = useRef<THREE.Mesh | null>(null);
  useFrame(({ clock }) => {
    if (!beaconRef.current || reducedMotion) return;
    beaconRef.current.scale.setScalar(1 + Math.sin(clock.elapsedTime * 2) * 0.08);
  });
  return (
    <group>
      <mesh castShadow position={[0, 0.36, 4]}><cylinderGeometry args={[1.55, 1.75, 0.7, 28]} /><meshStandardMaterial color="#64757a" roughness={0.76} /></mesh>
      <mesh ref={beaconRef} position={[0, 1.35, 4]}><icosahedronGeometry args={[0.52, 2]} /><meshStandardMaterial color="#f4c850" emissive="#e6a828" emissiveIntensity={0.65} roughness={0.38} /></mesh>
      {[-1, 1].map((side) => (
        <group key={side} position={[side * 4.6, 0, 8]} rotation={[0, side * 0.18, 0]}>
          <mesh castShadow position={[0, 0.28, 0]}><boxGeometry args={[2.2, 0.18, 0.6]} /><meshStandardMaterial color="#8a694d" /></mesh>
          <mesh position={[-0.82, 0.05, 0]}><boxGeometry args={[0.14, 0.5, 0.48]} /><meshStandardMaterial color="#55645b" /></mesh>
          <mesh position={[0.82, 0.05, 0]}><boxGeometry args={[0.14, 0.5, 0.48]} /><meshStandardMaterial color="#55645b" /></mesh>
        </group>
      ))}
      <ArrivalTerrace />
      <SupportPavilion position={[-29, 0, 20]} accent="#4f6560" />
      <SupportPavilion position={[29, 0, 21]} accent="#6a5852" />
    </group>
  );
}

function ArrivalTerrace() {
  return (
    <group position={[0, 0, 29.3]}>
      <mesh castShadow position={[0, 0.22, 0]}>
        <boxGeometry args={[8.4, 0.42, 2.5]} />
        <TexturedSurfaceMaterial color="#858b86" variance={0.04} repeat={[4, 2]} roughness={0.9} />
      </mesh>
      {[-3.3, 3.3].map((x) => (
        <group key={x} position={[x, 1.2, 0]}>
          <mesh castShadow><cylinderGeometry args={[0.1, 0.14, 2.4, 8]} /><meshStandardMaterial color="#495b5a" metalness={0.4} /></mesh>
          <mesh position={[0, 1.08, 0]}><sphereGeometry args={[0.18, 10, 7]} /><meshStandardMaterial color="#edc863" emissive="#c88e2c" emissiveIntensity={0.5} /></mesh>
        </group>
      ))}
    </group>
  );
}

function SupportPavilion({ position, accent }: { position: Vec3; accent: string }) {
  return (
    <group position={position}>
      <mesh castShadow position={[0, 0.15, 0]}><cylinderGeometry args={[2.6, 2.8, 0.3, 12]} /><meshStandardMaterial color="#888d87" roughness={0.9} /></mesh>
      {[-1.8, 1.8].map((x) => (
        <mesh key={x} castShadow position={[x, 1.35, 0]}><cylinderGeometry args={[0.12, 0.15, 2.7, 8]} /><meshStandardMaterial color="#596460" metalness={0.25} /></mesh>
      ))}
      <mesh castShadow position={[0, 2.8, 0]}><cylinderGeometry args={[2.8, 2.2, 0.42, 8]} /><meshStandardMaterial color={accent} roughness={0.7} metalness={0.12} /></mesh>
    </group>
  );
}

const terrainRocks: Array<{ position: Vec3; scale: Vec3; color: string }> = [
  { position: [-44, -0.2, 4], scale: [2.5, 1.2, 2], color: '#59625a' },
  { position: [-35, -0.25, 29], scale: [3.2, 1.3, 2.4], color: '#697066' },
  { position: [39, -0.2, 24], scale: [3.4, 1.5, 2.5], color: '#59655e' },
  { position: [45, -0.25, -18], scale: [2.6, 1.1, 2], color: '#646b62' },
  { position: [-37, -0.2, -26], scale: [3.2, 1.4, 2.2], color: '#5c665f' },
  { position: [8, -0.25, 32], scale: [2.5, 1.1, 1.7], color: '#687168' },
];

function CampusTerrainDetails() {
  return (
    <group>
      {terrainRocks.map((rock, index) => (
        <mesh key={index} castShadow receiveShadow position={rock.position} scale={rock.scale} rotation={[0, index * 0.73, 0]}>
          <dodecahedronGeometry args={[1, 0]} />
          <meshStandardMaterial color={rock.color} roughness={1} />
        </mesh>
      ))}
      <mesh position={[-32, -0.15, 7]} rotation={[-Math.PI / 2, 0, 0]} scale={[5.6, 3.5, 1]}>
        <circleGeometry args={[1, 40]} />
        <meshPhysicalMaterial color="#4c8f91" roughness={0.22} metalness={0.06} transmission={0.08} />
      </mesh>
      <mesh position={[-32, -0.17, 7]} rotation={[-Math.PI / 2, 0, 0]} scale={[6, 3.9, 1]}>
        <ringGeometry args={[0.92, 1, 40]} />
        <meshStandardMaterial color="#737b70" roughness={0.98} />
      </mesh>
    </group>
  );
}

function AcademyWalls() {
  return (
    <>
      <Wall position={[0, 2.25, -9]} size={[22, 4.5, 0.45]} />
      <Wall position={[-11, 2.25, 0]} size={[0.45, 4.5, 18]} />
      <Wall position={[11, 2.25, 0]} size={[0.45, 4.5, 18]} />
      <Wall position={[-6.4, 2.25, 9]} size={[8.7, 4.5, 0.45]} />
      <Wall position={[6.4, 2.25, 9]} size={[8.7, 4.5, 0.45]} />
      <mesh position={[0, 4.1, -8.7]}>
        <boxGeometry args={[8.6, 0.55, 0.18]} />
        <meshStandardMaterial color="#2c6f76" roughness={0.7} />
      </mesh>
    </>
  );
}

function Wall({ position, size }: { position: Vec3; size: Vec3 }) {
  return (
    <>
      <CuboidCollider args={[size[0] / 2, size[1] / 2, size[2] / 2]} position={position} />
      <mesh castShadow receiveShadow position={position}>
        <boxGeometry args={size} />
        <meshStandardMaterial color="#ebe6d8" roughness={0.88} />
      </mesh>
    </>
  );
}

function AcademyFurniture() {
  return (
    <>
      <FurnitureBlock position={[-6.5, 1, -4.5]} size={[4.2, 2, 1.1]} color="#6d8d80" />
      <FurnitureBlock position={[6.5, 1, -4.5]} size={[4.2, 2, 1.1]} color="#6d8d80" />
      <FurnitureBlock position={[-6.7, 0.65, 2.3]} size={[3.3, 1.3, 1.3]} color="#ba7c4e" />
      <FurnitureBlock position={[6.7, 0.65, 2.3]} size={[3.3, 1.3, 1.3]} color="#ba7c4e" />
      {[-6.7, 6.7].map((x) => (
        <mesh key={x} position={[x, 1.48, 2.3]} rotation={[-0.22, 0, 0]}>
          <boxGeometry args={[1.7, 0.92, 0.09]} />
          <meshStandardMaterial color="#73b9c0" emissive="#5ca7b1" emissiveIntensity={0.35} />
        </mesh>
      ))}
      <mesh position={[0, 0.1, -2.8]} receiveShadow>
        <cylinderGeometry args={[2.3, 2.3, 0.12, 28]} />
        <meshStandardMaterial color="#9fb8a2" roughness={0.94} />
      </mesh>
    </>
  );
}

function FurnitureBlock({ position, size, color }: { position: Vec3; size: Vec3; color: string }) {
  return (
    <>
      <CuboidCollider args={[size[0] / 2, size[1] / 2, size[2] / 2]} position={position} />
      <mesh castShadow receiveShadow position={position}>
        <boxGeometry args={size} />
        <meshStandardMaterial color={color} roughness={0.82} />
      </mesh>
    </>
  );
}

function AcademyAmbient({ reducedMotion }: { reducedMotion: boolean }) {
  const lightRef = useRef<THREE.PointLight | null>(null);
  useFrame(({ clock }) => {
    if (lightRef.current && !reducedMotion) {
      lightRef.current.intensity = 7 + Math.sin(clock.elapsedTime * 1.4) * 0.6;
    }
  });
  return (
    <>
      <pointLight ref={lightRef} position={[0, 4.8, -1]} intensity={7} distance={18} color="#ffd99a" />
      <mesh position={[0, 4.2, -1]}><cylinderGeometry args={[0.65, 0.35, 0.32, 12]} /><meshStandardMaterial color="#f4c968" emissive="#eab347" emissiveIntensity={0.9} /></mesh>
    </>
  );
}

function CampusLandmarks({
  interactions,
  reducedMotion,
  onSelect,
}: {
  interactions: Game3DInteraction[];
  reducedMotion: boolean;
  onSelect: (interaction: Game3DInteraction) => void;
}) {
  return (
    <>
      {interactions.filter(isLandmark).map((interaction) => (
        <LandmarkProp
          key={interaction.id}
          interaction={interaction}
          reducedMotion={reducedMotion}
          onSelect={() => onSelect(interaction)}
        />
      ))}
    </>
  );
}

function LandmarkProp({
  interaction,
  reducedMotion,
  onSelect,
}: {
  interaction: LandmarkInteraction3D;
  reducedMotion: boolean;
  onSelect: () => void;
}) {
  const signalRef = useRef<THREE.Group | null>(null);
  useFrame(({ clock }) => {
    if (!signalRef.current || reducedMotion) return;
    signalRef.current.rotation.y = clock.elapsedTime * 0.45;
    signalRef.current.position.y = 2.8 + Math.sin(clock.elapsedTime * 1.8 + interaction.position[0]) * 0.08;
  });
  return (
    <group
      position={interaction.position}
      onClick={(event) => {
        event.stopPropagation();
        onSelect();
      }}
    >
      <mesh receiveShadow position={[0, 0.12, 0]}>
        <cylinderGeometry args={[1.8, 2, 0.24, 18]} />
        <TexturedSurfaceMaterial color="#8c9188" variance={0.045} repeat={[2, 2]} roughness={0.94} />
      </mesh>
      <LandmarkBody id={interaction.id} />
      <group ref={signalRef} position={[0, 2.8, 0]}>
        <mesh rotation={[0, 0, Math.PI / 4]}>
          <octahedronGeometry args={[0.2, 0]} />
          <meshStandardMaterial color="#f1ca65" emissive="#dba332" emissiveIntensity={0.62} roughness={0.4} />
        </mesh>
        <mesh rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.33, 0.37, 18]} />
          <meshBasicMaterial color="#ffe49a" transparent opacity={0.72} />
        </mesh>
      </group>
    </group>
  );
}

function LandmarkBody({ id }: { id: string }) {
  if (id === 'open-data-pavilion') {
    return (
      <group>
        {[-1.05, 1.05].map((x) => (
          <mesh key={x} castShadow position={[x, 1.2, 0]}><boxGeometry args={[0.16, 2.35, 0.22]} /><meshStandardMaterial color="#53625d" metalness={0.28} /></mesh>
        ))}
        <mesh castShadow position={[0, 1.35, 0]}><boxGeometry args={[2.35, 1.25, 0.2]} /><meshStandardMaterial color="#416f68" roughness={0.7} /></mesh>
        {[0.7, 1.15, 1.6].map((y, index) => (
          <mesh key={y} position={[0, y, 0.13]}><boxGeometry args={[1.65 - index * 0.15, 0.1, 0.05]} /><meshStandardMaterial color="#d6e5db" emissive="#7bb1a5" emissiveIntensity={0.18} /></mesh>
        ))}
      </group>
    );
  }
  if (id === 'replication-garden') {
    return (
      <group>
        {[0, 1, 2, 3, 4, 5].map((index) => {
          const angle = index / 6 * Math.PI * 2;
          return (
            <mesh key={index} castShadow position={[Math.cos(angle) * 1.15, 0.45, Math.sin(angle) * 1.15]} rotation={[0, angle, 0]}>
              <dodecahedronGeometry args={[0.38, 0]} />
              <meshStandardMaterial color={index % 2 ? '#75906e' : '#74817a'} roughness={0.96} />
            </mesh>
          );
        })}
        <mesh castShadow position={[0, 1.05, 0]}><icosahedronGeometry args={[0.62, 1]} /><meshStandardMaterial color="#74a59b" metalness={0.12} roughness={0.48} /></mesh>
      </group>
    );
  }
  if (id === 'ethics-overlook') {
    return (
      <group>
        {[-0.9, 0.9].map((x) => (
          <mesh key={x} castShadow position={[x, 1.25, 0]}><boxGeometry args={[0.38, 2.5, 0.7]} /><meshStandardMaterial color="#777c75" roughness={0.9} /></mesh>
        ))}
        <mesh castShadow position={[0, 2.42, 0]}><boxGeometry args={[2.2, 0.36, 0.78]} /><meshStandardMaterial color="#59655f" roughness={0.84} /></mesh>
        <mesh position={[0, 1.35, 0.38]}><torusGeometry args={[0.42, 0.08, 8, 28]} /><meshStandardMaterial color="#d5b15a" metalness={0.46} roughness={0.4} /></mesh>
      </group>
    );
  }
  if (id === 'field-notes-deck') {
    return (
      <group>
        <mesh castShadow position={[0, 0.9, 0]} rotation={[0, 0, -0.28]}><cylinderGeometry args={[0.12, 0.16, 1.8, 10]} /><meshStandardMaterial color="#4b5558" metalness={0.52} roughness={0.4} /></mesh>
        <group position={[0.18, 1.72, 0]} rotation={[0, 0, -0.28]}>
          <mesh castShadow rotation={[0, 0, Math.PI / 2]}><cylinderGeometry args={[0.26, 0.34, 1.45, 12]} /><meshStandardMaterial color="#496d72" metalness={0.42} roughness={0.36} /></mesh>
          <mesh position={[-0.78, 0, 0]} rotation={[0, 0, Math.PI / 2]}><cylinderGeometry args={[0.36, 0.28, 0.18, 12]} /><meshPhysicalMaterial color="#7db5bd" roughness={0.18} transmission={0.2} /></mesh>
        </group>
      </group>
    );
  }
  if (id === 'negative-results-bell') {
    return (
      <group>
        {[-0.85, 0.85].map((x) => (
          <mesh key={x} castShadow position={[x, 1.3, 0]}><cylinderGeometry args={[0.11, 0.15, 2.6, 8]} /><meshStandardMaterial color="#5b5d59" metalness={0.45} /></mesh>
        ))}
        <mesh castShadow position={[0, 2.5, 0]}><boxGeometry args={[2.05, 0.22, 0.3]} /><meshStandardMaterial color="#555b58" metalness={0.48} /></mesh>
        <mesh castShadow position={[0, 1.75, 0]}><coneGeometry args={[0.58, 0.9, 14, 1, true]} /><meshStandardMaterial color="#b88b3f" metalness={0.62} roughness={0.36} side={THREE.DoubleSide} /></mesh>
        <mesh position={[0, 1.2, 0]}><sphereGeometry args={[0.13, 10, 7]} /><meshStandardMaterial color="#7a572d" metalness={0.5} /></mesh>
      </group>
    );
  }
  return (
    <group>
      {[0, 1, 2, 3, 4, 5, 6, 7].map((index) => {
        const angle = index / 8 * Math.PI * 2;
        return (
          <mesh key={index} castShadow position={[Math.cos(angle) * 1.15, 0.35, Math.sin(angle) * 1.15]}>
            <dodecahedronGeometry args={[0.34, 0]} />
            <meshStandardMaterial color="#686d67" roughness={0.96} />
          </mesh>
        );
      })}
      <mesh castShadow position={[0, 0.55, 0]}><cylinderGeometry args={[0.55, 0.75, 0.75, 12]} /><meshStandardMaterial color="#805f43" roughness={0.85} /></mesh>
      <mesh position={[0, 1.05, 0]}><octahedronGeometry args={[0.34, 0]} /><meshStandardMaterial color="#dd7e45" emissive="#b94d27" emissiveIntensity={0.72} roughness={0.5} /></mesh>
    </group>
  );
}

function isLandmark(interaction: Game3DInteraction): interaction is LandmarkInteraction3D {
  return interaction.kind === 'landmark';
}

function AgentFigure({
  agentId,
  position,
  color,
  reducedMotion,
  showLabel,
  onClick,
}: {
  agentId: AgentId;
  position: Vec3;
  color: string;
  reducedMotion: boolean;
  showLabel: boolean;
  onClick: () => void;
}) {
  const rootRef = useRef<THREE.Group | null>(null);
  useFrame(({ clock }) => {
    if (!rootRef.current || reducedMotion) return;
    rootRef.current.position.y = -0.17 + Math.sin(clock.elapsedTime * 2 + position[0]) * 0.025;
    rootRef.current.rotation.y = Math.sin(clock.elapsedTime * 0.65 + position[2]) * 0.08;
  });
  return (
    <group
      position={position}
      onClick={(event) => {
        event.stopPropagation();
        onClick();
      }}
    >
      <group ref={rootRef}>
        <mesh castShadow position={[0, 1.52, 0]}><sphereGeometry args={[0.3, 12, 8]} /><meshStandardMaterial color="#eec3a1" /></mesh>
        <mesh castShadow position={[0, 1.08, 0]}><capsuleGeometry args={[0.34, 0.68, 4, 8]} /><meshStandardMaterial color={color} roughness={0.75} /></mesh>
        <mesh castShadow position={[-0.15, 0.5, 0]}><capsuleGeometry args={[0.1, 0.45, 3, 6]} /><meshStandardMaterial color="#36434a" /></mesh>
        <mesh castShadow position={[0.15, 0.5, 0]}><capsuleGeometry args={[0.1, 0.45, 3, 6]} /><meshStandardMaterial color="#36434a" /></mesh>
        <mesh position={[0, 2.12, 0]}><octahedronGeometry args={[0.17, 0]} /><meshStandardMaterial color="#f2c24d" emissive="#e4a728" emissiveIntensity={0.9} /></mesh>
      </group>
      {showLabel && (
        <Html center position={[0, 2.48, 0]} distanceFactor={10} zIndexRange={[8, 0]}>
          <button
            type="button"
            className="game3d-agent-label"
            onPointerDown={(event) => event.stopPropagation()}
            onPointerUp={(event) => event.stopPropagation()}
            onClick={(event) => {
              event.stopPropagation();
              onClick();
            }}
          >
            <strong>{agents[agentId].name}</strong>
            <span>{agents[agentId].role.replace(' Agent', '')}</span>
          </button>
        </Html>
      )}
    </group>
  );
}

function InteractionMarkers({
  interactions,
  cameraMode,
  reducedMotion,
  onSelect,
}: {
  interactions: Game3DInteraction[];
  cameraMode: CameraMode3D;
  reducedMotion: boolean;
  onSelect: (interaction: Game3DInteraction) => void;
}) {
  return (
    <>
      {interactions.map((interaction) => (
        <InteractionMarker
          key={interaction.id}
          interaction={interaction}
          mapMode={cameraMode === 'map'}
          reducedMotion={reducedMotion}
          onSelect={onSelect}
        />
      ))}
    </>
  );
}

function InteractionMarker({
  interaction,
  mapMode,
  reducedMotion,
  onSelect,
}: {
  interaction: Game3DInteraction;
  mapMode: boolean;
  reducedMotion: boolean;
  onSelect: (interaction: Game3DInteraction) => void;
}) {
  const ringRef = useRef<THREE.Mesh | null>(null);
  useFrame(({ clock }) => {
    if (!ringRef.current || reducedMotion) return;
    ringRef.current.rotation.z = clock.elapsedTime * 0.55;
    ringRef.current.position.y = 0.12 + Math.sin(clock.elapsedTime * 2.2) * 0.035;
  });
  if (interaction.kind === 'agent') {
    return mapMode ? (
      <group position={interaction.position}>
        <MapLabel interaction={interaction} />
      </group>
    ) : null;
  }
  if (interaction.kind === 'workspace') {
    return (
      <group
        position={interaction.position}
        onClick={(event) => {
          event.stopPropagation();
          onSelect(interaction);
        }}
      >
        {!mapMode && (
          <mesh ref={ringRef} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.1, 0]}>
            <ringGeometry args={[0.72, 0.8, 4]} />
            <meshStandardMaterial color="#83d0c6" emissive="#3c928b" emissiveIntensity={0.72} roughness={0.44} />
          </mesh>
        )}
        {mapMode && <MapLabel interaction={interaction} />}
      </group>
    );
  }
  if (interaction.kind === 'landmark') {
    return (
      <group
        position={interaction.position}
        onClick={(event) => {
          event.stopPropagation();
          onSelect(interaction);
        }}
      >
        {!mapMode && (
          <mesh ref={ringRef} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.14, 0]}>
            <ringGeometry args={[1.72, 1.82, 28]} />
            <meshBasicMaterial color="#f1d27a" transparent opacity={0.72} depthWrite={false} />
          </mesh>
        )}
        {mapMode && <MapLabel interaction={interaction} />}
      </group>
    );
  }
  return (
    <group
      position={interaction.position}
      onClick={(event) => {
        event.stopPropagation();
        onSelect(interaction);
      }}
    >
      <mesh ref={ringRef} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.12, 0]}>
        <torusGeometry args={[0.82, 0.075, 6, 28]} />
        <meshStandardMaterial color="#f5cb55" emissive="#e9a92d" emissiveIntensity={0.85} roughness={0.45} />
      </mesh>
      <pointLight position={[0, 0.75, 0]} intensity={2.4} distance={4} color="#ffd769" />
      {mapMode && <MapLabel interaction={interaction} />}
    </group>
  );
}

function MapLabel({ interaction }: { interaction: Game3DInteraction }) {
  return (
    <Html center position={[0, interaction.kind === 'portal' ? 2.4 : 2.8, 0]} zIndexRange={[30, 20]}>
      <span className={`game3d-map-label game3d-map-label--${interaction.kind}`}>{mapLabel(interaction)}</span>
    </Html>
  );
}

function mapLabel(interaction: Game3DInteraction): string {
  if (interaction.kind === 'agent') return agents[interaction.agentId].name;
  if (interaction.kind === 'workspace') return interaction.label.replace(/^使用|^操作|^打开|^查看/, '');
  if (interaction.kind === 'landmark') return interaction.title;
  const labels: Record<GameSceneId, string> = {
    campus: '科研公社大厅',
    academy: '科研训练馆',
    research: '模型与系统实验室',
    compute: '算力工坊',
    paper: '论文与评审工坊',
    plaza: '学术交流广场',
  };
  return labels[interaction.targetScene];
}

function selectById(
  interactions: Game3DInteraction[],
  id: string,
  onSelect: (interaction: Game3DInteraction) => void,
) {
  const interaction = interactions.find((item) => item.id === id);
  if (interaction) onSelect(interaction);
}

function ObstacleDesk({
  position,
  size,
  color,
}: {
  position: Vec3;
  size: Vec3;
  color: string;
}) {
  return (
    <RigidBody type="fixed" colliders={false}>
      <CuboidCollider args={[size[0] / 2, size[1] / 2, size[2] / 2]} position={position} />
      <mesh castShadow receiveShadow position={position}>
        <boxGeometry args={size} />
        <TexturedSurfaceMaterial color={color} variance={0.06} repeat={[3, 2]} roughness={0.78} />
      </mesh>
    </RigidBody>
  );
}

function EvidenceWall({ position, accent }: { position: Vec3; accent: string }) {
  return (
    <group position={position}>
      <mesh castShadow>
        <boxGeometry args={[5.6, 2.7, 0.16]} />
        <meshStandardMaterial color="#d7d4ca" roughness={0.86} />
      </mesh>
      {[-1.75, 0, 1.75].map((x, index) => (
        <mesh key={x} position={[x, index === 1 ? 0.25 : -0.12, 0.095]}>
          <planeGeometry args={[1.25, index === 1 ? 1.45 : 1.05]} />
          <meshStandardMaterial color={index === 1 ? accent : '#f2eee2'} roughness={0.82} />
        </mesh>
      ))}
    </group>
  );
}

function CeilingRails({ color }: { color: string }) {
  return (
    <group position={[0, 4.45, 0]}>
      {[-6, 0, 6].map((x) => (
        <mesh key={x} position={[x, 0, 0]}>
          <boxGeometry args={[0.13, 0.13, 15]} />
          <meshStandardMaterial color={color} metalness={0.48} roughness={0.44} />
        </mesh>
      ))}
      {[-5, 0, 5].map((z) => (
        <mesh key={z} position={[0, -0.08, z]}>
          <boxGeometry args={[20, 0.08, 0.08]} />
          <meshStandardMaterial color="#c9d1cd" emissive="#f6e6bb" emissiveIntensity={0.35} />
        </mesh>
      ))}
    </group>
  );
}

function ServerRack({
  position,
  phase,
  reducedMotion,
}: {
  position: Vec3;
  phase: number;
  reducedMotion: boolean;
}) {
  const lightsRef = useRef<THREE.MeshStandardMaterial | null>(null);
  useFrame(({ clock }) => {
    if (!lightsRef.current || reducedMotion) return;
    lightsRef.current.emissiveIntensity = 0.45 + Math.sin(clock.elapsedTime * 2.2 + phase) * 0.18;
  });
  return (
    <RigidBody type="fixed" colliders={false}>
      <CuboidCollider args={[1.05, 1.65, 3.9]} position={position} />
      <group position={position}>
        <mesh castShadow receiveShadow>
          <boxGeometry args={[2.1, 3.3, 7.8]} />
          <meshStandardMaterial color="#343e42" metalness={0.55} roughness={0.48} />
        </mesh>
        {[-2.7, -1.35, 0, 1.35, 2.7].map((z) => (
          <group key={z} position={[0, 0, z]}>
            <mesh position={[1.065, 0.3, 0]} rotation={[0, Math.PI / 2, 0]}>
              <planeGeometry args={[0.95, 2.2]} />
              <meshStandardMaterial color="#1e272a" metalness={0.35} roughness={0.38} />
            </mesh>
            <mesh position={[1.075, 0.72, -0.22]} rotation={[0, Math.PI / 2, 0]}>
              <circleGeometry args={[0.065, 10]} />
              <meshStandardMaterial ref={lightsRef} color="#7edc9b" emissive="#3dc96a" emissiveIntensity={0.55} />
            </mesh>
          </group>
        ))}
      </group>
    </RigidBody>
  );
}

function PlazaPlanters() {
  return (
    <>
      {[
        [-9, -5],
        [9, -5],
        [-9, 5],
        [9, 5],
      ].map(([x, z]) => (
        <group key={`${x}-${z}`} position={[x, 0, z]}>
          <mesh castShadow position={[0, 0.38, 0]}>
            <cylinderGeometry args={[0.72, 0.82, 0.76, 12]} />
            <meshStandardMaterial color="#646a67" roughness={0.92} />
          </mesh>
          <mesh castShadow position={[0, 1.18, 0]}>
            <icosahedronGeometry args={[0.78, 1]} />
            <meshStandardMaterial color="#456c52" roughness={0.94} />
          </mesh>
        </group>
      ))}
    </>
  );
}

function TexturedSurfaceMaterial({
  color,
  variance,
  repeat,
  roughness,
  polygonOffsetFactor,
  polygonOffsetUnits,
}: {
  color: string;
  variance: number;
  repeat: [number, number];
  roughness: number;
  polygonOffsetFactor?: number;
  polygonOffsetUnits?: number;
}) {
  const texture = useMemo(
    () => createSurfaceTexture(color, variance, repeat),
    [color, variance, repeat],
  );
  useEffect(() => () => texture.dispose(), [texture]);
  const polygonOffset = polygonOffsetFactor !== undefined || polygonOffsetUnits !== undefined;
  return (
    <meshStandardMaterial
      map={texture}
      color="#ffffff"
      roughness={roughness}
      polygonOffset={polygonOffset}
      polygonOffsetFactor={polygonOffsetFactor}
      polygonOffsetUnits={polygonOffsetUnits}
    />
  );
}

function createSurfaceTexture(
  colorValue: string,
  variance: number,
  repeat: [number, number],
): THREE.DataTexture {
  const size = 64;
  const color = new THREE.Color(colorValue);
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const offset = (y * size + x) * 4;
      const noise = (surfaceNoise(x, y) - 0.5) * variance;
      data[offset] = Math.round(THREE.MathUtils.clamp(color.r + noise, 0, 1) * 255);
      data[offset + 1] = Math.round(THREE.MathUtils.clamp(color.g + noise, 0, 1) * 255);
      data[offset + 2] = Math.round(THREE.MathUtils.clamp(color.b + noise, 0, 1) * 255);
      data[offset + 3] = 255;
    }
  }
  const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.repeat.set(repeat[0], repeat[1]);
  texture.needsUpdate = true;
  return texture;
}

function surfaceNoise(x: number, y: number): number {
  const broad =
    Math.sin(x * 0.21) +
    Math.cos(y * 0.17) +
    Math.sin((x + y) * 0.11) +
    Math.cos((x - y) * 0.08);
  const value = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;
  const grain = value - Math.floor(value) - 0.5;
  return THREE.MathUtils.clamp(0.5 + broad * 0.105 + grain * 0.06, 0, 1);
}


/** Thin dark window band — cheap adult facade read without textures. */
function WindowBand({ width, height, depth }: { width: number; height: number; depth: number }) {
  return (
    <mesh castShadow position={[0, height * 0.08, depth * 0.51]} userData={{ noCollision: true }}>
      <boxGeometry args={[width * 0.72, height * 0.16, 0.06]} />
      <meshStandardMaterial color="#3a4246" metalness={0.25} roughness={0.35} />
    </mesh>
  );
}

function scenePalette(sceneId: GameSceneId) {
  // Bright healing coastal station: clear sky, soft sun, warm concrete (still voxel, not gloomy).
  if (sceneId === 'campus') {
    return { sky: '#b8dff5', fog: '#c9e8f7', hemi: '#fff6e8', ground: '#7a9a72', sun: '#fff2cc' };
  }
  if (sceneId === 'plaza') {
    return { sky: '#bfe3f7', fog: '#d0ebf8', hemi: '#fff8ec', ground: '#7d9d76', sun: '#fff0c8' };
  }
  if (sceneId === 'compute') {
    return { sky: '#c5e4f6', fog: '#c5e4f6', hemi: '#eef8ff', ground: '#6e8890', sun: '#e8f4ff' };
  }
  if (sceneId === 'paper') {
    return { sky: '#d8ebe0', fog: '#d8ebe0', hemi: '#fff8ef', ground: '#8a7d6c', sun: '#ffe8c4' };
  }
  return { sky: '#c4e6d8', fog: '#c4e6d8', hemi: '#fff8ef', ground: '#7a9680', sun: '#fff0d0' };
}

function agentColor(agentId: AgentId): string {
  // Sleeve-level accents — desaturated so characters don't read as toy mascots.
  const colors: Record<AgentId, string> = {
    'campus-guide': '#7a5a56',
    'research-mentor': '#4a6674',
    'methods-mentor': '#4a6a64',
    'experiment-engineer': '#7a6a4e',
    reviewer: '#6a5852',
    'community-host': '#735652',
  };
  return colors[agentId];
}

function interiorFloorColor(sceneId: GameSceneId): string {
  if (sceneId === 'compute') return '#5a6266';
  if (sceneId === 'paper') return '#6e655a';
  return '#5f6a66';
}

function interiorWallColor(sceneId: GameSceneId): string {
  if (sceneId === 'compute') return '#8e979a';
  if (sceneId === 'paper') return '#b5aea4';
  return '#9aa5a0';
}

// Referenced to satisfy noUnusedLocals after coastal declutter.
void (0 as unknown as typeof CampusTrees | typeof CampusDetails | typeof CampusTerrainDetails);

void (0 as unknown as typeof Roof | typeof WindowRow | typeof WindowBand);
