import { Html } from '@react-three/drei';
import { useFrame, type ThreeEvent } from '@react-three/fiber';
import { CuboidCollider, RigidBody } from '@react-three/rapier';
import { useEffect, useMemo, useRef } from 'react';
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
          cameraMode === 'map' ? (scene.id === 'campus' ? 90 : 55) : 18,
          cameraMode === 'map' ? (scene.id === 'campus' ? 170 : 110) : scene.id === 'campus' ? 88 : 56,
        ]}
      />
      <hemisphereLight args={[palette.hemi, palette.ground, 1.05]} />
      <directionalLight
        castShadow={cameraMode !== 'map'}
        position={[14, 22, 6]}
        intensity={1.15}
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
  return (
    <>
      <WaveOcean reducedMotion={reducedMotion} />
      <RigidBody type="fixed" colliders={false} friction={1.1}>
        <CuboidCollider args={[48, 0.55, 34]} position={[0, -0.55, 0]} />
        <mesh position={[0, -0.5, 0]} scale={[48, 1, 34]} receiveShadow>
          <cylinderGeometry args={[1, 1, 1, 72]} />
          <TexturedSurfaceMaterial color="#5c655c" variance={0.075} repeat={[12, 9]} roughness={0.98} />
        </mesh>
        <mesh position={[0, -1.12, 0]} scale={[48.7, 1.25, 34.7]}>
          <cylinderGeometry args={[1, 1, 1, 72]} />
          <TexturedSurfaceMaterial color="#505852" variance={0.06} repeat={[14, 10]} roughness={1} />
        </mesh>
      </RigidBody>

      <CampusPaths mapMode={cameraMode === 'map'} />
      <CampusTrees reducedMotion={reducedMotion} />
      {cameraMode === 'map' ? (
        <>
          <CampusMapBuildings />
          <CampusMapLandmarks interactions={interactions} />
        </>
      ) : (
        <>
          <CampusBuildings />
          <CampusTerrainDetails />
          <CampusDetails reducedMotion={reducedMotion} />
          <CampusLandmarks
            interactions={interactions}
            reducedMotion={reducedMotion}
            onSelect={onSelectInteraction}
          />
          <AgentFigure
            agentId="campus-guide"
            position={[0.8, 0, 9.5]}
            color="#7a5c56"
            reducedMotion={reducedMotion}
            showLabel
            onClick={() => selectById(interactions, 'guide', onSelectInteraction)}
          />
        </>
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
          <meshStandardMaterial color="#d9d2bc" roughness={0.92} />
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
            vec3 deep = vec3(0.16, 0.49, 0.58);
            vec3 lightWater = vec3(0.32, 0.68, 0.71);
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
      <PlazaSurface position={[0, 0, 4]} radius={[7.4, 6.8]} color="#9a968c" simplified={mapMode} />
      <PlazaSurface position={[0, 0, 27]} radius={[5.4, 4.1]} color="#8a867c" simplified={mapMode} />
      <PlazaSurface position={[-23, 0, -18]} radius={[4.7, 3.5]} color="#8c887e" simplified={mapMode} />
      <PlazaSurface position={[-38, 0, 1]} radius={[4.3, 4.1]} color="#7f8a85" simplified={mapMode} />
      <PlazaSurface position={[24, 0, -18]} radius={[4.7, 3.5]} color="#868b8a" simplified={mapMode} />
      <PlazaSurface position={[0, 0, -25]} radius={[5.1, 3.5]} color="#8e877f" simplified={mapMode} />
      <PlazaSurface position={[39, 0, 9]} radius={[4.5, 4]} color="#8f827c" simplified={mapMode} />
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

function CampusBuildings() {
  return (
    <>
      <CampusBuilding
        sceneId="academy"
        position={[-23, 1.9, -22]}
        size={[9, 3.8, 7]}
        wall="#e9dfc9"
        accent="#5c4e48"
        roof="gable"
      />
      <CampusBuilding
        sceneId="research"
        position={[-38, 2, -3]}
        size={[8, 4, 8]}
        wall="#d5e8dc"
        accent="#3f5552"
        roof="glass"
      />
      <CampusBuilding
        sceneId="compute"
        position={[24, 2.1, -22]}
        size={[9, 4.2, 7]}
        wall="#d9e0e2"
        accent="#6a6558"
        roof="antenna"
      />
      <CampusBuilding
        sceneId="paper"
        position={[0, 1.8, -30]}
        size={[10, 3.6, 6.5]}
        wall="#eee3d5"
        accent="#5a4d47"
        roof="gable"
      />
      <CampusBuilding
        sceneId="plaza"
        position={[39, 1.7, 4]}
        size={[8, 3.4, 9]}
        wall="#e8d7d0"
        accent="#5c4a46"
        roof="pavilion"
      />
    </>
  );
}

const campusMapBuildingDefinitions: Array<{
  position: Vec3;
  size: Vec3;
  wall: string;
  roof: string;
}> = [
  // Adult massing: ~2 storey heights, readable silhouettes (academy dome / lab / compute / paper / plaza)
  { position: [-23, 2.35, -22], size: [10.5, 4.7, 8.2], wall: '#b7b3a8', roof: '#5c4e48' },
  { position: [-38, 2.55, -3], size: [9.2, 5.1, 9.2], wall: '#aeb6b2', roof: '#3f5552' },
  { position: [24, 2.7, -22], size: [10.2, 5.4, 8], wall: '#aeb2b4', roof: '#4a5154' },
  { position: [0, 2.2, -30], size: [12, 4.4, 7.4], wall: '#b8b2a8', roof: '#5a4d47' },
  { position: [39, 2.1, 4], size: [9, 4.2, 10], wall: '#b4aaa6', roof: '#5c4a46' },
  // Secondary masses for density without decoration clutter
  { position: [-12, 1.6, 16], size: [4.2, 3.2, 3.6], wall: '#a8aba6', roof: '#555850' },
  { position: [14, 1.55, 17], size: [3.8, 3.1, 3.4], wall: '#a6aaa8', roof: '#545a58' },
  { position: [8, 1.7, -12], size: [3.4, 3.4, 3.2], wall: '#b0aea8', roof: '#5a524c' },
];

function CampusMapBuildings() {
  return (
    <>
      {campusMapBuildingDefinitions.map((building, index) => (
        <group key={index} position={building.position}>
          <mesh>
            <boxGeometry args={building.size} />
            <meshStandardMaterial color={building.wall} roughness={0.9} />
          </mesh>
          <WindowBand width={building.size[0]} height={building.size[1]} depth={building.size[2]} />
          <mesh position={[0, building.size[1] / 2 + 0.32, 0]}>
            <boxGeometry args={[building.size[0] * 1.04, 0.58, building.size[2] * 1.04]} />
            <meshStandardMaterial color={building.roof} roughness={0.76} />
          </mesh>
        </group>
      ))}
    </>
  );
}

function CampusMapLandmarks({ interactions }: { interactions: Game3DInteraction[] }) {
  return (
    <>
      {interactions.filter(isLandmark).map((interaction) => (
        <group key={interaction.id} position={interaction.position}>
          <mesh position={[0, 0.28, 0]}>
            <cylinderGeometry args={[1.35, 1.55, 0.56, 10]} />
            <meshStandardMaterial color="#818982" roughness={0.94} />
          </mesh>
          <mesh position={[0, 1.28, 0]}>
            <octahedronGeometry args={[0.52, 0]} />
            <meshStandardMaterial color="#9a9588" emissive="#4a4840" emissiveIntensity={0.08} />
          </mesh>
        </group>
      ))}
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
        <mesh castShadow receiveShadow>
          <boxGeometry args={size} />
          <TexturedSurfaceMaterial color={wall} variance={0.04} repeat={[3, 2]} roughness={0.86} />
        </mesh>
        <mesh castShadow position={[0, -size[1] / 2 + 0.18, size[2] / 2 + 0.04]}>
          <boxGeometry args={[size[0] + 0.08, 0.36, 0.12]} />
          <meshStandardMaterial color="#656c69" roughness={0.82} />
        </mesh>
        <mesh castShadow position={[0, 0.1, size[2] / 2 + 0.045]}>
          <boxGeometry args={[1.45, 2.05, 0.12]} />
          <TexturedSurfaceMaterial color={accent} variance={0.06} repeat={[2, 3]} roughness={0.68} />
        </mesh>
        <mesh position={[0, 0.45, size[2] / 2 + 0.11]}>
          <boxGeometry args={[0.6, 0.68, 0.04]} />
          <meshPhysicalMaterial color="#89aeb4" transmission={0.22} metalness={0.12} roughness={0.22} />
        </mesh>
        <mesh castShadow position={[0, 1.2, size[2] / 2 + 0.42]}>
          <boxGeometry args={[2.4, 0.12, 0.85]} />
          <meshStandardMaterial color="#4e5859" metalness={0.45} roughness={0.42} />
        </mesh>
        <Roof kind={roof} size={size} color={accent} />
        <WindowRow width={size[0]} depth={size[2]} y={0.55} />
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
  // Calm overcast research campus: low chroma sky/fog, cool sun, muted ground.
  if (sceneId === 'campus') {
    return { sky: '#7f9098', fog: '#8a98a0', hemi: '#c8d0d2', ground: '#3d4542', sun: '#e4e0d6' };
  }
  if (sceneId === 'plaza') {
    return { sky: '#82929a', fog: '#8d9aa1', hemi: '#cbd2d4', ground: '#424a47', sun: '#e2ddd4' };
  }
  if (sceneId === 'compute') {
    return { sky: '#8a9499', fog: '#8a9499', hemi: '#c5cccf', ground: '#353b3e', sun: '#d7dde0' };
  }
  if (sceneId === 'paper') {
    return { sky: '#8e8a84', fog: '#8e8a84', hemi: '#d0cbc4', ground: '#433e39', sun: '#e6e0d4' };
  }
  return { sky: '#84918e', fog: '#84918e', hemi: '#c9d0cd', ground: '#3d4542', sun: '#e0ddd4' };
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
