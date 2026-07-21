import { Canvas, useFrame } from '@react-three/fiber';
import { Physics } from '@react-three/rapier';
import type { EcctrlHandle } from 'ecctrl';
import { Suspense, useRef, type MutableRefObject } from 'react';
import * as THREE from 'three';
import type { AvatarPresetId } from '../game/types';
import { CameraRig3D } from './CameraRig3D';
import { PlayerController3D, RemoteAvatar3D } from './PlayerAvatar3D';
import { ResearchWorld3D } from './ResearchWorld3D';
import type {
  CameraMode3D,
  Game3DInteraction,
  Game3DSceneDefinition,
  MovementState3D,
  NavigationCommand3D,
  PlayerWorldState3D,
  RemotePlayer3D,
  Vec3,
} from './types3d';

export function ResearchGame3DCanvas({
  scene,
  avatarPreset,
  initialPosition,
  movementRef,
  cameraMode,
  inputEnabled,
  reducedMotion,
  remotePlayers,
  focusPosition,
  onPosition,
  onMovementChange,
  onNearbyChange,
  navigationCommand,
  onSelectGround,
  onSelectInteraction,
  onActivateInteraction,
  onNavigationComplete,
  onReady,
}: {
  scene: Game3DSceneDefinition;
  avatarPreset: AvatarPresetId;
  initialPosition: Vec3;
  movementRef: MutableRefObject<MovementState3D>;
  cameraMode: CameraMode3D;
  inputEnabled: boolean;
  reducedMotion: boolean;
  remotePlayers: RemotePlayer3D[];
  focusPosition: Vec3 | null;
  onPosition: (state: PlayerWorldState3D) => void;
  onMovementChange: (moving: boolean) => void;
  onNearbyChange: (interaction: Game3DInteraction | null) => void;
  navigationCommand: NavigationCommand3D | null;
  onSelectGround: (position: Vec3) => void;
  onSelectInteraction: (interaction: Game3DInteraction) => void;
  onActivateInteraction: (interaction: Game3DInteraction) => void;
  onNavigationComplete: (command: NavigationCommand3D) => void;
  onReady: () => void;
}) {
  const controllerRef = useRef<EcctrlHandle | null>(null);
  const isMobile =
    typeof window !== 'undefined' && window.matchMedia('(max-width: 767px)').matches;
  const dpr = isMobile ? 1.25 : 1.5;

  return (
    <Canvas
      shadows={!isMobile && cameraMode !== 'map'}
      dpr={[1, dpr]}
      gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
      className="game3d-canvas"
      onCreated={({ gl }) => {
        gl.outputColorSpace = THREE.SRGBColorSpace;
        gl.toneMapping = THREE.ACESFilmicToneMapping;
        gl.toneMappingExposure = 1.05;
      }}
    >
      <Suspense fallback={null}>
        <Physics gravity={[0, -18, 0]} timeStep={1 / 60}>
          <ResearchWorld3D
            scene={scene}
            cameraMode={cameraMode}
            reducedMotion={reducedMotion}
            onSelectGround={onSelectGround}
            onSelectInteraction={onSelectInteraction}
          />
          <PlayerController3D
            controllerRef={controllerRef}
            movementRef={movementRef}
            presetId={avatarPreset}
            initialPosition={initialPosition}
            scene={scene}
            enabled={inputEnabled && (cameraMode === 'follow' || cameraMode === 'first')}
            hideAvatar={cameraMode === 'first'}
            reducedMotion={reducedMotion}
            onPosition={onPosition}
            onMovementChange={onMovementChange}
            navigationCommand={navigationCommand}
            onNavigationComplete={onNavigationComplete}
          />
          {remotePlayers.map((player) => (
            <RemoteAvatar3D key={player.clientId} player={player} />
          ))}
          <CameraRig3D
            controllerRef={controllerRef}
            mode={cameraMode}
            scene={scene}
            focusPosition={focusPosition}
            reducedMotion={reducedMotion}
          />
          <WorldLogic3D
            controllerRef={controllerRef}
            scene={scene}
            cameraMode={cameraMode}
            onNearbyChange={onNearbyChange}
            onInteraction={onActivateInteraction}
          />
          <NavigationTargetMarker command={navigationCommand} visible={cameraMode === 'follow'} />
          <PlayerMapMarker controllerRef={controllerRef} visible={cameraMode === 'map'} />
          <Diagnostics3D
            controllerRef={controllerRef}
            scene={scene}
            cameraMode={cameraMode}
            navigationCommand={navigationCommand}
            onReady={onReady}
          />
        </Physics>
      </Suspense>
    </Canvas>
  );
}

function WorldLogic3D({
  controllerRef,
  scene,
  cameraMode,
  onNearbyChange,
  onInteraction,
}: {
  controllerRef: React.RefObject<EcctrlHandle | null>;
  scene: Game3DSceneDefinition;
  cameraMode: CameraMode3D;
  onNearbyChange: (interaction: Game3DInteraction | null) => void;
  onInteraction: (interaction: Game3DInteraction) => void;
}) {
  const lastCheckRef = useRef(0);
  const lastNearbyIdRef = useRef<string | null>(null);
  const enteredPortalRef = useRef<string | null>(null);

  useFrame(({ clock }) => {
    const controller = controllerRef.current;
    if (!controller || clock.elapsedTime - lastCheckRef.current < 0.08) return;
    lastCheckRef.current = clock.elapsedTime;
    const position = controller.currPos;
    let nearest: Game3DInteraction | null = null;
    let nearestDistance = Number.POSITIVE_INFINITY;
    for (const interaction of scene.interactions) {
      const dx = position.x - interaction.position[0];
      const dz = position.z - interaction.position[2];
      const distance = Math.hypot(dx, dz);
      if (distance <= interaction.radius && distance < nearestDistance) {
        nearest = interaction;
        nearestDistance = distance;
      }
      if (
        cameraMode === 'follow' &&
        interaction.kind === 'portal' &&
        distance < 0.72 &&
        enteredPortalRef.current !== interaction.id
      ) {
        enteredPortalRef.current = interaction.id;
        onInteraction(interaction);
      }
      if (
        interaction.kind === 'portal' &&
        enteredPortalRef.current === interaction.id &&
        distance > 1.9
      ) {
        enteredPortalRef.current = null;
      }
    }
    const nextId = nearest?.id ?? null;
    if (nextId !== lastNearbyIdRef.current) {
      lastNearbyIdRef.current = nextId;
      onNearbyChange(nearest);
    }
  });
  return null;
}

function PlayerMapMarker({
  controllerRef,
  visible,
}: {
  controllerRef: React.RefObject<EcctrlHandle | null>;
  visible: boolean;
}) {
  const markerRef = useRef<THREE.Group | null>(null);
  useFrame(() => {
    const marker = markerRef.current;
    const controller = controllerRef.current;
    if (!marker || !controller) return;
    marker.position.set(controller.currPos.x, 0.35, controller.currPos.z);
  });
  return (
    <group ref={markerRef} visible={visible}>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.55, 0.82, 24]} />
        <meshBasicMaterial color="#ffffff" depthTest={false} />
      </mesh>
      <mesh position={[0, 0.12, 0]}>
        <cylinderGeometry args={[0.26, 0.26, 0.25, 12]} />
        <meshBasicMaterial color="#df3e35" depthTest={false} />
      </mesh>
    </group>
  );
}

function NavigationTargetMarker({
  command,
  visible,
}: {
  command: NavigationCommand3D | null;
  visible: boolean;
}) {
  const ringRef = useRef<THREE.Mesh | null>(null);
  useFrame(({ clock }) => {
    if (!ringRef.current) return;
    const pulse = 1 + Math.sin(clock.elapsedTime * 5) * 0.08;
    ringRef.current.scale.setScalar(pulse);
    ringRef.current.rotation.z = clock.elapsedTime * 0.7;
  });
  if (!command || !visible) return null;
  return (
    <group position={command.destination}>
      <mesh ref={ringRef} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.035, 0]}>
        <ringGeometry args={[0.25, 0.42, 24]} />
        <meshBasicMaterial color="#fff1a6" transparent opacity={0.92} depthWrite={false} />
      </mesh>
      <mesh position={[0, 0.05, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[0.09, 16]} />
        <meshBasicMaterial color="#f3b63d" transparent opacity={0.9} depthWrite={false} />
      </mesh>
    </group>
  );
}

function Diagnostics3D({
  controllerRef,
  scene,
  cameraMode,
  navigationCommand,
  onReady,
}: {
  controllerRef: React.RefObject<EcctrlHandle | null>;
  scene: Game3DSceneDefinition;
  cameraMode: CameraMode3D;
  navigationCommand: NavigationCommand3D | null;
  onReady: () => void;
}) {
  const frameRef = useRef(0);
  const sampleStartRef = useRef(0);
  const fpsRef = useRef(0);
  const readyRef = useRef(false);
  const projectedPlayerRef = useRef(new THREE.Vector3());
  useFrame(({ clock, gl, camera }) => {
    const controller = controllerRef.current;
    frameRef.current += 1;
    if (!readyRef.current && frameRef.current > 2 && controller) {
      readyRef.current = true;
      onReady();
    }
    if (sampleStartRef.current === 0) sampleStartRef.current = clock.elapsedTime;
    const elapsed = clock.elapsedTime - sampleStartRef.current;
    if (elapsed >= 0.5) {
      fpsRef.current = Math.round(frameRef.current / elapsed);
      frameRef.current = 0;
      sampleStartRef.current = clock.elapsedTime;
    }
    if (!controller) return;
    projectedPlayerRef.current
      .set(controller.currPos.x, controller.currPos.y + 0.8, controller.currPos.z)
      .project(camera);
    window.__THREE_GAME_DIAGNOSTICS__ = {
      renderer: 'webgl',
      sceneId: scene.id,
      cameraMode,
      camera: {
        x: round(camera.position.x),
        y: round(camera.position.y),
        z: round(camera.position.z),
      },
      player: {
        x: round(controller.currPos.x),
        y: round(controller.currPos.y),
        z: round(controller.currPos.z),
        moving: controller.isMoving,
      },
      grounded: controller.isOnGround,
      footY: round(controller.currPos.y - 0.803),
      navigationTarget: navigationCommand
        ? { x: round(navigationCommand.destination[0]), z: round(navigationCommand.destination[2]) }
        : null,
      playerScreen: {
        x: round((projectedPlayerRef.current.x + 1) / 2),
        y: round((1 - projectedPlayerRef.current.y) / 2),
        visible:
          Math.abs(projectedPlayerRef.current.x) <= 1 &&
          Math.abs(projectedPlayerRef.current.y) <= 1 &&
          projectedPlayerRef.current.z >= -1 &&
          projectedPlayerRef.current.z <= 1,
      },
      nearbyInteractionId: nearestInteractionId(scene, controller.currPos),
      drawCalls: gl.info.render.calls,
      triangles: gl.info.render.triangles,
      fps: fpsRef.current,
    };
  });
  return null;
}

function nearestInteractionId(
  scene: Game3DSceneDefinition,
  position: THREE.Vector3,
): string | null {
  let nearest: string | null = null;
  let min = Number.POSITIVE_INFINITY;
  for (const interaction of scene.interactions) {
    const distance = Math.hypot(
      position.x - interaction.position[0],
      position.z - interaction.position[2],
    );
    if (distance <= interaction.radius && distance < min) {
      min = distance;
      nearest = interaction.id;
    }
  }
  return nearest;
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}
