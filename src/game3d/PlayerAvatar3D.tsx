import { Html } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import { Ecctrl, type EcctrlHandle } from 'ecctrl';
import {
  forwardRef,
  useEffect,
  useMemo,
  useRef,
  type MutableRefObject,
  type RefObject,
} from 'react';
import * as THREE from 'three';
import type { AvatarPresetId } from '../game/types';
import { avatarPresetById } from '../game/worldContent';
import type {
  Game3DSceneDefinition,
  MovementState3D,
  NavigationCommand3D,
  PlayerWorldState3D,
  RemotePlayer3D,
  Vec3,
} from './types3d';

export function PlayerController3D({
  controllerRef,
  movementRef,
  presetId,
  initialPosition,
  scene,
  enabled,
  reducedMotion,
  onPosition,
  onMovementChange,
  navigationCommand,
  onNavigationComplete,
}: {
  controllerRef: RefObject<EcctrlHandle | null>;
  movementRef: MutableRefObject<MovementState3D>;
  presetId: AvatarPresetId;
  initialPosition: Vec3;
  scene: Game3DSceneDefinition;
  enabled: boolean;
  reducedMotion: boolean;
  onPosition: (state: PlayerWorldState3D) => void;
  onMovementChange: (moving: boolean) => void;
  navigationCommand: NavigationCommand3D | null;
  onNavigationComplete: (command: NavigationCommand3D) => void;
}) {
  const lastReportRef = useRef(0);
  const lastMovingRef = useRef(false);
  const spawn = useMemo(() => new THREE.Vector3(...scene.spawn), [scene.spawn]);
  const targetDirectionRef = useRef(new THREE.Vector3());
  const cameraForwardRef = useRef(new THREE.Vector3());
  const routeRef = useRef({
    commandId: -1,
    waypointIndex: 0,
    completedId: -1,
    lastDistance: Number.POSITIVE_INFINITY,
  });

  useEffect(() => {
    if (!navigationCommand) return;
    routeRef.current.commandId = navigationCommand.id;
    routeRef.current.waypointIndex = 0;
    routeRef.current.completedId = -1;
    routeRef.current.lastDistance = Number.POSITIVE_INFINITY;
  }, [navigationCommand]);

  useFrame(({ clock, camera }) => {
    const controller = controllerRef.current;
    if (!controller) return;
    const movement = movementRef.current;
    const manualInput = Math.hypot(movement.joystick.x, movement.joystick.y) > 0.05;
    let commandMoving = false;
    if (enabled && navigationCommand && !manualInput) {
      const route = routeRef.current;
      if (route.commandId !== navigationCommand.id) {
        route.commandId = navigationCommand.id;
        route.waypointIndex = 0;
        route.completedId = -1;
        route.lastDistance = Number.POSITIVE_INFINITY;
      }
      const interactionDistance = navigationCommand.interaction
        ? Math.hypot(
            navigationCommand.interaction.position[0] - controller.currPos.x,
            navigationCommand.interaction.position[2] - controller.currPos.z,
          )
        : Number.POSITIVE_INFINITY;
      const interactionArrival = navigationCommand.interaction
        ? navigationCommand.interaction.kind === 'portal'
          ? 0.78
          : Math.max(1.15, navigationCommand.interaction.radius * 0.66)
        : 0;
      if (interactionDistance <= interactionArrival) {
        route.waypointIndex = navigationCommand.points.length;
      }
      const target = navigationCommand.points[route.waypointIndex];
      if (target) {
        const dx = target[0] - controller.currPos.x;
        const dz = target[2] - controller.currPos.z;
        const distance = Math.hypot(dx, dz);
        const lastPoint = route.waypointIndex === navigationCommand.points.length - 1;
        const arrivalDistance = lastPoint ? navigationCommand.stopDistance : 0.38;
        const passedTarget =
          distance > route.lastDistance + 0.025 &&
          route.lastDistance < Math.max(1.05, arrivalDistance * 2.8);
        if (distance <= arrivalDistance || passedTarget) {
          route.waypointIndex += 1;
          route.lastDistance = Number.POSITIVE_INFINITY;
        } else {
          route.lastDistance = distance;
          targetDirectionRef.current.set(dx, 0, dz).normalize();
          controller.setForwardDir(targetDirectionRef.current);
          controller.setMovement({
            forward: false,
            backward: false,
            leftward: false,
            rightward: false,
            run: distance > 5,
            jump: false,
            joystick: { x: 0, y: 1 },
          });
          commandMoving = true;
        }
      }
      if (
        route.waypointIndex >= navigationCommand.points.length &&
        route.completedId !== navigationCommand.id
      ) {
        route.completedId = navigationCommand.id;
        controller.setMovement(STOPPED_MOVEMENT);
        onNavigationComplete(navigationCommand);
      }
    }
    if (!commandMoving) {
      camera.getWorldDirection(cameraForwardRef.current);
      cameraForwardRef.current.y = 0;
      if (cameraForwardRef.current.lengthSq() > 0.001) {
        controller.setForwardDir(cameraForwardRef.current.normalize());
      }
      controller.setMovement(enabled ? movement : STOPPED_MOVEMENT);
    }

    const body = controller.body;
    if (!body) return;
    if (!navigationCommand && routeRef.current.commandId !== -1) {
      const velocity = body.linvel();
      body.setLinvel({ x: 0, y: velocity.y, z: 0 }, true);
      routeRef.current.commandId = -1;
      routeRef.current.waypointIndex = 0;
      routeRef.current.lastDistance = Number.POSITIVE_INFINITY;
    }
    if (!enabled) {
      const velocity = body.linvel();
      if (Math.abs(velocity.x) > 0.01 || Math.abs(velocity.z) > 0.01) {
        body.setLinvel({ x: 0, y: velocity.y, z: 0 }, true);
      }
    }
    const position = controller.currPos;
    if (
      position.y < -4 ||
      position.x < scene.bounds.minX - 3 ||
      position.x > scene.bounds.maxX + 3 ||
      position.z < scene.bounds.minZ - 3 ||
      position.z > scene.bounds.maxZ + 3
    ) {
      body.setTranslation(spawn, true);
      body.setLinvel({ x: 0, y: 0, z: 0 }, true);
      body.setAngvel({ x: 0, y: 0, z: 0 }, true);
      return;
    }

    const moving = enabled && controller.isMoving && controller.moveSpeed > 0.08;
    if (moving !== lastMovingRef.current) {
      lastMovingRef.current = moving;
      onMovementChange(moving);
    }
    if (clock.elapsedTime - lastReportRef.current < 0.1) return;
    lastReportRef.current = clock.elapsedTime;
    onPosition({
      x: position.x,
      y: position.y,
      z: position.z,
      moving,
      grounded: controller.isOnGround,
      direction: controller.currLinVel.x < -0.03 ? 'left' : 'right',
    });
  });

  return (
    <Ecctrl
      ref={controllerRef}
      position={initialPosition}
      enable
      useCustomForward
      canSleep={false}
      colliders={false}
      enabledRotations={[false, true, false]}
      maxWalkVel={3.1}
      maxRunVel={5.1}
      accDeltaTime={0.18}
      decDeltaTime={0.12}
      jumpVel={4.2}
      capsuleHalfHeight={0.52}
      capsuleRadius={0.3}
      floatHeight={0.06}
      rayHitForgiveness={0.16}
      springK={92}
      dampingC={9}
      autoBalanceSpringK={0.55}
      autoBalanceDampingC={0.04}
      friction={-0.35}
      linearDamping={0.1}
      angularDamping={0.3}
      userData={{ kind: 'local-player' }}
    >
      <AvatarFigure3D
        presetId={presetId}
        controllerRef={controllerRef}
        reducedMotion={reducedMotion}
      />
    </Ecctrl>
  );
}

const STOPPED_MOVEMENT: MovementState3D = {
  forward: false,
  backward: false,
  leftward: false,
  rightward: false,
  run: false,
  jump: false,
  joystick: { x: 0, y: 0 },
};

export function RemoteAvatar3D({ player }: { player: RemotePlayer3D }) {
  const groupRef = useRef<THREE.Group | null>(null);
  const target = useMemo(() => new THREE.Vector3(), []);
  const movingRef = useRef(false);

  useFrame((_, delta) => {
    const group = groupRef.current;
    if (!group) return;
    target.set(...player.position);
    const distance = group.position.distanceTo(target);
    movingRef.current = distance > 0.025;
    group.position.lerp(target, 1 - Math.exp(-delta * 10));
    group.rotation.y = player.direction === 'left' ? -Math.PI / 2 : Math.PI / 2;
  });

  return (
    <group ref={groupRef} position={player.position}>
      <AvatarFigure3D presetId={player.avatarPreset} movingRef={movingRef} reducedMotion={false} />
      <Html center position={[0, 1.85, 0]} distanceFactor={11} zIndexRange={[5, 0]}>
        <span className="game3d-nameplate">{player.displayName}</span>
      </Html>
    </group>
  );
}

function AvatarFigure3D({
  presetId,
  controllerRef,
  movingRef,
  reducedMotion,
}: {
  presetId: AvatarPresetId;
  controllerRef?: RefObject<EcctrlHandle | null>;
  movingRef?: MutableRefObject<boolean>;
  reducedMotion: boolean;
}) {
  const rootRef = useRef<THREE.Group | null>(null);
  const leftArmRef = useRef<THREE.Group | null>(null);
  const rightArmRef = useRef<THREE.Group | null>(null);
  const leftLegRef = useRef<THREE.Group | null>(null);
  const rightLegRef = useRef<THREE.Group | null>(null);
  const preset = avatarPresetById[presetId];

  useFrame(({ clock }) => {
    const root = rootRef.current;
    if (!root) return;
    const moving = controllerRef?.current
      ? controllerRef.current.isMoving && controllerRef.current.moveSpeed > 0.08
      : Boolean(movingRef?.current);
    const time = clock.elapsedTime;
    const stride = moving && !reducedMotion ? Math.sin(time * 10) * 0.62 : 0;
    const breath = reducedMotion ? 0 : Math.sin(time * 2.1) * 0.012;
    root.position.y = -0.9 + breath + (moving && !reducedMotion ? Math.abs(Math.sin(time * 10)) * 0.035 : 0);
    if (leftArmRef.current) leftArmRef.current.rotation.x = stride;
    if (rightArmRef.current) rightArmRef.current.rotation.x = -stride;
    if (leftLegRef.current) leftLegRef.current.rotation.x = -stride * 0.72;
    if (rightLegRef.current) rightLegRef.current.rotation.x = stride * 0.72;
  });

  // Adult low-poly: muted charcoal outfit, role color only on collar + sleeve bands.
  const cloth = '#3a444a'
  const skin = '#d2b49a'
  const hair = '#2a3136'
  const accent = preset.cssColor
  return (
    <group ref={rootRef} position={[0, -1.02, 0]} scale={1}>
      <mesh castShadow position={[0, 1.62, 0]}>
        <boxGeometry args={[0.4, 0.4, 0.4]} />
        <meshStandardMaterial color={skin} roughness={0.95} flatShading />
      </mesh>
      <mesh castShadow position={[0, 1.86, 0]}>
        <boxGeometry args={[0.44, 0.16, 0.44]} />
        <meshStandardMaterial color={hair} roughness={0.95} flatShading />
      </mesh>
      {/* torso */}
      <mesh castShadow position={[0, 1.12, 0]}>
        <boxGeometry args={[0.55, 0.85, 0.35]} />
        <meshStandardMaterial color={cloth} roughness={0.92} flatShading />
      </mesh>
      {/* collar accent */}
      <mesh castShadow position={[0, 1.38, 0.06]}>
        <boxGeometry args={[0.34, 0.08, 0.16]} />
        <meshStandardMaterial color={accent} roughness={0.7} />
      </mesh>
      {/* ID badge strip */}
      <mesh castShadow position={[0.12, 1.18, 0.22]}>
        <boxGeometry args={[0.1, 0.14, 0.02]} />
        <meshStandardMaterial color={accent} roughness={0.55} metalness={0.15} />
      </mesh>

      <Limb ref={leftArmRef} position={[-0.34, 1.32, 0]} color={cloth} accent={accent} />
      <Limb ref={rightArmRef} position={[0.34, 1.32, 0]} color={cloth} accent={accent} />
      <Limb ref={leftLegRef} position={[-0.12, 0.68, 0]} color="#2f383e" leg />
      <Limb ref={rightLegRef} position={[0.12, 0.68, 0]} color="#2f383e" leg />
    </group>
  );
}

const Limb = forwardRef<
  THREE.Group,
  { position: Vec3; color: string; leg?: boolean; accent?: string }
>(function Limb({ position, color, leg = false, accent }, ref) {
    return (
      <group ref={ref} position={position}>
        <mesh castShadow position={[0, leg ? -0.28 : -0.22, 0]}>
          <boxGeometry args={[leg ? 0.18 : 0.16, leg ? 0.55 : 0.48, leg ? 0.18 : 0.16]} />
          <meshStandardMaterial color={color} roughness={0.92} flatShading />
        </mesh>
        {!leg && accent && (
          <mesh castShadow position={[0, -0.38, 0]}>
            <boxGeometry args={[0.16, 0.08, 0.16]} />
            <meshStandardMaterial color={accent} roughness={0.65} />
          </mesh>
        )}
        {leg && (
          <mesh castShadow position={[0, -0.54, 0.06]}>
            <boxGeometry args={[0.18, 0.1, 0.3]} />
            <meshStandardMaterial color="#1f2a30" roughness={0.92} />
          </mesh>
        )}
      </group>
    );
  },
);


