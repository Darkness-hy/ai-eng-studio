import { OrthographicCamera, PerspectiveCamera } from '@react-three/drei';
import { useFrame, useThree } from '@react-three/fiber';
import { useRapier } from '@react-three/rapier';
import type { EcctrlHandle } from 'ecctrl';
import { useEffect, useMemo, useRef, type MutableRefObject, type RefObject } from 'react';
import * as THREE from 'three';
import type { LookDirState } from './lookDir';
import { setLookFromYawPitch } from './lookDir';
import type { CameraMode3D, Game3DSceneDefinition, Vec3 } from './types3d';

export function CameraRig3D({
  controllerRef,
  mode,
  scene,
  focusPosition,
  reducedMotion,
  lookDirRef,
}: {
  controllerRef: RefObject<EcctrlHandle | null>;
  mode: CameraMode3D;
  scene: Game3DSceneDefinition;
  focusPosition: Vec3 | null;
  reducedMotion: boolean;
  lookDirRef?: MutableRefObject<LookDirState>;
}) {
  const followRef = useRef<THREE.PerspectiveCamera | null>(null);
  const mapRef = useRef<THREE.OrthographicCamera | null>(null);
  const yawRef = useRef(0);
  const pitchRef = useRef(0.46);
  const distanceRef = useRef(8.4);
  const draggingRef = useRef(false);
  const pointerRef = useRef({ x: 0, y: 0 });
  const currentTarget = useMemo(() => new THREE.Vector3(...scene.spawn), [scene.spawn]);
  const cameraAnchor = useMemo(() => new THREE.Vector3(), []);
  const desiredTarget = useMemo(() => new THREE.Vector3(), []);
  const desiredCamera = useMemo(() => new THREE.Vector3(), []);
  const cameraDirection = useMemo(() => new THREE.Vector3(), []);
  const { gl, size } = useThree();
  const { world, rapier } = useRapier();

  useEffect(() => {
    const canvas = gl.domElement;
    const onPointerDown = (event: PointerEvent) => {
      if ((mode !== 'follow' && mode !== 'first') || (event.button !== 1 && event.button !== 2 && !(mode === 'first' && event.button === 0))) return;
      event.preventDefault();
      draggingRef.current = true;
      pointerRef.current = { x: event.clientX, y: event.clientY };
      canvas.setPointerCapture(event.pointerId);
    };
    const onPointerMove = (event: PointerEvent) => {
      if (!draggingRef.current || (mode !== 'follow' && mode !== 'first')) return;
      const dx = event.clientX - pointerRef.current.x;
      const dy = event.clientY - pointerRef.current.y;
      pointerRef.current = { x: event.clientX, y: event.clientY };
      yawRef.current -= dx * 0.006;
      pitchRef.current = THREE.MathUtils.clamp(pitchRef.current + dy * 0.0035, 0.12, 0.85);
      if (lookDirRef) setLookFromYawPitch(lookDirRef.current, yawRef.current, pitchRef.current);
    };
    const onPointerUp = (event: PointerEvent) => {
      draggingRef.current = false;
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    };
    const onWheel = (event: WheelEvent) => {
      if (mode === 'first') return;
      if (mode !== 'follow') return;
      distanceRef.current = THREE.MathUtils.clamp(distanceRef.current + event.deltaY * 0.006, 4.8, 10.5);
    };
    const onContextMenu = (event: MouseEvent) => event.preventDefault();
    const onClick = () => {
      if (mode === 'first' && document.pointerLockElement !== canvas) {
        canvas.requestPointerLock?.();
      }
    };
    const onPointerLockMove = (event: MouseEvent) => {
      if (mode !== 'first' || document.pointerLockElement !== canvas) return;
      yawRef.current -= event.movementX * 0.0022;
      pitchRef.current = THREE.MathUtils.clamp(
        pitchRef.current + event.movementY * 0.0016,
        0.05,
        0.85,
      );
      if (lookDirRef) setLookFromYawPitch(lookDirRef.current, yawRef.current, pitchRef.current);
    };
    const onPointerLockChange = () => {
      if (document.pointerLockElement !== canvas) {
        draggingRef.current = false;
      }
    };
    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerup', onPointerUp);
    canvas.addEventListener('pointercancel', onPointerUp);
    canvas.addEventListener('wheel', onWheel, { passive: true });
    canvas.addEventListener('contextmenu', onContextMenu);
    canvas.addEventListener('click', onClick);
    document.addEventListener('mousemove', onPointerLockMove);
    document.addEventListener('pointerlockchange', onPointerLockChange);
    return () => {
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', onPointerUp);
      canvas.removeEventListener('pointercancel', onPointerUp);
      canvas.removeEventListener('wheel', onWheel);
      canvas.removeEventListener('contextmenu', onContextMenu);
      canvas.removeEventListener('click', onClick);
      document.removeEventListener('mousemove', onPointerLockMove);
      document.removeEventListener('pointerlockchange', onPointerLockChange);
      if (document.pointerLockElement === canvas) document.exitPointerLock?.();
    };
  }, [gl, mode]);

  useFrame((_, delta) => {
    // Publish look basis before / regardless of camera lerp so movement can read it.
    if (lookDirRef) setLookFromYawPitch(lookDirRef.current, yawRef.current, pitchRef.current);

    const controller = controllerRef.current;
    const followCamera = followRef.current;
    if (!followCamera) return;
    const player = controller?.currPos;
    if (player) cameraAnchor.set(player.x, player.y + 0.72, player.z);
    else cameraAnchor.set(scene.spawn[0], scene.spawn[1] + 0.72, scene.spawn[2]);
    desiredTarget.copy(cameraAnchor);

    if ((mode === 'dialogue' || mode === 'workstation') && focusPosition) {
      const focus = new THREE.Vector3(...focusPosition).add(new THREE.Vector3(0, 1.45, 0));
      desiredTarget.lerp(focus, mode === 'dialogue' ? 0.72 : 0.48);
      const dist = mode === 'dialogue' ? 2.85 : 3.6;
      desiredCamera.set(
        desiredTarget.x + dist * 0.72,
        desiredTarget.y + (mode === 'dialogue' ? 1.15 : 1.8),
        desiredTarget.z + dist * 0.72,
      );
    } else if (mode === 'first') {
      // Eye height first-person; look along yaw/pitch.
      const eyeY = 1.55;
      desiredCamera.set(cameraAnchor.x, cameraAnchor.y + eyeY - 0.72, cameraAnchor.z);
      const lookDist = 6;
      const pitch = pitchRef.current - 0.2;
      desiredTarget.set(
        cameraAnchor.x - Math.sin(yawRef.current) * Math.cos(pitch) * lookDist,
        cameraAnchor.y + eyeY - 0.72 + Math.sin(pitch) * lookDist * 0.35,
        cameraAnchor.z - Math.cos(yawRef.current) * Math.cos(pitch) * lookDist,
      );
    } else {
      const horizontal = Math.cos(pitchRef.current) * distanceRef.current;
      desiredCamera.set(
        cameraAnchor.x + Math.sin(yawRef.current) * horizontal,
        cameraAnchor.y + Math.sin(pitchRef.current) * distanceRef.current + 1.35,
        cameraAnchor.z + Math.cos(yawRef.current) * horizontal,
      );
      desiredTarget.set(
        cameraAnchor.x - Math.sin(yawRef.current) * 0.7,
        cameraAnchor.y,
        cameraAnchor.z - Math.cos(yawRef.current) * 0.7,
      );
      if (controller) {
        cameraDirection.subVectors(desiredCamera, cameraAnchor);
        const maxDistance = cameraDirection.length();
        cameraDirection.normalize();
        const ray = new rapier.Ray(cameraAnchor, cameraDirection);
        const hit = world.castRay(
          ray,
          maxDistance,
          true,
          undefined,
          undefined,
          undefined,
          controller.body,
        );
        if (hit) {
          const safeDistance = Math.max(1.6, hit.timeOfImpact - 0.32);
          desiredCamera.copy(cameraAnchor).addScaledVector(cameraDirection, safeDistance);
        }
      }
    }

    const smoothing = reducedMotion ? 1 : 1 - Math.exp(-delta * (mode === 'dialogue' ? 5.5 : 8.5));
    currentTarget.lerp(desiredTarget, smoothing);
    followCamera.position.lerp(desiredCamera, smoothing);
    followCamera.lookAt(currentTarget);

    const mapCamera = mapRef.current;
    if (mapCamera) {
      const centerX = (scene.bounds.minX + scene.bounds.maxX) / 2;
      const centerZ = (scene.bounds.minZ + scene.bounds.maxZ) / 2;
      const spanX = scene.bounds.maxX - scene.bounds.minX;
      const spanZ = scene.bounds.maxZ - scene.bounds.minZ;
      const maxSpan = Math.max(spanX, spanZ);
      const campusScale = scene.id === 'campus';
      const compactMap = campusScale && size.width < 600;
      mapCamera.up.set(0, 1, 0);
      mapCamera.position.set(
        centerX + (campusScale ? spanX * 0.34 : 15),
        campusScale ? maxSpan * 0.92 : 24,
        centerZ + (campusScale ? spanZ * 0.52 : 18),
      );
      mapCamera.lookAt(centerX, 0, centerZ);
      mapCamera.zoom = Math.min(
        size.width /
          (spanX + (compactMap ? 13 : campusScale ? 24 : 9)),
        size.height /
          (spanZ + (compactMap ? 13 : campusScale ? 18 : 8)),
      );
      mapCamera.updateProjectionMatrix();
    }
  }, -1);

  return (
    <>
      <PerspectiveCamera
        ref={followRef}
        makeDefault={mode !== 'map'}
        fov={48}
        near={0.08}
        far={220}
        position={[0, 6, 12]}
      />
      <OrthographicCamera
        ref={mapRef}
        makeDefault={mode === 'map'}
        near={0.1}
        far={260}
        position={[0, 45, 0.001]}
      />
    </>
  );
}
