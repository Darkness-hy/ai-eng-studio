import * as THREE from 'three';

/**
 * Shared camera-relative basis for WASD (Minecraft-style).
 * Prefer publishing from the live camera after lookAt.
 */
export type LookDirState = {
  forward: THREE.Vector3;
  right: THREE.Vector3;
  yaw: number;
  pitch: number;
};

export function createLookDirState(): LookDirState {
  return {
    forward: new THREE.Vector3(0, 0, -1),
    right: new THREE.Vector3(1, 0, 0),
    yaw: 0,
    pitch: 0.46,
  };
}

function setRightFromForward(state: LookDirState) {
  // right = forward × worldUp  (fx,0,fz)×(0,1,0) = (-fz, 0, fx)
  state.right.set(-state.forward.z, 0, state.forward.x);
  if (state.right.lengthSq() < 1e-8) state.right.set(1, 0, 0);
  else state.right.normalize();
}

/** Update basis from yaw (radians). Used for first-person mouse look. */
export function setLookFromYawPitch(state: LookDirState, yaw: number, pitch: number) {
  state.yaw = yaw;
  state.pitch = pitch;
  // Matches CameraRig first-person look target: (-sin(yaw), 0, -cos(yaw))
  state.forward.set(-Math.sin(yaw), 0, -Math.cos(yaw));
  if (state.forward.lengthSq() < 1e-8) state.forward.set(0, 0, -1);
  else state.forward.normalize();
  setRightFromForward(state);
}

/** Prefer live camera horizontal forward (third-person orbit). */
export function setLookFromCamera(
  state: LookDirState,
  camera: THREE.Camera,
  yawFallback: number,
  pitchFallback: number,
) {
  state.yaw = yawFallback;
  state.pitch = pitchFallback;
  camera.getWorldDirection(state.forward);
  state.forward.y = 0;
  if (state.forward.lengthSq() < 1e-6) {
    setLookFromYawPitch(state, yawFallback, pitchFallback);
    return;
  }
  state.forward.normalize();
  setRightFromForward(state);
}
