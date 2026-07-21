import * as THREE from 'three';

/**
 * Shared look basis for movement (WASD relative to view).
 * CameraRig writes; PlayerController reads — avoids useFrame order bugs
 * when first-person camera is updated after the character controller.
 */
export type LookDirState = {
  /** Horizontal unit forward (y = 0). */
  forward: THREE.Vector3;
  yaw: number;
  pitch: number;
};

export function createLookDirState(): LookDirState {
  return {
    forward: new THREE.Vector3(0, 0, -1),
    yaw: 0,
    pitch: 0.46,
  };
}

export function setLookFromYawPitch(state: LookDirState, yaw: number, pitch: number) {
  state.yaw = yaw;
  state.pitch = pitch;
  // Match CameraRig convention: look toward -sin(yaw), -cos(yaw) on XZ.
  state.forward.set(-Math.sin(yaw), 0, -Math.cos(yaw));
  if (state.forward.lengthSq() < 1e-8) state.forward.set(0, 0, -1);
  else state.forward.normalize();
}
