import * as THREE from 'three';

/**
 * Shared camera-relative basis for WASD.
 * CameraRig publishes from the *actual* camera look (horizontal).
 * PlayerController consumes this so movement matches what you see.
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

/** Update basis from yaw (radians). Forward = camera look on XZ. */
export function setLookFromYawPitch(state: LookDirState, yaw: number, pitch: number) {
  state.yaw = yaw;
  state.pitch = pitch;
  // Camera sits at (+sin(yaw), +, +cos(yaw)) relative to player and looks toward
  // the player / world ahead → world-forward is (-sin(yaw), 0, -cos(yaw)).
  state.forward.set(-Math.sin(yaw), 0, -Math.cos(yaw));
  if (state.forward.lengthSq() < 1e-8) state.forward.set(0, 0, -1);
  else state.forward.normalize();
  // right = forward × worldUp?  For move: right should be (+cos, 0, -sin) for yaw=0 → +X
  // cross(forward, up) = (cos, 0, -sin) when forward=(-sin,0,-cos)
  state.right.set(Math.cos(yaw), 0, -Math.sin(yaw));
  if (state.right.lengthSq() < 1e-8) state.right.set(1, 0, 0);
  else state.right.normalize();
}

/** Prefer live camera horizontal forward (most accurate for third-person). */
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
  // right = forward × up (Y)
  state.right.set(state.forward.z, 0, -state.forward.x).normalize();
  // Wait: forward × (0,1,0) = (fz, 0, -fx)? 
  // (fx,0,fz)×(0,1,0) = (-fz, 0, fx) 
  state.right.set(-state.forward.z, 0, state.forward.x);
  if (state.right.lengthSq() < 1e-8) state.right.set(1, 0, 0);
  else state.right.normalize();
}
