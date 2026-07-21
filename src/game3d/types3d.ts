import type { ZoneId } from '../lib/researchGame';
import type { AgentId, AvatarPresetId, GameSceneId } from '../game/types';

export type Vec3 = [number, number, number];

export type CameraMode3D = 'follow' | 'first' | 'map' | 'dialogue' | 'workstation';

export type MovementState3D = {
  forward: boolean;
  backward: boolean;
  leftward: boolean;
  rightward: boolean;
  run: boolean;
  jump: boolean;
  joystick: { x: number; y: number };
};

export type Game3DInteraction =
  | {
      id: string;
      kind: 'portal';
      label: string;
      position: Vec3;
      targetScene: GameSceneId;
      radius: number;
    }
  | {
      id: string;
      kind: 'agent';
      label: string;
      position: Vec3;
      agentId: AgentId;
      radius: number;
    }
  | {
      id: string;
      kind: 'workspace';
      label: string;
      position: Vec3;
      workspaceZone: ZoneId;
      radius: number;
    }
  | {
      id: string;
      kind: 'landmark';
      label: string;
      title: string;
      note: string;
      position: Vec3;
      radius: number;
      /** When set, inspecting feeds the Case Detective v0 store. */
      casePoiId?: string;
    };

export type LandmarkInteraction3D = Extract<Game3DInteraction, { kind: 'landmark' }>;

export type NavigationObstacle3D = {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
};

export type NavigationConfig3D = {
  cellSize: number;
  shape: 'rectangle' | 'ellipse';
  padding: number;
  obstacles: NavigationObstacle3D[];
  paths?: NavigationPath3D[];
  zones?: NavigationZone3D[];
};

export type NavigationPath3D = {
  id: string;
  points: Array<[number, number]>;
  width: number;
};

export type NavigationZone3D = {
  x: number;
  z: number;
  radiusX: number;
  radiusZ: number;
};

export type Game3DSceneDefinition = {
  id: GameSceneId;
  name: string;
  subtitle: string;
  spawn: Vec3;
  bounds: {
    minX: number;
    maxX: number;
    minZ: number;
    maxZ: number;
  };
  navigation: NavigationConfig3D;
  interactions: Game3DInteraction[];
};

export type NavigationCommand3D = {
  id: number;
  points: Vec3[];
  destination: Vec3;
  interaction: Game3DInteraction | null;
  stopDistance: number;
};

export type PlayerWorldState3D = {
  x: number;
  y: number;
  z: number;
  moving: boolean;
  grounded: boolean;
  direction: 'left' | 'right';
};

export type RemotePlayer3D = {
  clientId: string;
  displayName: string;
  avatarPreset: AvatarPresetId;
  position: Vec3;
  direction: 'left' | 'right';
  updatedAt: number;
};

export type ThreeGameDiagnostics = {
  renderer: 'webgl';
  sceneId: Game3DSceneDefinition['id'];
  cameraMode: CameraMode3D;
  camera: { x: number; y: number; z: number };
  player: { x: number; y: number; z: number; moving: boolean };
  grounded: boolean;
  footY: number;
  navigationTarget: { x: number; z: number } | null;
  playerScreen: { x: number; y: number; visible: boolean };
  nearbyInteractionId: string | null;
  drawCalls: number;
  triangles: number;
  fps: number;
};

declare global {
  interface Window {
    __THREE_GAME_DIAGNOSTICS__?: ThreeGameDiagnostics;
  }
}
