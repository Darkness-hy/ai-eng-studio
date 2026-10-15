import type { ZoneId } from '../lib/researchGame';

export type GameSceneId = 'campus' | 'academy' | 'research' | 'compute' | 'paper' | 'plaza';

export type AvatarPresetId =
  | 'theory-blue'
  | 'systems-teal'
  | 'vision-green'
  | 'robotics-gold'
  | 'agents-red'
  | 'safety-violet';

export type AvatarPreset = {
  id: AvatarPresetId;
  name: string;
  field: string;
  asset: string;
  color: number;
  cssColor: string;
};

export type WorldPoint = { x: number; y: number };

export type AgentId =
  | 'research-mentor'
  | 'campus-guide'
  | 'methods-mentor'
  | 'experiment-engineer'
  | 'reviewer'
  | 'community-host';

export type AgentDefinition = {
  id: AgentId;
  name: string;
  role: string;
  greeting: string;
  focus: string;
  asset: string;
  workspaceZone?: ZoneId;
};

export type GameInteraction =
  | {
      id: string;
      kind: 'portal';
      label: string;
      position: WorldPoint;
      targetScene: GameSceneId;
      spawn?: WorldPoint;
    }
  | {
      id: string;
      kind: 'agent';
      label: string;
      position: WorldPoint;
      agentId: AgentId;
    }
  | {
      id: string;
      kind: 'workspace';
      label: string;
      position: WorldPoint;
      workspaceZone: ZoneId;
    };

export type GameSceneDefinition = {
  id: GameSceneId;
  name: string;
  subtitle: string;
  background: string;
  spawn: WorldPoint;
  tint: number;
  interactions: GameInteraction[];
};

export type PlayerGameProfile = {
  avatarPreset: AvatarPresetId;
  currentScene: GameSceneId;
  lastPosition: WorldPoint;
  tutorialStep: number;
};

export type RemotePlayerState = {
  clientId: string;
  userId: string;
  displayName: string;
  avatarPreset: AvatarPresetId;
  sceneId: GameSceneId;
  x: number;
  y: number;
  direction: 'left' | 'right';
  updatedAt: number;
};

export type LocalPositionEvent = {
  sceneId: GameSceneId;
  x: number;
  y: number;
  direction: 'left' | 'right';
};

export type GameRuntimeCallbacks = {
  onInteraction: (interaction: GameInteraction) => void;
  onSceneChange: (sceneId: GameSceneId, position: WorldPoint) => void;
  onPosition: (position: LocalPositionEvent) => void;
  onNearbyChange: (label: string | null) => void;
  onMovementChange: (moving: boolean) => void;
};

export interface ResearchGameRuntime {
  destroy(): void;
  setScene(sceneId: GameSceneId, position?: WorldPoint): void;
  setInputEnabled(enabled: boolean): void;
  setRemotePlayers(players: RemotePlayerState[]): void;
  waveRemote(clientId: string): void;
  wave(): void;
}
