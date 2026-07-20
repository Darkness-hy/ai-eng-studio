import { useCallback, useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { useAuth } from '../lib/auth';
import type { ZoneId } from '../lib/researchGame';
import { AgentPanel } from './AgentPanel';
import { AvatarSetup } from './AvatarSetup';
import { CaseIslandHud } from './case/CaseIslandHud';
import { useCase } from './case/caseStore';
import {
  loadCloudGameProfile,
  loadLocalGameProfile,
  saveCloudGameProfile,
  saveLocalGameProfile,
} from './gameProfile';
import { ResearchWorkspaceOverlay } from './ResearchWorkspaceOverlay';
import type {
  AgentId,
  AvatarPresetId,
  GameInteraction,
  GameSceneId,
  LocalPositionEvent,
  PlayerGameProfile,
  RemotePlayerState,
  ResearchGameRuntime,
  WorldPoint,
} from './types';
import { useGamePresence } from './useGamePresence';
import {
  avatarPresetById,
  gameScenes,
  isGameSceneId,
} from './worldContent';

type OpenPanel =
  | { kind: 'agent'; agentId: AgentId }
  | { kind: 'workspace'; zoneId: ZoneId }
  | null;

export function ResearchGameShell() {
  const { sceneId: routeSceneId } = useParams<{ sceneId?: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const caseFile = useCase();
  const force2D = new URLSearchParams(location.search).get('renderer') === '2d';
  const auth = useAuth();
  const requestedScene = sceneFromRoute(routeSceneId);
  const initialSceneRef = useRef(requestedScene);
  const mountRef = useRef<HTMLDivElement | null>(null);
  const runtimeRef = useRef<ResearchGameRuntime | null>(null);
  const latestProfileRef = useRef<PlayerGameProfile | null>(null);
  const currentSceneRef = useRef<GameSceneId>(requestedScene);
  const currentPositionRef = useRef<WorldPoint>(gameScenes[requestedScene].spawn);
  const broadcastPositionRef = useRef<(position: LocalPositionEvent) => void>(() => undefined);
  const remotePlayersRef = useRef<RemotePlayerState[]>([]);
  const gameInputEnabledRef = useRef(true);
  const lastLocalSaveRef = useRef(0);
  const arrivalTimerRef = useRef<number | undefined>(undefined);

  const [profileReady, setProfileReady] = useState(false);
  const [playerProfile, setPlayerProfile] = useState<PlayerGameProfile | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<AvatarPresetId>('theory-blue');
  const [customizingAvatar, setCustomizingAvatar] = useState(false);
  const [currentScene, setCurrentScene] = useState<GameSceneId>(requestedScene);
  const [sceneJoinPosition, setSceneJoinPosition] = useState<WorldPoint>(
    gameScenes[requestedScene].spawn,
  );
  const [openPanel, setOpenPanel] = useState<OpenPanel>(null);
  const [nearbyLabel, setNearbyLabel] = useState<string | null>(null);
  const [runtimeReady, setRuntimeReady] = useState(false);
  const [runtimeError, setRuntimeError] = useState<string | null>(null);
  const [playerMoving, setPlayerMoving] = useState(false);
  const [arrivalScene, setArrivalScene] = useState<GameSceneId | null>(null);

  const userId = auth.profile?.id ?? null;
  const displayName =
    auth.profile?.display_name ?? auth.profile?.email.split('@')[0] ?? '访客研究员';
  const avatarPreset = playerProfile?.avatarPreset ?? selectedPreset;
  const runtimeAvatarPreset = playerProfile?.avatarPreset;
  const hasPlayerProfile = playerProfile !== null;

  useEffect(() => {
    currentSceneRef.current = currentScene;
  }, [currentScene]);

  useEffect(() => {
    currentPositionRef.current = sceneJoinPosition;
  }, [sceneJoinPosition]);

  useEffect(
    () => () => {
      window.clearTimeout(arrivalTimerRef.current);
    },
    [],
  );

  const persistProfile = useCallback(
    (next: PlayerGameProfile, cloud = false) => {
      latestProfileRef.current = next;
      saveLocalGameProfile(next, userId);
      if (cloud && userId) {
        void saveCloudGameProfile(userId, next).catch((error: unknown) => {
          console.warn('[research-game] cloud profile save failed', error);
        });
      }
    },
    [userId],
  );

  useEffect(() => {
    if (auth.loading) return;
    let active = true;

    const loadProfile = async () => {
      let loaded = loadLocalGameProfile(userId);
      if (userId) {
        try {
          loaded = (await loadCloudGameProfile(userId)) ?? loaded;
        } catch (error) {
          console.warn('[research-game] cloud profile load failed', error);
        }
      }
      if (!active) return;

      if (loaded) {
        const scene = initialSceneRef.current;
        const position =
          loaded.currentScene === scene ? loaded.lastPosition : gameScenes[scene].spawn;
        const adopted = { ...loaded, currentScene: scene, lastPosition: position };
        latestProfileRef.current = adopted;
        currentSceneRef.current = scene;
        currentPositionRef.current = position;
        setSelectedPreset(adopted.avatarPreset);
        setCurrentScene(scene);
        setSceneJoinPosition(position);
        setPlayerProfile(adopted);
      }
      setProfileReady(true);
    };

    void loadProfile();
    return () => {
      active = false;
    };
  }, [auth.loading, userId]);

  const { connected, remotePlayers, remoteEmote, broadcastPosition, broadcastEmote } = useGamePresence({
    userId,
    displayName,
    avatarPreset,
    sceneId: currentScene,
    initialPosition: sceneJoinPosition,
  });

  useEffect(() => {
    broadcastPositionRef.current = broadcastPosition;
  }, [broadcastPosition]);

  useEffect(() => {
    remotePlayersRef.current = remotePlayers;
    runtimeRef.current?.setRemotePlayers(remotePlayers);
  }, [remotePlayers]);

  useEffect(() => {
    if (remoteEmote?.emote === 'wave') runtimeRef.current?.waveRemote(remoteEmote.clientId);
  }, [remoteEmote]);

  useEffect(() => {
    const enabled = !openPanel && !customizingAvatar;
    gameInputEnabledRef.current = enabled;
    runtimeRef.current?.setInputEnabled(enabled);
  }, [customizingAvatar, openPanel]);

  useEffect(() => {
    if (!profileReady || !hasPlayerProfile) return;
    const canonicalScene = sceneFromRoute(routeSceneId);
    if (routeSceneId && !isGameSceneId(routeSceneId)) {
      navigate(scenePath(canonicalScene, force2D), { replace: true });
    }
    if (canonicalScene === currentSceneRef.current) return;

    const position = gameScenes[canonicalScene].spawn;
    currentSceneRef.current = canonicalScene;
    currentPositionRef.current = position;
    setCurrentScene(canonicalScene);
    setSceneJoinPosition(position);
    setOpenPanel(null);
    runtimeRef.current?.setScene(canonicalScene, position);
  }, [force2D, hasPlayerProfile, navigate, profileReady, routeSceneId]);

  useEffect(() => {
    const parent = mountRef.current;
    if (!parent || !profileReady || !runtimeAvatarPreset) return;
    let disposed = false;
    let runtime: ResearchGameRuntime | null = null;
    setRuntimeReady(false);
    setRuntimeError(null);

    const start = async () => {
      try {
        const { createResearchGame } = await import('./runtime');
        if (disposed) return;
        runtime = await createResearchGame(parent, {
          sceneId: currentSceneRef.current,
          position: currentPositionRef.current,
          avatarPreset: runtimeAvatarPreset,
          displayName,
          callbacks: {
            onInteraction: (interaction) => setOpenPanel(panelForInteraction(interaction)),
            onNearbyChange: setNearbyLabel,
            onMovementChange: setPlayerMoving,
            onPosition: (position) => {
              currentPositionRef.current = { x: position.x, y: position.y };
              broadcastPositionRef.current(position);
              const now = Date.now();
              const current = latestProfileRef.current;
              if (!current || now - lastLocalSaveRef.current < 1500) return;
              lastLocalSaveRef.current = now;
              persistProfile({
                ...current,
                currentScene: position.sceneId,
                lastPosition: { x: position.x, y: position.y },
              });
            },
            onSceneChange: (scene, position) => {
              currentSceneRef.current = scene;
              currentPositionRef.current = position;
              setCurrentScene(scene);
              setSceneJoinPosition(position);
              setOpenPanel(null);
              setPlayerMoving(false);
              setArrivalScene(scene);
              window.clearTimeout(arrivalTimerRef.current);
              arrivalTimerRef.current = window.setTimeout(() => setArrivalScene(null), 1800);
              const current = latestProfileRef.current;
              if (current) {
                const next = { ...current, currentScene: scene, lastPosition: position };
                setPlayerProfile(next);
                persistProfile(next, true);
              }
              navigate(scenePath(scene, force2D));
            },
          },
        });
        if (disposed) {
          runtime.destroy();
          return;
        }
        runtimeRef.current = runtime;
        runtime.setRemotePlayers(remotePlayersRef.current);
        runtime.setInputEnabled(gameInputEnabledRef.current);
        setRuntimeReady(true);
      } catch (error) {
        if (!disposed) {
          setRuntimeError(error instanceof Error ? error.message : '游戏场景加载失败');
        }
      }
    };

    void start();
    return () => {
      disposed = true;
      const current = latestProfileRef.current;
      if (current) {
        saveLocalGameProfile(
          {
            ...current,
            currentScene: currentSceneRef.current,
            lastPosition: currentPositionRef.current,
          },
          userId,
        );
      }
      runtimeRef.current = null;
      runtime?.destroy();
    };
  }, [
    displayName,
    force2D,
    navigate,
    persistProfile,
    profileReady,
    runtimeAvatarPreset,
    userId,
  ]);

  const confirmAvatar = () => {
    const scene = playerProfile?.currentScene ?? 'campus';
    const position = playerProfile?.lastPosition ?? gameScenes.campus.spawn;
    const next: PlayerGameProfile = {
      avatarPreset: selectedPreset,
      currentScene: scene,
      lastPosition: position,
      tutorialStep: Math.max(1, playerProfile?.tutorialStep ?? 0),
    };
    latestProfileRef.current = next;
    currentSceneRef.current = scene;
    currentPositionRef.current = position;
    setCurrentScene(scene);
    setSceneJoinPosition(position);
    setPlayerProfile(next);
    setCustomizingAvatar(false);
    persistProfile(next, true);
    navigate(scenePath(scene, force2D));
  };

  const travel = (scene: GameSceneId) => {
    setOpenPanel(null);
    runtimeRef.current?.setScene(scene, gameScenes[scene].spawn);
  };

  const openWorkspace = (zoneId: ZoneId) => {
    setOpenPanel({ kind: 'workspace', zoneId });
  };

  const wave = () => {
    runtimeRef.current?.wave();
    broadcastEmote('wave');
  };

  const scene = gameScenes[currentScene];
  const preset = avatarPresetById[avatarPreset];
  const showAvatarSetup = profileReady && (!playerProfile || customizingAvatar);

  return (
    <div
      className="research-game-ui relative h-dvh min-h-[520px] overflow-hidden bg-[#dce8e4] text-ink"
      data-current-scene={currentScene}
      data-player-moving={playerMoving ? 'true' : 'false'}
    >
      <div ref={mountRef} className="absolute inset-0 touch-none" aria-label={`${scene.name}游戏场景`} />

      <div className="pointer-events-none absolute inset-x-0 top-0 z-20 flex items-start justify-between gap-3 p-3 sm:p-4">
        <div className="max-w-[min(76vw,480px)] rounded-md border border-white/70 bg-paper/92 px-3 py-2 shadow-lift backdrop-blur-sm sm:px-4 sm:py-3">
          <p className="font-mono text-[8.5px] tracking-[0.14em] text-faint">RESEARCH COMMONS</p>
          <h1 className="mt-1 truncate font-serif text-[17px] font-semibold sm:text-[21px]">{scene.name}</h1>
          <p className="mt-1 hidden truncate text-[11.5px] text-faint sm:block">{scene.subtitle}</p>
        </div>

        <div className="pointer-events-auto flex items-center gap-2 rounded-md border border-white/70 bg-paper/92 p-1.5 shadow-lift backdrop-blur-sm">
          <span
            className={`hidden px-2 text-[10.5px] sm:inline ${connected ? 'text-ink-green' : 'text-faint'}`}
          >
            {connected ? `附近 ${remotePlayers.length}` : userId ? '正在连接' : '本地模式'}
          </span>
          <button
            type="button"
            onClick={wave}
            className="min-h-9 rounded-md border border-hairline bg-paper px-3 text-[11.5px] hover:bg-bone"
          >
            挥手
          </button>
          <button
            type="button"
            onClick={() => {
              setSelectedPreset(avatarPreset);
              setCustomizingAvatar(true);
            }}
            className="min-h-9 rounded-md border px-3 text-[11.5px] hover:bg-bone"
            style={{ borderColor: preset.cssColor }}
          >
            {preset.name}
          </button>
          <button
            type="button"
            onClick={() => navigate('/')}
            className="min-h-9 rounded-md border border-hairline bg-paper px-3 text-[11.5px] hover:bg-bone"
          >
            退出
          </button>
        </div>
      </div>

      {nearbyLabel && !openPanel && (
        <div className="pointer-events-none absolute bottom-5 left-1/2 z-20 -translate-x-1/2 rounded-md border border-white/70 bg-ink/88 px-4 py-2 text-center text-[12px] text-paper shadow-lift">
          {nearbyLabel}
        </div>
      )}

      {!openPanel && !customizingAvatar && (
        <CaseIslandHud caseFile={caseFile} />
      )}

      {arrivalScene && playerProfile && !customizingAvatar && (
        <div
          className="pointer-events-none absolute inset-0 z-30 flex items-center justify-center bg-ink/10"
          aria-live="polite"
        >
          <div className="scene-arrival w-full border-y border-white/70 bg-paper/92 px-5 py-5 text-center shadow-lift backdrop-blur-sm">
            <p className="font-mono text-[9px] tracking-[0.18em] text-faint">DESTINATION LOADED</p>
            <h2 className="mt-1 font-serif text-[25px] font-semibold sm:text-[32px]">
              {gameScenes[arrivalScene].name}
            </h2>
            <p className="mx-auto mt-1 max-w-xl text-[12px] text-faint sm:text-[13px]">
              {gameScenes[arrivalScene].subtitle}
            </p>
          </div>
        </div>
      )}

      {!runtimeReady && playerProfile && !runtimeError && (
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center bg-canvas/65 text-[13px] text-faint backdrop-blur-[1px]">
          正在进入科研公社……
        </div>
      )}

      {!profileReady && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-canvas text-[13px] text-faint">
          正在读取角色资料……
        </div>
      )}

      {runtimeError && (
        <div className="absolute inset-0 z-30 flex items-center justify-center bg-canvas/95 p-5">
          <div className="max-w-lg border-l-2 border-ink-red pl-4">
            <h2 className="font-serif text-[22px] font-semibold">场景加载失败</h2>
            <p className="mt-2 text-[13px] leading-relaxed text-ink-red">{runtimeError}</p>
            <button
              type="button"
              onClick={() => window.location.reload()}
              className="mt-4 min-h-10 rounded-md border border-hairline bg-paper px-4 text-[12.5px]"
            >
              重新加载
            </button>
          </div>
        </div>
      )}

      {openPanel?.kind === 'agent' && (
        <AgentPanel
          key={openPanel.agentId}
          agentId={openPanel.agentId}
          onClose={() => setOpenPanel(null)}
          onWorkspace={openWorkspace}
          onTravel={travel}
        />
      )}

      {openPanel?.kind === 'workspace' && (
        <ResearchWorkspaceOverlay
          zoneId={openPanel.zoneId}
          onClose={() => setOpenPanel(null)}
        />
      )}

      {showAvatarSetup && (
        <AvatarSetup
          selected={selectedPreset}
          displayName={displayName}
          confirmLabel={playerProfile ? '保存角色' : '进入科研公社'}
          onSelect={setSelectedPreset}
          onConfirm={confirmAvatar}
        />
      )}
    </div>
  );
}

function panelForInteraction(interaction: GameInteraction): OpenPanel {
  if (interaction.kind === 'agent') return { kind: 'agent', agentId: interaction.agentId };
  if (interaction.kind === 'workspace') {
    return { kind: 'workspace', zoneId: interaction.workspaceZone };
  }
  return null;
}

function sceneFromRoute(value: string | undefined): GameSceneId {
  if (isGameSceneId(value)) return value;
  if (value === 'observatory' || value === 'lab') return 'research';
  return 'campus';
}

function scenePath(scene: GameSceneId, force2D = false): string {
  const path = scene === 'campus' ? '/research-game' : `/research-game/${scene}`;
  return force2D ? `${path}?renderer=2d` : path;
}
