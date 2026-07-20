import {
  BookOpen,
  DoorOpen,
  Hand,
  LogOut,
  Map,
  MessageCircle,
  MonitorCog,
  UserRoundCog,
  X,
} from 'lucide-react';
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { useNavigate } from 'react-router-dom';
import { AvatarSetup } from '../game/AvatarSetup';
import {
  CaseEvidenceOverlay,
  CaseIslandHud,
  CaseMentorOverlay,
} from '../game/case/CaseIslandHud';
import { getCase, talkToMentor, useCase, visitPoi } from '../game/case/caseStore';
import { caseInteractionsForScene, isCaseMentorAgent } from '../game/case/caseWorld';
import {
  loadCloudGameProfile,
  loadLocalGameProfile,
  saveCloudGameProfile,
  saveLocalGameProfile,
} from '../game/gameProfile';
import { ResearchWorkspaceOverlay } from '../game/ResearchWorkspaceOverlay';
import type {
  AgentId,
  AvatarPresetId,
  GameSceneId,
  LocalPositionEvent,
  PlayerGameProfile,
} from '../game/types';
import { useGamePresence } from '../game/useGamePresence';
import { avatarPresetById, gameScenes } from '../game/worldContent';
import { useAuth } from '../lib/auth';
import type { ZoneId } from '../lib/researchGame';
import { legacyToWorld3D, world3DToLegacy } from './coordinates';
import { DiscoveryDialogueOverlay } from './DiscoveryDialogueOverlay';
import { MobileGameControls } from './MobileGameControls';
import { createNavigationCommand } from './navigation3d';
import { NpcDialogueOverlay } from './NpcDialogueOverlay';
import { ResearchGame3DCanvas } from './ResearchGame3DCanvas';
import type {
  CameraMode3D,
  Game3DInteraction,
  Game3DSceneDefinition,
  LandmarkInteraction3D,
  NavigationCommand3D,
  PlayerWorldState3D,
  Vec3,
} from './types3d';
import { clearMovement3D, useGame3DInput } from './useGame3DInput';
import { gameScenes3D } from './world3d';

type OpenPanel =
  | { kind: 'dialogue'; agentId: AgentId; interaction: Game3DInteraction }
  | { kind: 'discovery'; interaction: LandmarkInteraction3D; isNew: boolean }
  | { kind: 'workspace'; zoneId: ZoneId; interaction: Game3DInteraction | null }
  | {
      kind: 'case-evidence';
      title: string;
      body: string;
      ok: boolean;
      reason?: string;
    }
  | { kind: 'case-mentor' }
  | null;

export function ResearchGame3DShell({
  sceneId,
}: {
  sceneId: Game3DSceneDefinition['id'];
}) {
  const auth = useAuth();
  const navigate = useNavigate();
  const caseFile = useCase();
  const scene = useMemo((): Game3DSceneDefinition => {
    const base = gameScenes3D[sceneId];
    // Only inject case POIs when a case is active (keeps non-case exploration clean).
    if (!caseFile || caseFile.status === 'closed') return base;
    const extra = caseInteractionsForScene(sceneId);
    if (extra.length === 0) return base;
    const seen = new Set(base.interactions.map((i) => i.id));
    return {
      ...base,
      interactions: [...base.interactions, ...extra.filter((i) => !seen.has(i.id))],
    };
  }, [caseFile, sceneId]);
  const userId = auth.profile?.id ?? null;
  const [mentorMsg, setMentorMsg] = useState<string | null>(null);
  const displayName =
    auth.profile?.display_name ?? auth.profile?.email.split('@')[0] ?? '访客研究员';
  const initialSceneRef = useRef(sceneId);
  const latestProfileRef = useRef<PlayerGameProfile | null>(null);
  const currentWorldPositionRef = useRef<Vec3>([...scene.spawn]);
  const nearbyRef = useRef<Game3DInteraction | null>(null);
  const broadcastPositionRef = useRef<(position: LocalPositionEvent) => void>(() => undefined);
  const lastLocalSaveRef = useRef(0);
  const transitionTimerRef = useRef<number | undefined>(undefined);
  const navigationIdRef = useRef(0);
  const [profileReady, setProfileReady] = useState(false);
  const [playerProfile, setPlayerProfile] = useState<PlayerGameProfile | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<AvatarPresetId>('theory-blue');
  const [joinPosition, setJoinPosition] = useState<Vec3>([...scene.spawn]);
  const [customizingAvatar, setCustomizingAvatar] = useState(false);
  const [openPanel, setOpenPanel] = useState<OpenPanel>(null);
  const [nearby, setNearby] = useState<Game3DInteraction | null>(null);
  const [mapOpen, setMapOpen] = useState(false);
  const [runtimeReady, setRuntimeReady] = useState(false);
  const [playerMoving, setPlayerMoving] = useState(false);
  const [navigationCommand, setNavigationCommand] = useState<NavigationCommand3D | null>(null);
  const [transitionTarget, setTransitionTarget] = useState<GameSceneId | null>(null);
  const [arrivalVisible, setArrivalVisible] = useState(sceneId !== 'campus');
  const [discoveredLandmarks, setDiscoveredLandmarks] = useState<string[]>(
    () => loadDiscoveredLandmarks(userId),
  );
  const reducedMotion = useReducedMotion();

  const avatarPreset = playerProfile?.avatarPreset ?? selectedPreset;
  const preset = avatarPresetById[avatarPreset];
  const landmarkTotal = useMemo(
    () => scene.interactions.filter((interaction) => interaction.kind === 'landmark').length,
    [scene.interactions],
  );
  const discoveredInScene = useMemo(
    () => scene.interactions.filter(
      (interaction) => interaction.kind === 'landmark' && discoveredLandmarks.includes(interaction.id),
    ).length,
    [discoveredLandmarks, scene.interactions],
  );

  useEffect(() => {
    let active = true;
    Promise.resolve().then(() => {
      if (active) setDiscoveredLandmarks(loadDiscoveredLandmarks(userId));
    });
    return () => {
      active = false;
    };
  }, [userId]);

  const persistProfile = useCallback(
    (next: PlayerGameProfile, cloud = false) => {
      latestProfileRef.current = next;
      saveLocalGameProfile(next, userId);
      if (cloud && userId) {
        void saveCloudGameProfile(userId, next).catch((error: unknown) => {
          console.warn('[research-game-3d] cloud profile save failed', error);
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
          console.warn('[research-game-3d] cloud profile load failed', error);
        }
      }
      if (!active) return;
      if (loaded) {
        const routeScene = initialSceneRef.current;
        const legacyPosition =
          loaded.currentScene === routeScene ? loaded.lastPosition : gameScenes[routeScene].spawn;
        const worldPosition = legacyToWorld3D(routeScene, legacyPosition);
        const adopted: PlayerGameProfile = {
          ...loaded,
          currentScene: routeScene,
          lastPosition: legacyPosition,
        };
        latestProfileRef.current = adopted;
        currentWorldPositionRef.current = worldPosition;
        setSelectedPreset(adopted.avatarPreset);
        setJoinPosition(worldPosition);
        setPlayerProfile(adopted);
      }
      setProfileReady(true);
    };
    void loadProfile();
    return () => {
      active = false;
    };
  }, [auth.loading, userId]);

  useEffect(() => {
    if (!arrivalVisible) return;
    const timer = window.setTimeout(() => setArrivalVisible(false), 1800);
    return () => window.clearTimeout(timer);
  }, [arrivalVisible]);

  useEffect(
    () => () => {
      window.clearTimeout(transitionTimerRef.current);
      const current = latestProfileRef.current;
      if (!current) return;
      const [x, , z] = currentWorldPositionRef.current;
      saveLocalGameProfile(
        {
          ...current,
          currentScene: sceneId,
          lastPosition: world3DToLegacy(sceneId, { x, z }),
        },
        userId,
      );
      delete window.__THREE_GAME_DIAGNOSTICS__;
    },
    [sceneId, userId],
  );

  const presenceUserId = playerProfile ? userId : null;
  const presenceStart = useMemo(() => {
    const [x, , z] = joinPosition;
    return world3DToLegacy(sceneId, { x, z });
  }, [joinPosition, sceneId]);
  const { connected, remotePlayers, broadcastPosition, broadcastEmote } =
    useGamePresence({
      userId: presenceUserId,
      displayName,
      avatarPreset,
      sceneId,
      initialPosition: presenceStart,
    });

  useEffect(() => {
    broadcastPositionRef.current = broadcastPosition;
  }, [broadcastPosition]);

  const travel = useCallback(
    (targetScene: GameSceneId) => {
      if (transitionTarget) return;
      const current = latestProfileRef.current;
      const targetPosition = gameScenes[targetScene].spawn;
      if (current) {
        const next: PlayerGameProfile = {
          ...current,
          currentScene: targetScene,
          lastPosition: targetPosition,
          tutorialStep: Math.max(current.tutorialStep, targetScene === 'academy' ? 2 : 1),
        };
        setPlayerProfile(next);
        persistProfile(next, true);
      }
      setNavigationCommand(null);
      setMapOpen(false);
      setOpenPanel(null);
      setTransitionTarget(targetScene);
      transitionTimerRef.current = window.setTimeout(() => {
        navigate(scenePath(targetScene));
      }, reducedMotion ? 0 : 360);
    },
    [navigate, persistProfile, reducedMotion, transitionTarget],
  );

  const activateInteraction = useCallback(
    (interaction: Game3DInteraction) => {
      if (transitionTarget) return;
      if (interaction.kind === 'portal') {
        travel(interaction.targetScene);
        return;
      }
      if (interaction.kind === 'workspace') {
        setMapOpen(false);
        setOpenPanel({
          kind: 'workspace',
          zoneId: interaction.workspaceZone,
          interaction,
        });
        return;
      }
      if (interaction.kind === 'landmark') {
        // Case POIs write into the shared case store (detective v0).
        if (interaction.casePoiId && getCase()) {
          const r = visitPoi(interaction.casePoiId);
          setMapOpen(false);
          setOpenPanel({
            kind: 'case-evidence',
            title: interaction.title,
            body: interaction.note,
            ok: r.ok,
            reason: r.reason,
          });
          return;
        }
        const isNew = !discoveredLandmarks.includes(interaction.id);
        const next = isNew ? [...discoveredLandmarks, interaction.id] : discoveredLandmarks;
        if (isNew) {
          setDiscoveredLandmarks(next);
          saveDiscoveredLandmarks(userId, next);
        }
        setMapOpen(false);
        setOpenPanel({ kind: 'discovery', interaction, isNew });
        return;
      }
      // Case mentor gate: intercept academy mentor when a case is active.
      if (
        interaction.kind === 'agent' &&
        isCaseMentorAgent(interaction.agentId) &&
        getCase() &&
        getCase()?.status !== 'closed'
      ) {
        setMapOpen(false);
        setMentorMsg(null);
        setOpenPanel({ kind: 'case-mentor' });
        return;
      }
      setMapOpen(false);
      setOpenPanel({ kind: 'dialogue', agentId: interaction.agentId, interaction });
    },
    [discoveredLandmarks, transitionTarget, travel, userId],
  );

  const interactNearby = useCallback(() => {
    if (openPanel || mapOpen) return;
    const interaction = nearbyRef.current;
    if (interaction) {
      setNavigationCommand(null);
      activateInteraction(interaction);
    }
  }, [activateInteraction, mapOpen, openPanel]);

  const closeOverlay = useCallback(() => {
    if (openPanel) setOpenPanel(null);
    else if (mapOpen) setMapOpen(false);
    else setNavigationCommand(null);
  }, [mapOpen, openPanel]);

  const toggleMap = useCallback(() => {
    if (openPanel || transitionTarget || !playerProfile) return;
    setMapOpen((current) => !current);
  }, [openPanel, playerProfile, transitionTarget]);

  const inputEnabled =
    runtimeReady && Boolean(playerProfile) && !openPanel && !customizingAvatar && !transitionTarget;
  const movementRef = useGame3DInput({
    movementEnabled: inputEnabled && !mapOpen,
    shortcutsEnabled: Boolean(playerProfile) && !customizingAvatar && !transitionTarget,
    onInteract: interactNearby,
    onToggleMap: toggleMap,
    onEscape: closeOverlay,
  });

  useEffect(() => {
    if (!inputEnabled || mapOpen) clearMovement3D(movementRef);
  }, [inputEnabled, mapOpen, movementRef]);

  const navigateTo = useCallback(
    (target: Vec3, interaction: Game3DInteraction | null = null) => {
      if (!inputEnabled || mapOpen || transitionTarget) return;
      const command = createNavigationCommand(
        ++navigationIdRef.current,
        scene,
        currentWorldPositionRef.current,
        target,
        interaction,
      );
      setNavigationCommand(command);
    },
    [inputEnabled, mapOpen, scene, transitionTarget],
  );

  const selectInteraction = useCallback(
    (interaction: Game3DInteraction) => {
      navigateTo(interaction.position, interaction);
    },
    [navigateTo],
  );

  const completeNavigation = useCallback(
    (command: NavigationCommand3D) => {
      setNavigationCommand((current) => (current?.id === command.id ? null : current));
      if (command.interaction) activateInteraction(command.interaction);
    },
    [activateInteraction],
  );

  const handlePosition = useCallback(
    (state: PlayerWorldState3D) => {
      currentWorldPositionRef.current = [state.x, state.y, state.z];
      const legacy = world3DToLegacy(sceneId, state);
      broadcastPositionRef.current({
        sceneId,
        x: legacy.x,
        y: legacy.y,
        direction: state.direction,
      });
      const now = Date.now();
      const current = latestProfileRef.current;
      if (!current || now - lastLocalSaveRef.current < 1500) return;
      lastLocalSaveRef.current = now;
      persistProfile({ ...current, currentScene: sceneId, lastPosition: legacy });
    },
    [persistProfile, sceneId],
  );

  const handleNearbyChange = useCallback((interaction: Game3DInteraction | null) => {
    nearbyRef.current = interaction;
    setNearby(interaction);
  }, []);

  const confirmAvatar = () => {
    const legacyPosition = playerProfile?.currentScene === sceneId
      ? playerProfile.lastPosition
      : gameScenes[sceneId].spawn;
    const worldPosition = legacyToWorld3D(sceneId, legacyPosition);
    const next: PlayerGameProfile = {
      avatarPreset: selectedPreset,
      currentScene: sceneId,
      lastPosition: legacyPosition,
      tutorialStep: Math.max(1, playerProfile?.tutorialStep ?? 0),
    };
    latestProfileRef.current = next;
    currentWorldPositionRef.current = worldPosition;
    setJoinPosition(worldPosition);
    setPlayerProfile(next);
    setCustomizingAvatar(false);
    persistProfile(next, true);
  };

  const openWorkspace = (zoneId: ZoneId) => {
    setOpenPanel({ kind: 'workspace', zoneId, interaction: null });
  };

  const wave = () => {
    broadcastEmote('wave');
  };

  const cameraMode: CameraMode3D = mapOpen
    ? 'map'
    : openPanel?.kind === 'dialogue' ||
        openPanel?.kind === 'discovery' ||
        openPanel?.kind === 'case-evidence' ||
        openPanel?.kind === 'case-mentor'
      ? 'dialogue'
      : openPanel?.kind === 'workspace'
        ? 'workstation'
        : 'follow';
  const focusPosition =
    openPanel && 'interaction' in openPanel && openPanel.interaction
      ? openPanel.interaction.position
      : null;
  const remotePlayers3D = remotePlayers.map((player) => ({
    clientId: player.clientId,
    displayName: player.displayName,
    avatarPreset: player.avatarPreset,
    position: groundedRemotePosition(legacyToWorld3D(sceneId, { x: player.x, y: player.y })),
    direction: player.direction,
    updatedAt: player.updatedAt,
  }));
  const showAvatarSetup = profileReady && (!playerProfile || customizingAvatar);
  const NearbyIcon = nearby?.kind === 'portal'
    ? DoorOpen
    : nearby?.kind === 'workspace'
      ? MonitorCog
      : nearby?.kind === 'landmark'
        ? BookOpen
        : MessageCircle;
  const caseHudMentor =
    sceneId === 'academy' && caseFile && caseFile.status !== 'closed'
      ? () => {
          setMentorMsg(null);
          setOpenPanel({ kind: 'case-mentor' });
        }
      : undefined;

  return (
    <div
      className="research-game-ui game3d-shell relative h-dvh min-h-[520px] overflow-hidden bg-[#b8dce2] text-ink"
      data-renderer="three"
      data-current-scene={sceneId}
      data-camera-mode={cameraMode}
      data-player-moving={playerMoving ? 'true' : 'false'}
      data-discovered-count={discoveredInScene}
      data-discovery-total={landmarkTotal}
    >
      {playerProfile && (
        <div className="absolute inset-0 touch-none" aria-label={`${scene.name} 3D 游戏场景`}>
          <ResearchGame3DCanvas
            scene={scene}
            avatarPreset={avatarPreset}
            initialPosition={joinPosition}
            movementRef={movementRef}
            cameraMode={cameraMode}
            inputEnabled={inputEnabled}
            reducedMotion={reducedMotion}
            remotePlayers={remotePlayers3D}
            focusPosition={focusPosition}
            onPosition={handlePosition}
            onMovementChange={setPlayerMoving}
            onNearbyChange={handleNearbyChange}
            navigationCommand={navigationCommand}
            onSelectGround={(position) => navigateTo(position)}
            onSelectInteraction={selectInteraction}
            onActivateInteraction={activateInteraction}
            onNavigationComplete={completeNavigation}
            onReady={() => setRuntimeReady(true)}
          />
        </div>
      )}

      <header className="pointer-events-none absolute inset-x-0 top-0 z-20 flex items-start justify-between gap-3 p-3 pt-[max(0.75rem,env(safe-area-inset-top))] sm:p-4">
        <div className="game3d-hud-panel max-w-[min(70vw,470px)] px-3 py-2.5 sm:px-4 sm:py-3">
          <p className="font-mono text-[10px] tracking-[0.14em] text-white/65">RESEARCH ISLAND · 3D</p>
          <h1 className="mt-1 truncate font-serif text-[22px] font-semibold text-white sm:text-[26px]">{scene.name}</h1>
          <p className="mt-1 hidden truncate text-[14px] text-white/68 sm:block">{scene.subtitle}</p>
        </div>

        <nav className="game3d-hud-panel pointer-events-auto flex items-center gap-1.5 p-1.5" aria-label="游戏操作">
          <span className={`hidden px-2 text-[13px] sm:inline ${connected ? 'text-[#a9efbb]' : 'text-white/58'}`}>
            {connected ? `附近 ${remotePlayers.length}` : userId ? '正在连接' : '本地模式'}
          </span>
          <HudIconButton label={mapOpen ? '关闭世界地图' : '打开世界地图'} onClick={toggleMap} active={mapOpen}>
            {mapOpen ? <X size={18} /> : <Map size={18} />}
          </HudIconButton>
          <HudIconButton label="挥手" onClick={wave}><Hand size={18} /></HudIconButton>
          <HudIconButton
            label={`调整角色：${preset.name}`}
            onClick={() => {
              setSelectedPreset(avatarPreset);
              setCustomizingAvatar(true);
            }}
            accent={preset.cssColor}
          >
            <UserRoundCog size={18} />
          </HudIconButton>
          <HudIconButton label="退出科研岛" onClick={() => navigate('/')}><LogOut size={18} /></HudIconButton>
        </nav>
      </header>

      {mapOpen && (
        <div className="pointer-events-none absolute inset-x-0 top-[118px] z-20 flex justify-center px-4 sm:top-24" aria-live="polite">
          <div className="game3d-hud-panel px-4 py-2 text-center text-white">
            <p className="font-mono text-[11px] tracking-[0.16em] text-white/60">WORLD MAP</p>
            <h2 className="mt-0.5 font-serif text-[22px] font-semibold">{scene.name}全景</h2>
            {landmarkTotal > 0 && (
              <p className="mt-1 font-mono text-[12px] text-[#f0d78d]">科研手记 {discoveredInScene}/{landmarkTotal}</p>
            )}
          </div>
        </div>
      )}

      {landmarkTotal > 0 && !mapOpen && !openPanel && !transitionTarget && (
        <div className="game3d-discovery-tracker pointer-events-none absolute left-3 top-[112px] z-20 sm:left-4 sm:top-[126px]">
          <BookOpen size={15} aria-hidden="true" />
          <span>科研手记</span>
          <strong>{discoveredInScene}/{landmarkTotal}</strong>
          <i aria-hidden="true"><b style={{ width: `${discoveredInScene / landmarkTotal * 100}%` }} /></i>
        </div>
      )}

      {nearby && !openPanel && !mapOpen && !transitionTarget && (
        <button
          type="button"
          onClick={interactNearby}
          className="game3d-interaction-prompt absolute bottom-5 left-1/2 z-20 hidden min-h-12 -translate-x-1/2 items-center gap-2.5 px-5 py-2.5 text-[15px] text-white md:flex"
        >
          <NearbyIcon size={17} aria-hidden="true" />
          <span>{nearby.label}</span>
        </button>
      )}

      <MobileGameControls
        movementRef={movementRef}
        nearby={nearby}
        mapOpen={mapOpen}
        disabled={!inputEnabled}
        onInteract={interactNearby}
        onToggleMap={toggleMap}
        onManualMovement={() => setNavigationCommand(null)}
      />

      {!runtimeReady && playerProfile && (
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center bg-[#b8dce2] text-[16px] text-[#29464d]">
          正在构建 3D 科研岛……
        </div>
      )}

      {!profileReady && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-canvas text-[16px] text-faint">
          正在读取角色资料……
        </div>
      )}

      {arrivalVisible && playerProfile && runtimeReady && (
        <div className="pointer-events-none absolute inset-0 z-30 flex items-center justify-center bg-[#17252a]/10" aria-live="polite">
          <div className="scene-arrival w-full border-y border-white/60 bg-[#17252a]/82 px-5 py-5 text-center text-white backdrop-blur-sm">
            <p className="font-mono text-[11px] tracking-[0.18em] text-white/60">DESTINATION LOADED</p>
            <h2 className="mt-1 font-serif text-[32px] font-semibold sm:text-[38px]">{scene.name}</h2>
            <p className="mx-auto mt-1 max-w-xl text-[15px] text-white/70">{scene.subtitle}</p>
          </div>
        </div>
      )}

      {transitionTarget && (
        <div className="game3d-scene-transition pointer-events-none absolute inset-0 z-40 bg-[#17252a]" aria-label={`正在前往${gameScenes[transitionTarget].name}`} />
      )}

      {openPanel?.kind === 'dialogue' && (
        <NpcDialogueOverlay
          key={openPanel.agentId}
          agentId={openPanel.agentId}
          reducedMotion={reducedMotion}
          onClose={() => setOpenPanel(null)}
          onWorkspace={openWorkspace}
          onTravel={travel}
        />
      )}

      {openPanel?.kind === 'discovery' && (
        <DiscoveryDialogueOverlay
          interaction={openPanel.interaction}
          isNew={openPanel.isNew}
          discoveredCount={discoveredInScene}
          total={landmarkTotal}
          onClose={() => setOpenPanel(null)}
        />
      )}

      {openPanel?.kind === 'workspace' && (
        <ResearchWorkspaceOverlay zoneId={openPanel.zoneId} onClose={() => setOpenPanel(null)} />
      )}

      {openPanel?.kind === 'case-evidence' && (
        <CaseEvidenceOverlay
          title={openPanel.title}
          body={openPanel.body}
          ok={openPanel.ok}
          reason={openPanel.reason}
          onClose={() => setOpenPanel(null)}
        />
      )}

      {openPanel?.kind === 'case-mentor' && caseFile && (
        <CaseMentorOverlay
          flags={caseFile.flags}
          message={mentorMsg}
          onVague={() => {
            const r = talkToMentor(false);
            setMentorMsg(r.message.zh);
          }}
          onClear={() => {
            const r = talkToMentor(true);
            setMentorMsg(r.message.zh);
          }}
          onClose={() => setOpenPanel(null)}
        />
      )}

      {!openPanel && !mapOpen && !transitionTarget && !showAvatarSetup && (
        <CaseIslandHud caseFile={caseFile} onOpenMentor={caseHudMentor} />
      )}

      {showAvatarSetup && (
        <AvatarSetup
          selected={selectedPreset}
          displayName={displayName}
          confirmLabel={playerProfile ? '保存角色' : '进入 3D 科研岛'}
          onSelect={setSelectedPreset}
          onConfirm={confirmAvatar}
        />
      )}
    </div>
  );
}

function HudIconButton({
  label,
  onClick,
  children,
  active = false,
  accent,
}: {
  label: string;
  onClick: () => void;
  children: ReactNode;
  active?: boolean;
  accent?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`grid h-9 w-9 shrink-0 place-items-center rounded-md border text-white transition-colors ${active ? 'border-[#f4c850] bg-[#f4c850]/22' : 'border-white/20 bg-white/8 hover:bg-white/15'}`}
      style={accent ? { borderColor: accent } : undefined}
      aria-label={label}
      title={label}
    >
      {children}
    </button>
  );
}

function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () => typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  );
  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onChange = () => setReduced(media.matches);
    media.addEventListener('change', onChange);
    return () => media.removeEventListener('change', onChange);
  }, []);
  return reduced;
}

function scenePath(scene: GameSceneId): string {
  return scene === 'campus' ? '/research-game' : `/research-game/${scene}`;
}

function groundedRemotePosition(position: Vec3): Vec3 {
  return [position[0], 0.8, position[2]];
}

const DISCOVERY_STORAGE_PREFIX = 'research-game:field-notes-v1';

function discoveryStorageKey(userId: string | null): string {
  return `${DISCOVERY_STORAGE_PREFIX}:${userId ?? 'guest'}`;
}

function loadDiscoveredLandmarks(userId: string | null): string[] {
  try {
    const raw = window.localStorage.getItem(discoveryStorageKey(userId));
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((value): value is string => typeof value === 'string') : [];
  } catch {
    return [];
  }
}

function saveDiscoveredLandmarks(userId: string | null, ids: string[]): void {
  window.localStorage.setItem(discoveryStorageKey(userId), JSON.stringify(ids));
}
