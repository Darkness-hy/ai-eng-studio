import {
  BookOpen,
  ClipboardList,
  DoorOpen,
  Eye,
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
  CaseExperimentOverlay,
  CaseIslandHud,
  CaseMentorOverlay,
} from '../game/case/CaseIslandHud';
import {
  canRunCaseExperiment,
  getCase,
  runCaseExperiment,
  talkToMentor,
  useCase,
  visitPoi,
} from '../game/case/caseStore';
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
import { gameScenes } from '../game/worldContent';
import { useAuth } from '../lib/auth';
import type { ZoneId } from '../lib/researchGame';
import { CaseWorkbenchPage } from '../pages/CaseWorkbenchPage';
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
  | { kind: 'case-experiment' }
  | { kind: 'case-desk' }
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
  const [expBusy, setExpBusy] = useState(false);
  const [expDetail, setExpDetail] = useState<string | null>(null);
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
  const [personView, setPersonView] = useState<'third' | 'first'>('third');
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
  const { remotePlayers, broadcastPosition } =
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
          setMapOpen(false);
          if (interaction.casePoiId === 'poi_bench') {
            setOpenPanel({ kind: 'case-experiment' });
            return;
          }
          const r = visitPoi(interaction.casePoiId);
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
    if (openPanel?.kind === 'case-desk') {
      setOpenPanel(null);
      return;
    }
    if (openPanel || transitionTarget || !playerProfile) return;
    setMapOpen((current) => !current);
  }, [openPanel, playerProfile, transitionTarget]);

  const toggleDossier = useCallback(() => {
    if (transitionTarget || !playerProfile) return;
    setMapOpen(false);
    setOpenPanel((current) => (current?.kind === 'case-desk' ? null : { kind: 'case-desk' }));
  }, [playerProfile, transitionTarget]);

  const togglePersonView = useCallback(() => {
    if (mapOpen || openPanel || transitionTarget || !playerProfile) return;
    setPersonView((v) => {
      const next = v === 'third' ? 'first' : 'third';
      // Drop click-pathfinding so first-person keyboard is never blocked by commandMoving.
      if (next === 'first') setNavigationCommand(null);
      return next;
    });
  }, [mapOpen, openPanel, playerProfile, transitionTarget]);

  const inputEnabled =
    runtimeReady && Boolean(playerProfile) && !openPanel && !customizingAvatar && !transitionTarget;
  const movementRef = useGame3DInput({
    movementEnabled: inputEnabled && !mapOpen,
    shortcutsEnabled: Boolean(playerProfile) && !customizingAvatar && !transitionTarget,
    onInteract: interactNearby,
    onToggleMap: toggleMap,
    onToggleDossier: toggleDossier,
    onTogglePersonView: togglePersonView,
    onEscape: closeOverlay,
  });

  useEffect(() => {
    if (!inputEnabled || mapOpen) clearMovement3D(movementRef);
  }, [inputEnabled, mapOpen, movementRef]);

  const navigateTo = useCallback(
    (target: Vec3, interaction: Game3DInteraction | null = null) => {
      // First-person is keyboard/look driven; skip click-to-walk pathfinding.
      if (!inputEnabled || mapOpen || transitionTarget || personView === 'first') return;
      const command = createNavigationCommand(
        ++navigationIdRef.current,
        scene,
        currentWorldPositionRef.current,
        target,
        interaction,
      );
      setNavigationCommand(command);
    },
    [inputEnabled, mapOpen, personView, scene, transitionTarget],
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

  const cameraMode: CameraMode3D = mapOpen
    ? 'map'
    : openPanel?.kind === 'dialogue' ||
        openPanel?.kind === 'discovery' ||
        openPanel?.kind === 'case-evidence' ||
        openPanel?.kind === 'case-mentor' ||
        openPanel?.kind === 'case-experiment' ||
        openPanel?.kind === 'case-desk'
      ? 'dialogue'
      : openPanel?.kind === 'workspace'
        ? 'workstation'
        : personView === 'first'
          ? 'first'
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
      className="research-game-ui game3d-shell relative h-dvh min-h-[520px] overflow-hidden bg-[#b8dff5] text-ink"
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
        <div className="game3d-hud-panel max-w-[min(72vw,420px)] px-3.5 py-2.5 text-[#2a3a36]">
          <h1 className="truncate font-serif text-[20px] font-semibold text-[#1e332c] sm:text-[24px]">{scene.name}</h1>
          <p className="mt-1 truncate text-[14px] leading-snug text-[#3d5a50]">
            {caseFile && caseFile.status !== 'closed'
              ? (caseFile.completeness >= 0.85
                  ? '卷宗将齐——打开组板送审'
                  : caseFile.flags.includes('knows_leak')
                    ? '去档案柜与机房钉证据'
                    : '开放世界 · WASD 探索 · E 交互 · V 视角')
              : scene.subtitle}
          </p>
        </div>

        <nav className="game3d-hud-panel pointer-events-auto flex items-center gap-1 p-1" aria-label="游戏操作">
          <HudIconButton label={mapOpen ? '关闭地图' : '地图 M'} onClick={toggleMap} active={mapOpen}>
            {mapOpen ? <X size={18} /> : <Map size={18} />}
          </HudIconButton>
          <HudIconButton
            label={personView === 'first' ? '第三人称 V' : '第一人称 V'}
            onClick={togglePersonView}
            active={personView === 'first'}
          >
            <Eye size={18} />
          </HudIconButton>
          <HudIconButton
            label="卷宗 J"
            onClick={toggleDossier}
            active={openPanel?.kind === 'case-desk'}
          >
            <ClipboardList size={18} />
          </HudIconButton>
          <HudIconButton
            label="角色"
            onClick={() => {
              setSelectedPreset(avatarPreset);
              setCustomizingAvatar(true);
            }}
          >
            <UserRoundCog size={18} />
          </HudIconButton>
          <HudIconButton label="退出" onClick={() => navigate('/')}>
            <LogOut size={18} />
          </HudIconButton>
        </nav>
      </header>

      {mapOpen && (
        <div className="pointer-events-none absolute inset-x-0 top-[108px] z-20 flex justify-center px-4" aria-live="polite">
          <div className="game3d-map-banner px-5 py-3 text-center text-white">
            <p className="font-mono text-[11px] tracking-[0.12em] text-white/55">园区总览</p>
            <h2 className="mt-0.5 font-serif text-[22px] font-semibold">{scene.name}</h2>
            <p className="mt-1 text-[13px] text-white/65">点选目的地移动 · 再按 M 关闭</p>
          </div>
        </div>
      )}

      {nearby && !openPanel && !mapOpen && !transitionTarget && (
        <button
          type="button"
          onClick={interactNearby}
          className="game3d-interaction-prompt absolute bottom-5 left-1/2 z-20 hidden min-h-12 -translate-x-1/2 items-center gap-2.5 px-5 py-2.5 text-[16px] text-[#1e332c] md:flex"
        >
          <kbd className="rounded border border-emerald-900/25 bg-white/80 px-1.5 font-mono text-[12px] text-[#1e332c]">E</kbd>
          <NearbyIcon size={18} aria-hidden="true" />
          <span>{nearby.label}</span>
        </button>
      )}

      {personView === 'first' && !openPanel && !mapOpen && !transitionTarget && !nearby && (
        <div className="pointer-events-none absolute bottom-5 left-1/2 z-20 hidden -translate-x-1/2 rounded border border-emerald-900/15 bg-white/90 px-3 py-1.5 font-mono text-[12px] text-[#2a4a40] shadow md:block">
          WASD 移动 · 点击画面锁定视角 · E 交互 · V 第三人称
        </div>
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
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center bg-[#b8dff5] text-[16px] text-[#2a4a40]">
          正在流式加载体素大陆与洞穴……
        </div>
      )}

      {!profileReady && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-canvas text-[16px] text-faint">
          正在读取角色资料……
        </div>
      )}

      {arrivalVisible && playerProfile && runtimeReady && (
        <div className="pointer-events-none absolute inset-0 z-30 flex items-center justify-center bg-[#17252a]/10" aria-live="polite">
          <div className="scene-arrival mx-auto max-w-lg rounded-lg border border-white/15 bg-[#121c20]/88 px-6 py-5 text-center text-white backdrop-blur-md">
            <h2 className="font-serif text-[28px] font-semibold sm:text-[34px]">{scene.name}</h2>
            <p className="mx-auto mt-2 max-w-md text-[15px] text-white/72">
              {caseFile && caseFile.status !== 'closed' ? '在园区走动 · 靠近线索按 E' : scene.subtitle}
            </p>
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
          mentorStage={caseFile.mentorStage}
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

      {openPanel?.kind === 'case-experiment' && caseFile && (
        <CaseExperimentOverlay
          canRun={canRunCaseExperiment(caseFile)}
          busy={expBusy}
          lastDetail={expDetail}
          onRun={() => {
            setExpBusy(true);
            void runCaseExperiment()
              .then((r) => {
                setExpDetail(
                  r.mode === 'live'
                    ? `live · ${r.link.researchRunId?.slice(0, 8) ?? ''}…`
                    : r.detail || 'simulated',
                );
              })
              .catch(() => setExpDetail('soft_gate_evidence'))
              .finally(() => setExpBusy(false));
          }}
          onClose={() => setOpenPanel(null)}
        />
      )}

      {!openPanel && !mapOpen && !transitionTarget && !showAvatarSetup && (
        <CaseIslandHud
          caseFile={caseFile}
          onOpenMentor={caseHudMentor}
          onOpenDesk={() => {
            setMapOpen(false);
            setOpenPanel({ kind: 'case-desk' });
          }}
        />
      )}

      {openPanel?.kind === 'case-desk' && (
        <div className="case-desk-shell-overlay absolute inset-0 z-50">
          <button
            type="button"
            className="absolute right-3 top-3 z-[60] min-h-10 rounded-md border border-white/20 bg-[#121c20]/9 px-3 text-[13px] text-white/90 hover:bg-white/10 sm:right-5 sm:top-4"
            onClick={() => setOpenPanel(null)}
          >
            返回岛上 · Esc
          </button>
          <CaseWorkbenchPage />
        </div>
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
      className={`grid h-10 w-10 shrink-0 place-items-center rounded-md border transition-colors ${active ? 'border-[#5a9a88] bg-[#d8f0e6] text-[#1e332c]' : 'border-[#8ab5a8]/55 bg-white/50 text-[#2a3a36] hover:bg-white/80'}`}
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
