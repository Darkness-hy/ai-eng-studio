import Phaser from 'phaser';
import { agents, avatarPresetById, avatarPresets, gameScenes } from './worldContent';
import type {
  AvatarPresetId,
  GameInteraction,
  GameRuntimeCallbacks,
  GameSceneId,
  LocalPositionEvent,
  RemotePlayerState,
  ResearchGameRuntime,
  WorldPoint,
} from './types';

const WORLD_WIDTH = 1600;
const WORLD_HEIGHT = 900;
const MOVE_SPEED = 300;
const INTERACTION_DISTANCE = 90;
const AUTO_PORTAL_DISTANCE = 58;
const WALK_STEP_MS = 185;

type RemoteVisual = {
  sprite: Phaser.GameObjects.Image;
  shadow: Phaser.GameObjects.Ellipse;
  label: Phaser.GameObjects.Text;
  target: WorldPoint;
  baseScaleX: number;
  baseScaleY: number;
  walkPhase: number;
};

class ResearchWorldScene extends Phaser.Scene {
  private readonly callbacks: GameRuntimeCallbacks;
  private readonly avatarPreset: AvatarPresetId;
  private readonly displayName: string;
  private currentSceneId: GameSceneId;
  private initialPosition: WorldPoint;
  private player?: Phaser.Physics.Arcade.Image;
  private playerShadow?: Phaser.GameObjects.Ellipse;
  private roleRing?: Phaser.GameObjects.Arc;
  private keys?: Record<'up' | 'down' | 'left' | 'right' | 'interact' | 'wave', Phaser.Input.Keyboard.Key>;
  private cursors?: Phaser.Types.Input.Keyboard.CursorKeys;
  private moveTarget: WorldPoint | null = null;
  private pendingInteraction: GameInteraction | null = null;
  private interactionObjects: Array<{ definition: GameInteraction; object: Phaser.GameObjects.GameObject }> = [];
  private inputEnabled = true;
  private lastDirection: 'left' | 'right' = 'right';
  private lastPositionEmit = 0;
  private lastFootstepIndex = -1;
  private nearbyLabel: string | null = null;
  private movementActive = false;
  private transitioning = false;
  private portalCooldownUntil = 0;
  private reducedMotion = false;
  private playerBaseScaleX = 1;
  private playerBaseScaleY = 1;
  private remotePlayers = new Map<string, RemoteVisual>();

  constructor(
    sceneId: GameSceneId,
    position: WorldPoint,
    avatarPreset: AvatarPresetId,
    displayName: string,
    callbacks: GameRuntimeCallbacks,
  ) {
    super({ key: 'research-world' });
    this.currentSceneId = sceneId;
    this.initialPosition = position;
    this.avatarPreset = avatarPreset;
    this.displayName = displayName;
    this.callbacks = callbacks;
  }

  preload(): void {
    for (const preset of avatarPresets) {
      this.load.image(`avatar-${preset.id}`, preset.asset);
    }
    this.queueSceneAssets(this.currentSceneId);
  }

  create(): void {
    this.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.physics.world.setBounds(0, 0, WORLD_WIDTH, WORLD_HEIGHT);
    this.cameras.main.setBounds(0, 0, WORLD_WIDTH, WORLD_HEIGHT);
    this.createInput();
    this.renderScene(this.currentSceneId, this.initialPosition);
    this.scale.on('resize', this.updateCameraZoom, this);
    this.updateCameraZoom();
  }

  update(time: number): void {
    const player = this.player;
    if (!player) return;
    this.updatePlayerMovement(time, player);
    this.updateRemotePlayers(time);
    this.updateNearbyInteraction(player, time);
    this.clampCameraToWorld();
  }

  changeScene(sceneId: GameSceneId, position?: WorldPoint): void {
    if (!this.sys.isActive()) {
      this.currentSceneId = sceneId;
      this.initialPosition = position ?? gameScenes[sceneId].spawn;
      return;
    }
    this.loadAndRenderScene(sceneId, position ?? gameScenes[sceneId].spawn);
  }

  setGameInputEnabled(enabled: boolean): void {
    this.inputEnabled = enabled;
    if (!enabled) this.stopPlayer();
  }

  applyRemotePlayers(players: RemotePlayerState[]): void {
    const activeIds = new Set(
      players.filter((player) => player.sceneId === this.currentSceneId).map((player) => player.clientId),
    );
    for (const [clientId, visual] of this.remotePlayers) {
      if (!activeIds.has(clientId)) {
        visual.sprite.destroy();
        visual.shadow.destroy();
        visual.label.destroy();
        this.remotePlayers.delete(clientId);
      }
    }
    for (const state of players) {
      if (state.sceneId !== this.currentSceneId) continue;
      const existing = this.remotePlayers.get(state.clientId);
      if (existing) {
        const preset = avatarPresetById[state.avatarPreset] ?? avatarPresetById['theory-blue'];
        existing.sprite.setTexture(`avatar-${preset.id}`);
        existing.target = { x: state.x, y: state.y };
        existing.sprite.setFlipX(state.direction === 'left');
        existing.label.setText(state.displayName);
        continue;
      }
      const preset = avatarPresetById[state.avatarPreset] ?? avatarPresetById['theory-blue'];
      const shadow = this.add.ellipse(state.x, state.y + 34, 44, 13, 0x101820, 0.2);
      const sprite = this.add
        .image(state.x, state.y, `avatar-${preset.id}`)
        .setDisplaySize(56, 84)
        .setAlpha(0.78)
        .setFlipX(state.direction === 'left');
      const label = this.add
        .text(state.x, state.y - 53, state.displayName, {
          fontFamily: 'sans-serif',
          fontSize: '12px',
          color: '#ffffff',
          backgroundColor: '#263238cc',
          padding: { x: 5, y: 3 },
        })
        .setOrigin(0.5);
      this.remotePlayers.set(state.clientId, {
        sprite,
        shadow,
        label,
        target: { x: state.x, y: state.y },
        baseScaleX: sprite.scaleX,
        baseScaleY: sprite.scaleY,
        walkPhase: Phaser.Math.FloatBetween(0, Math.PI * 2),
      });
    }
  }

  playWave(): void {
    const player = this.player;
    if (!player) return;
    this.tweens.add({
      targets: player,
      angle: { from: -4, to: 4 },
      yoyo: true,
      repeat: 3,
      duration: 100,
      onComplete: () => player.setAngle(0),
    });
  }

  playRemoteWave(clientId: string): void {
    const visual = this.remotePlayers.get(clientId);
    if (!visual) return;
    this.tweens.add({
      targets: visual.sprite,
      angle: { from: -4, to: 4 },
      yoyo: true,
      repeat: 3,
      duration: 100,
      onComplete: () => visual.sprite.setAngle(0),
    });
  }

  private renderScene(sceneId: GameSceneId, position: WorldPoint): void {
    this.currentSceneId = sceneId;
    this.tweens.killAll();
    this.children.removeAll(true);
    this.interactionObjects = [];
    this.remotePlayers.clear();
    this.moveTarget = null;
    this.pendingInteraction = null;
    this.nearbyLabel = null;
    this.transitioning = false;
    this.portalCooldownUntil = this.time.now + 800;
    this.lastFootstepIndex = -1;
    if (this.movementActive) {
      this.movementActive = false;
      this.callbacks.onMovementChange(false);
    }
    this.callbacks.onNearbyChange(null);

    const definition = gameScenes[sceneId];
    this.cameras.main.setBackgroundColor(definition.tint);
    this.add.rectangle(0, 0, WORLD_WIDTH, WORLD_HEIGHT, definition.tint).setOrigin(0);
    const background = this.add
      .image(WORLD_WIDTH / 2, WORLD_HEIGHT / 2, `scene-${sceneId}`)
      .setDisplaySize(WORLD_WIDTH, WORLD_HEIGHT)
      .setAlpha(sceneId === 'campus' ? 1 : 0.66);
    background.setDepth(0);
    if (sceneId !== 'campus') {
      this.add
        .rectangle(0, 0, WORLD_WIDTH, WORLD_HEIGHT, definition.tint, 0.14)
        .setOrigin(0)
        .setDepth(1);
    }

    this.createAmbientAnimation(sceneId);
    for (const interaction of definition.interactions) this.createInteraction(interaction);
    this.createPlayer(position);
    this.cameras.main.startFollow(this.player!, true, 0.12, 0.12);
    this.cameras.main.centerOn(position.x, position.y);
    this.clampCameraToWorld();
    this.cameras.main.fadeIn(260, 245, 247, 244);
    this.callbacks.onSceneChange(sceneId, position);
  }

  private createInput(): void {
    const keyboard = this.input.keyboard;
    if (!keyboard) return;
    this.keys = {
      up: keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.W),
      down: keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.S),
      left: keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.A),
      right: keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.D),
      interact: keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.E),
      wave: keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.F),
    };
    this.cursors = keyboard.createCursorKeys();
    this.input.on('pointerdown', (pointer: Phaser.Input.Pointer) => {
      if (!this.inputEnabled) return;
      this.pendingInteraction = null;
      this.moveTarget = {
        x: Phaser.Math.Clamp(pointer.worldX, 55, WORLD_WIDTH - 55),
        y: Phaser.Math.Clamp(pointer.worldY, 110, WORLD_HEIGHT - 45),
      };
    });
  }

  private createPlayer(position: WorldPoint): void {
    const preset = avatarPresetById[this.avatarPreset];
    this.playerShadow = this.add.ellipse(position.x, position.y + 35, 48, 14, 0x101820, 0.24).setDepth(400);
    this.roleRing = this.add
      .circle(position.x, position.y + 35, 28, preset.color, 0.17)
      .setStrokeStyle(2, preset.color, 0.92)
      .setDepth(401);
    this.player = this.physics.add
      .image(position.x, position.y, `avatar-${preset.id}`)
      .setDisplaySize(62, 93)
      .setDepth(430)
      .setCollideWorldBounds(true);
    this.playerBaseScaleX = this.player.scaleX;
    this.playerBaseScaleY = this.player.scaleY;
    this.player.body!.setSize(26, 20).setOffset(18, 71);
    this.add
      .text(position.x, position.y - 61, this.displayName, {
        fontFamily: 'sans-serif',
        fontSize: '12px',
        color: '#ffffff',
        backgroundColor: '#1f2937d9',
        padding: { x: 6, y: 3 },
      })
      .setName('player-label')
      .setOrigin(0.5)
      .setDepth(450);
  }

  private createInteraction(interaction: GameInteraction): void {
    const { x, y } = interaction.position;
    const color = interaction.kind === 'portal' ? 0x1565c0 : interaction.kind === 'agent' ? 0x00897b : 0xf9a825;
    const radius = interaction.kind === 'portal' ? 38 : 32;
    const glow = this.add
      .circle(x, y, radius, color, 0.16)
      .setStrokeStyle(3, color, 0.88)
      .setDepth(y + 8);
    this.tweens.add({ targets: glow, alpha: { from: 0.55, to: 1 }, duration: 1100, yoyo: true, repeat: -1 });

    if (interaction.kind === 'agent') {
      const agent = agents[interaction.agentId];
      const agentImage = this.add
        .image(x, y - 42, `agent-${agent.id}`)
        .setDisplaySize(72, 108)
        .setDepth(y + 10);
      if (!this.reducedMotion) {
        this.tweens.add({
          targets: agentImage,
          y: y - 45,
          scaleY: agentImage.scaleY * 1.012,
          duration: 1700 + Phaser.Math.Between(0, 350),
          ease: 'Sine.easeInOut',
          yoyo: true,
          repeat: -1,
        });
      }
    } else {
      this.add
        .text(x, y - 3, interaction.kind === 'portal' ? '↗' : '▣', {
          fontFamily: 'sans-serif',
          fontSize: interaction.kind === 'portal' ? '25px' : '22px',
          color: '#ffffff',
        })
        .setOrigin(0.5)
        .setDepth(y + 10);
    }
    const label = this.add
      .text(x, y + radius + 10, interaction.label, {
        fontFamily: 'sans-serif',
        fontSize: '13px',
        color: '#263238',
        backgroundColor: '#fffdf6e8',
        padding: { x: 8, y: 5 },
      })
      .setOrigin(0.5)
      .setDepth(y + 12);

    const hitArea = this.add.zone(x, y, Math.max(100, label.width + 18), 120).setInteractive({ useHandCursor: true });
    hitArea.on('pointerdown', (_pointer: Phaser.Input.Pointer, _x: number, _y: number, event: Phaser.Types.Input.EventData) => {
      event.stopPropagation();
      if (!this.inputEnabled) return;
      this.pendingInteraction = interaction;
      this.moveTarget = { x, y: y + 55 };
    });
    this.interactionObjects.push({ definition: interaction, object: hitArea });
  }

  private updatePlayerMovement(time: number, player: Phaser.Physics.Arcade.Image): void {
    const body = player.body as Phaser.Physics.Arcade.Body;
    if (!this.inputEnabled || this.transitioning) {
      body.setVelocity(0);
      this.animateLocalPlayer(time, false);
      this.setMovementActive(false);
      return;
    }
    const horizontal = Number(this.keys?.right.isDown || this.cursors?.right.isDown) - Number(this.keys?.left.isDown || this.cursors?.left.isDown);
    const vertical = Number(this.keys?.down.isDown || this.cursors?.down.isDown) - Number(this.keys?.up.isDown || this.cursors?.up.isDown);

    let vx = horizontal;
    let vy = vertical;
    if (horizontal || vertical) {
      this.moveTarget = null;
      this.pendingInteraction = null;
    } else if (this.moveTarget) {
      const distance = Phaser.Math.Distance.Between(player.x, player.y, this.moveTarget.x, this.moveTarget.y);
      if (distance < 8) {
        this.stopPlayer();
        this.maybeTriggerPendingInteraction(player);
      } else {
        const angle = Phaser.Math.Angle.Between(player.x, player.y, this.moveTarget.x, this.moveTarget.y);
        vx = Math.cos(angle);
        vy = Math.sin(angle);
      }
    }

    const moving = Boolean(vx || vy);
    if (moving) {
      const length = Math.hypot(vx, vy) || 1;
      body.setVelocity((vx / length) * MOVE_SPEED, (vy / length) * MOVE_SPEED);
      if (Math.abs(vx) > 0.08) {
        this.lastDirection = vx < 0 ? 'left' : 'right';
        player.setFlipX(this.lastDirection === 'left');
      }
    } else {
      body.setVelocity(0);
    }
    this.animateLocalPlayer(time, moving);
    this.setMovementActive(moving);

    const label = this.children.getByName('player-label') as Phaser.GameObjects.Text | null;
    this.playerShadow?.setPosition(player.x, player.y + 35).setDepth(player.y - 2);
    this.roleRing?.setPosition(player.x, player.y + 35).setDepth(player.y - 1);
    player.setDepth(player.y + 2);
    label?.setPosition(player.x, player.y - 61).setDepth(player.y + 4);

    if (this.keys && Phaser.Input.Keyboard.JustDown(this.keys.interact)) {
      this.triggerNearestInteraction(player);
    }
    if (this.keys && Phaser.Input.Keyboard.JustDown(this.keys.wave)) this.playWave();
    if (time - this.lastPositionEmit >= 110 && moving) {
      this.lastPositionEmit = time;
      const event: LocalPositionEvent = {
        sceneId: this.currentSceneId,
        x: player.x,
        y: player.y,
        direction: this.lastDirection,
      };
      this.callbacks.onPosition(event);
    }
  }

  private animateLocalPlayer(time: number, moving: boolean): void {
    const player = this.player;
    if (!player) return;
    if (this.reducedMotion) {
      player.setAngle(0).setOrigin(0.5).setScale(this.playerBaseScaleX, this.playerBaseScaleY);
      this.playerShadow?.setScale(1);
      this.roleRing?.setAlpha(0.82);
      return;
    }

    if (!moving) {
      const breath = Math.sin(time / 720) * 0.004;
      player
        .setAngle(0)
        .setOrigin(0.5)
        .setScale(this.playerBaseScaleX * (1 - breath), this.playerBaseScaleY * (1 + breath));
      this.playerShadow?.setScale(1, 1);
      this.roleRing?.setAlpha(0.72 + Math.sin(time / 650) * 0.08);
      return;
    }

    const phase = (time / WALK_STEP_MS) * Math.PI;
    const stride = Math.sin(phase);
    const lift = Math.abs(stride);
    const bob = lift * 3.4;
    player
      .setAngle(stride * 2.6)
      .setOrigin(0.5 + stride * 0.012, 0.5 + bob / 93)
      .setScale(
        this.playerBaseScaleX * (1 + lift * 0.014),
        this.playerBaseScaleY * (1 - lift * 0.012),
      );
    this.playerShadow?.setScale(1 - lift * 0.12, 1 - lift * 0.07);
    this.roleRing?.setAlpha(0.76 + lift * 0.18);

    const footstepIndex = Math.floor(time / WALK_STEP_MS);
    if (footstepIndex !== this.lastFootstepIndex) {
      this.lastFootstepIndex = footstepIndex;
      this.createFootstep(player.x, player.y + 39, footstepIndex % 2 === 0 ? -5 : 5);
    }
  }

  private createFootstep(x: number, y: number, offsetX: number): void {
    if (this.reducedMotion) return;
    const preset = avatarPresetById[this.avatarPreset];
    const mark = this.add
      .ellipse(x + offsetX, y, 10, 4, preset.color, 0.12)
      .setStrokeStyle(1, preset.color, 0.34)
      .setDepth(y - 6);
    this.tweens.add({
      targets: mark,
      alpha: 0,
      scaleX: 1.8,
      scaleY: 1.45,
      y: y + 2,
      duration: 430,
      ease: 'Sine.easeOut',
      onComplete: () => mark.destroy(),
    });
  }

  private setMovementActive(moving: boolean): void {
    if (this.movementActive === moving) return;
    this.movementActive = moving;
    this.callbacks.onMovementChange(moving);
  }

  private updateRemotePlayers(time: number): void {
    for (const visual of this.remotePlayers.values()) {
      const distance = Phaser.Math.Distance.Between(
        visual.sprite.x,
        visual.sprite.y,
        visual.target.x,
        visual.target.y,
      );
      const moving = distance > 1.2;
      visual.sprite.x = Phaser.Math.Linear(visual.sprite.x, visual.target.x, 0.22);
      visual.sprite.y = Phaser.Math.Linear(visual.sprite.y, visual.target.y, 0.22);
      if (!this.reducedMotion && moving) {
        visual.walkPhase += 0.28;
        const stride = Math.sin(visual.walkPhase);
        const lift = Math.abs(stride);
        visual.sprite
          .setAngle(stride * 2.1)
          .setOrigin(0.5 + stride * 0.01, 0.5 + (lift * 2.4) / 84)
          .setScale(
            visual.baseScaleX * (1 + lift * 0.012),
            visual.baseScaleY * (1 - lift * 0.01),
          );
        visual.shadow.setScale(1 - lift * 0.1, 1 - lift * 0.06);
      } else if (!this.reducedMotion) {
        const breath = Math.sin(time / 760 + visual.walkPhase) * 0.003;
        visual.sprite
          .setAngle(0)
          .setOrigin(0.5)
          .setScale(
            visual.baseScaleX * (1 - breath),
            visual.baseScaleY * (1 + breath),
          );
        visual.shadow.setScale(1);
      }
      visual.shadow.setPosition(visual.sprite.x, visual.sprite.y + 34).setDepth(visual.sprite.y - 4);
      visual.sprite.setDepth(visual.sprite.y);
      visual.label.setPosition(visual.sprite.x, visual.sprite.y - 53).setDepth(visual.sprite.y + 2);
    }
  }

  private updateNearbyInteraction(player: Phaser.Physics.Arcade.Image, time: number): void {
    let nearest: GameInteraction | null = null;
    let nearestDistance = Number.POSITIVE_INFINITY;
    for (const item of this.interactionObjects) {
      const distance = Phaser.Math.Distance.Between(
        player.x,
        player.y,
        item.definition.position.x,
        item.definition.position.y,
      );
      if (distance < nearestDistance) {
        nearest = item.definition;
        nearestDistance = distance;
      }
    }
    const nextLabel = nearest && nearestDistance <= INTERACTION_DISTANCE ? nearest.label : null;
    if (nextLabel !== this.nearbyLabel) {
      this.nearbyLabel = nextLabel;
      this.callbacks.onNearbyChange(nextLabel);
    }
    if (
      nearest?.kind === 'portal' &&
      nearestDistance <= AUTO_PORTAL_DISTANCE &&
      time >= this.portalCooldownUntil &&
      !this.transitioning
    ) {
      this.triggerInteraction(nearest);
    }
  }

  private triggerNearestInteraction(player: Phaser.Physics.Arcade.Image): void {
    const nearest = this.interactionObjects
      .map((item) => ({
        interaction: item.definition,
        distance: Phaser.Math.Distance.Between(
          player.x,
          player.y,
          item.definition.position.x,
          item.definition.position.y,
        ),
      }))
      .sort((left, right) => left.distance - right.distance)[0];
    if (nearest && nearest.distance <= INTERACTION_DISTANCE) this.triggerInteraction(nearest.interaction);
  }

  private maybeTriggerPendingInteraction(player: Phaser.Physics.Arcade.Image): void {
    const pending = this.pendingInteraction;
    if (!pending) return;
    const distance = Phaser.Math.Distance.Between(
      player.x,
      player.y,
      pending.position.x,
      pending.position.y,
    );
    if (distance <= INTERACTION_DISTANCE + 15) this.triggerInteraction(pending);
  }

  private triggerInteraction(interaction: GameInteraction): void {
    this.pendingInteraction = null;
    this.moveTarget = null;
    if (interaction.kind === 'portal') {
      if (this.transitioning) return;
      this.transitioning = true;
      this.stopPlayer();
      this.setMovementActive(false);
      const target = interaction.spawn ?? gameScenes[interaction.targetScene].spawn;
      this.cameras.main.fadeOut(180, 245, 247, 244, (_camera: Phaser.Cameras.Scene2D.Camera, progress: number) => {
        if (progress === 1) this.loadAndRenderScene(interaction.targetScene, target);
      });
      return;
    }
    this.stopPlayer();
    this.callbacks.onInteraction(interaction);
  }

  private stopPlayer(): void {
    this.moveTarget = null;
    const body = this.player?.body as Phaser.Physics.Arcade.Body | undefined;
    body?.setVelocity(0);
    this.animateLocalPlayer(this.time.now, false);
    this.setMovementActive(false);
  }

  private createAmbientAnimation(sceneId: GameSceneId): void {
    if (this.reducedMotion) return;
    if (sceneId === 'campus') {
      const waveSites: Array<[number, number, number, number]> = [
        [70, 120, 104, -12],
        [250, 54, 78, 8],
        [790, 24, 116, 0],
        [1360, 72, 96, -8],
        [1540, 205, 108, 82],
        [1545, 735, 112, 78],
        [1260, 868, 124, 2],
        [230, 858, 118, -4],
        [42, 665, 104, 86],
        [630, 478, 52, 68],
        [972, 548, 58, 102],
      ];
      waveSites.forEach(([x, y, width, angle], index) => {
        const wave = this.add
          .arc(x, y, width / 2, 205, 335, false, 0xffffff, 0)
          .setClosePath(false)
          .setStrokeStyle(1.5, 0xe8fbff, 0.5)
          .setAngle(angle)
          .setAlpha(0.06)
          .setScale(0.72, 0.17)
          .setDepth(3);
        this.tweens.add({
          targets: wave,
          alpha: 0.38,
          scaleX: 1.02,
          scaleY: 0.22,
          x: x + Math.cos(Phaser.Math.DegToRad(angle)) * 5,
          y: y + Math.sin(Phaser.Math.DegToRad(angle)) * 3,
          duration: 1750 + (index % 4) * 240,
          delay: (index % 3) * 180,
          ease: 'Sine.easeInOut',
          yoyo: true,
          repeat: -1,
        });
      });

      const treeSites: WorldPoint[] = [
        { x: 350, y: 318 },
        { x: 520, y: 320 },
        { x: 650, y: 350 },
        { x: 815, y: 310 },
        { x: 965, y: 348 },
        { x: 1100, y: 386 },
        { x: 1190, y: 610 },
        { x: 470, y: 700 },
        { x: 1350, y: 742 },
      ];
      treeSites.forEach((site, siteIndex) => {
        for (let leafIndex = 0; leafIndex < 3; leafIndex += 1) {
          const x = site.x + (leafIndex - 1) * 8;
          const y = site.y + (leafIndex % 2) * 5;
          const leaf = this.add
            .triangle(x, y, 0, 7, 4, 0, 8, 7, leafIndex % 2 ? 0x78985f : 0x5f854f, 0.62)
            .setAngle(-18 + leafIndex * 22)
            .setDepth(4);
          this.tweens.add({
            targets: leaf,
            x: x + (leafIndex - 1) * 3,
            y: y - 3 - leafIndex,
            angle: leaf.angle + (leafIndex % 2 ? 9 : -9),
            duration: 1700 + ((siteIndex + leafIndex) % 4) * 310,
            delay: leafIndex * 140,
            ease: 'Sine.easeInOut',
            yoyo: true,
            repeat: -1,
          });
        }
      });
      return;
    }

    const indicatorSites: Record<Exclude<GameSceneId, 'campus'>, WorldPoint[]> = {
      academy: [{ x: 565, y: 352 }, { x: 1010, y: 330 }, { x: 1180, y: 540 }],
      research: [{ x: 352, y: 405 }, { x: 1168, y: 540 }, { x: 1250, y: 574 }],
      compute: [{ x: 955, y: 420 }, { x: 1055, y: 442 }, { x: 1160, y: 465 }],
      paper: [{ x: 600, y: 410 }, { x: 980, y: 490 }, { x: 1120, y: 560 }],
      plaza: [{ x: 510, y: 430 }, { x: 850, y: 370 }, { x: 1160, y: 460 }],
    };
    const accent = gameScenes[sceneId].tint === 0xf8f2e6 ? 0xf2b84b : 0x63c6bf;
    indicatorSites[sceneId].forEach((site, index) => {
      const light = this.add.rectangle(site.x, site.y, 7, 4, accent, 0.34).setDepth(4);
      this.tweens.add({
        targets: light,
        alpha: 0.92,
        scaleX: 1.45,
        duration: 800 + index * 230,
        delay: index * 280,
        ease: 'Sine.easeInOut',
        yoyo: true,
        repeat: -1,
      });
    });

    for (let index = 0; index < 7; index += 1) {
      const x = 250 + ((index * 197) % 1120);
      const y = 260 + ((index * 83) % 390);
      const mote = this.add
        .triangle(x, y, 0, 4, 2, 0, 4, 4, 0xffffff, 0.11)
        .setDepth(3);
      this.tweens.add({
        targets: mote,
        x: x + 8 + (index % 3) * 4,
        y: y - 18 - (index % 2) * 8,
        alpha: 0.02,
        angle: 70 + index * 17,
        duration: 2600 + index * 260,
        delay: index * 190,
        ease: 'Sine.easeInOut',
        yoyo: true,
        repeat: -1,
      });
    }
  }

  private loadAndRenderScene(sceneId: GameSceneId, position: WorldPoint): void {
    if (this.sceneAssetsLoaded(sceneId)) {
      this.renderScene(sceneId, position);
      return;
    }
    this.stopPlayer();
    this.queueSceneAssets(sceneId);
    this.load.once(Phaser.Loader.Events.COMPLETE, () => this.renderScene(sceneId, position));
    this.load.start();
  }

  private queueSceneAssets(sceneId: GameSceneId): void {
    const definition = gameScenes[sceneId];
    const sceneKey = `scene-${sceneId}`;
    if (!this.textures.exists(sceneKey)) this.load.image(sceneKey, definition.background);
    for (const interaction of definition.interactions) {
      if (interaction.kind !== 'agent') continue;
      const key = `agent-${interaction.agentId}`;
      if (!this.textures.exists(key)) this.load.image(key, agents[interaction.agentId].asset);
    }
  }

  private sceneAssetsLoaded(sceneId: GameSceneId): boolean {
    const definition = gameScenes[sceneId];
    if (!this.textures.exists(`scene-${sceneId}`)) return false;
    return definition.interactions.every(
      (interaction) =>
        interaction.kind !== 'agent' || this.textures.exists(`agent-${interaction.agentId}`),
    );
  }

  private updateCameraZoom(): void {
    const width = this.scale.width;
    const height = this.scale.height;
    const coverZoom = Math.max(width / WORLD_WIDTH, height / WORLD_HEIGHT);
    this.cameras.main.setZoom(Phaser.Math.Clamp(coverZoom, 0.9, 1.25));
    this.clampCameraToWorld();
  }

  private clampCameraToWorld(): void {
    const camera = this.cameras.main;
    const visibleWidth = camera.width / camera.zoom;
    const visibleHeight = camera.height / camera.zoom;
    camera.scrollX = Phaser.Math.Clamp(camera.scrollX, 0, Math.max(0, WORLD_WIDTH - visibleWidth));
    camera.scrollY = Phaser.Math.Clamp(camera.scrollY, 0, Math.max(0, WORLD_HEIGHT - visibleHeight));
  }
}

export async function createResearchGame(
  parent: HTMLElement,
  options: {
    sceneId: GameSceneId;
    position: WorldPoint;
    avatarPreset: AvatarPresetId;
    displayName: string;
    callbacks: GameRuntimeCallbacks;
  },
): Promise<ResearchGameRuntime> {
  const scene = new ResearchWorldScene(
    options.sceneId,
    options.position,
    options.avatarPreset,
    options.displayName,
    options.callbacks,
  );
  const game = new Phaser.Game({
    type: Phaser.AUTO,
    parent,
    width: parent.clientWidth || 1280,
    height: parent.clientHeight || 720,
    backgroundColor: '#e9efec',
    transparent: false,
    render: { antialias: true, pixelArt: false, roundPixels: true },
    scale: { mode: Phaser.Scale.RESIZE, autoCenter: Phaser.Scale.CENTER_BOTH },
    physics: { default: 'arcade', arcade: { debug: false } },
    scene,
    input: { activePointers: 2 },
  });

  return {
    destroy: () => game.destroy(true),
    setScene: (sceneId, position) => scene.changeScene(sceneId, position),
    setInputEnabled: (enabled) => scene.setGameInputEnabled(enabled),
    setRemotePlayers: (players) => scene.applyRemotePlayers(players),
    waveRemote: (clientId) => scene.playRemoteWave(clientId),
    wave: () => scene.playWave(),
  };
}
