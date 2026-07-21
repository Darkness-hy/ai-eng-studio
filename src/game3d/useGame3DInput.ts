import { useEffect, useRef, type MutableRefObject } from 'react';
import type { MovementState3D } from './types3d';

export function createMovementState3D(): MovementState3D {
  return {
    forward: false,
    backward: false,
    leftward: false,
    rightward: false,
    run: false,
    jump: false,
    joystick: { x: 0, y: 0 },
  };
}

export function clearMovement3D(ref: MutableRefObject<MovementState3D>): void {
  ref.current.forward = false;
  ref.current.backward = false;
  ref.current.leftward = false;
  ref.current.rightward = false;
  ref.current.run = false;
  ref.current.jump = false;
  ref.current.joystick.x = 0;
  ref.current.joystick.y = 0;
}

/**
 * Keyboard + shortcut input for the 3D island.
 * WASD / arrows drive Ecctrl via movementRef (third- and first-person).
 * E interact · V person view · M map · J/Tab dossier · Esc close.
 */
export function useGame3DInput(options: {
  movementEnabled: boolean;
  shortcutsEnabled: boolean;
  onInteract: () => void;
  onToggleMap: () => void;
  onToggleDossier?: () => void;
  onTogglePersonView?: () => void;
  onEscape: () => void;
}) {
  const {
    movementEnabled,
    shortcutsEnabled,
    onInteract,
    onToggleMap,
    onToggleDossier,
    onTogglePersonView,
    onEscape,
  } = options;
  const movementRef = useRef<MovementState3D>(createMovementState3D());
  const keysRef = useRef({
    forward: false,
    backward: false,
    leftward: false,
    rightward: false,
    run: false,
    jump: false,
  });
  /** True while keyboard last wrote the joystick (so key-up can clear sticky walk). */
  const keyboardOwnsStickRef = useRef(false);

  const applyKeysToMovement = () => {
    const k = keysRef.current;
    const m = movementRef.current;

    m.forward = k.forward;
    m.backward = k.backward;
    m.leftward = k.leftward;
    m.rightward = k.rightward;
    m.run = k.run;
    m.jump = k.jump;

    let x = 0;
    let y = 0;
    if (k.leftward) x -= 1;
    if (k.rightward) x += 1;
    if (k.forward) y += 1;
    if (k.backward) y -= 1;
    const len = Math.hypot(x, y);

    if (len > 0) {
      // Keyboard directions always win over a stale stick value.
      m.joystick.x = x / len;
      m.joystick.y = y / len;
      keyboardOwnsStickRef.current = true;
      return;
    }

    // No keyboard direction held: clear stick only if keyboard owned it.
    // Leave mobile virtual stick alone (it zeros itself on pointer up).
    if (keyboardOwnsStickRef.current) {
      m.joystick.x = 0;
      m.joystick.y = 0;
      keyboardOwnsStickRef.current = false;
    }
  };

  useEffect(() => {
    if (!movementEnabled) {
      keysRef.current = {
        forward: false,
        backward: false,
        leftward: false,
        rightward: false,
        run: false,
        jump: false,
      };
      keyboardOwnsStickRef.current = false;
      clearMovement3D(movementRef);
    }
  }, [movementEnabled]);

  useEffect(() => {
    const setKey = (code: string, pressed: boolean): boolean => {
      const k = keysRef.current;
      switch (code) {
        case 'KeyW':
        case 'ArrowUp':
          k.forward = pressed;
          return true;
        case 'KeyS':
        case 'ArrowDown':
          k.backward = pressed;
          return true;
        case 'KeyA':
        case 'ArrowLeft':
          k.leftward = pressed;
          return true;
        case 'KeyD':
        case 'ArrowRight':
          k.rightward = pressed;
          return true;
        case 'ShiftLeft':
        case 'ShiftRight':
          k.run = pressed;
          return true;
        case 'Space':
          k.jump = pressed;
          return true;
        default:
          return false;
      }
    };

    const onKeyDown = (event: KeyboardEvent) => {
      if (isEditableTarget(event.target)) return;

      if (event.code === 'Escape' && shortcutsEnabled) {
        event.preventDefault();
        onEscape();
        return;
      }
      if (event.code === 'KeyM' && shortcutsEnabled && !event.repeat) {
        event.preventDefault();
        onToggleMap();
        return;
      }
      if (event.code === 'KeyV' && shortcutsEnabled && !event.repeat && onTogglePersonView) {
        event.preventDefault();
        onTogglePersonView();
        return;
      }
      if (
        (event.code === 'KeyJ' || event.code === 'Tab') &&
        shortcutsEnabled &&
        !event.repeat &&
        onToggleDossier
      ) {
        event.preventDefault();
        onToggleDossier();
        return;
      }
      if (event.code === 'KeyE' && shortcutsEnabled && !event.repeat) {
        event.preventDefault();
        onInteract();
        return;
      }

      if (!movementEnabled) return;
      if (event.repeat) {
        // Ignore OS key-repeat so we don't fight keyup edge cases.
        if (
          event.code === 'KeyW' ||
          event.code === 'KeyA' ||
          event.code === 'KeyS' ||
          event.code === 'KeyD' ||
          event.code.startsWith('Arrow') ||
          event.code === 'Space'
        ) {
          event.preventDefault();
        }
        return;
      }
      if (setKey(event.code, true)) {
        if (
          event.code.startsWith('Arrow') ||
          event.code === 'Space' ||
          event.code === 'KeyW' ||
          event.code === 'KeyA' ||
          event.code === 'KeyS' ||
          event.code === 'KeyD'
        ) {
          event.preventDefault();
        }
        applyKeysToMovement();
      }
    };

    const onKeyUp = (event: KeyboardEvent) => {
      // Always clear movement keys even if focus moved to a button — otherwise sticky walk.
      if (setKey(event.code, false)) {
        applyKeysToMovement();
      }
    };

    const onBlur = () => {
      keysRef.current = {
        forward: false,
        backward: false,
        leftward: false,
        rightward: false,
        run: false,
        jump: false,
      };
      keyboardOwnsStickRef.current = false;
      clearMovement3D(movementRef);
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onBlur);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onBlur);
      keysRef.current = {
        forward: false,
        backward: false,
        leftward: false,
        rightward: false,
        run: false,
        jump: false,
      };
      keyboardOwnsStickRef.current = false;
      clearMovement3D(movementRef);
    };
  }, [
    movementEnabled,
    onEscape,
    onInteract,
    onToggleDossier,
    onToggleMap,
    onTogglePersonView,
    shortcutsEnabled,
  ]);

  return movementRef;
}

function isEditableTarget(target: EventTarget | null): boolean {
  return (
    target instanceof HTMLElement &&
    (target.isContentEditable || target.matches('input, textarea, select, button'))
  );
}
