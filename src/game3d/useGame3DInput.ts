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
  Object.assign(ref.current, createMovementState3D());
}

export function useGame3DInput(options: {
  movementEnabled: boolean;
  shortcutsEnabled: boolean;
  onInteract: () => void;
  onToggleMap: () => void;
  onToggleDossier?: () => void;
  onEscape: () => void;
}) {
  const { movementEnabled, shortcutsEnabled, onInteract, onToggleMap, onToggleDossier, onEscape } =
    options;
  const movementRef = useRef<MovementState3D>(createMovementState3D());

  useEffect(() => {
    if (!movementEnabled) clearMovement3D(movementRef);
  }, [movementEnabled]);

  useEffect(() => {
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
    };

    const onBlur = () => clearMovement3D(movementRef);
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('blur', onBlur);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('blur', onBlur);
    };
  }, [movementEnabled, onEscape, onInteract, onToggleDossier, onToggleMap, shortcutsEnabled]);

  return movementRef;
}

function isEditableTarget(target: EventTarget | null): boolean {
  return target instanceof HTMLElement &&
    (target.isContentEditable || target.matches('input, textarea, select'));
}
