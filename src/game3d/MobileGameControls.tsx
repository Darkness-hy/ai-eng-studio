import { DoorOpen, Map, MessageCircle, MonitorCog } from 'lucide-react';
import {
  useCallback,
  useRef,
  type MutableRefObject,
  type PointerEvent as ReactPointerEvent,
} from 'react';
import type { Game3DInteraction, MovementState3D } from './types3d';

const JOYSTICK_RADIUS = 42;

export function MobileGameControls({
  movementRef,
  nearby,
  mapOpen,
  disabled,
  onInteract,
  onToggleMap,
  onManualMovement,
}: {
  movementRef: MutableRefObject<MovementState3D>;
  nearby: Game3DInteraction | null;
  mapOpen: boolean;
  disabled: boolean;
  onInteract: () => void;
  onToggleMap: () => void;
  onManualMovement: () => void;
}) {
  const originRef = useRef({ x: 0, y: 0 });
  const knobRef = useRef<HTMLDivElement | null>(null);

  const updateJoystick = useCallback(
    (clientX: number, clientY: number) => {
      const dx = clientX - originRef.current.x;
      const dy = clientY - originRef.current.y;
      const length = Math.hypot(dx, dy);
      const scale = length > JOYSTICK_RADIUS ? JOYSTICK_RADIUS / length : 1;
      const x = dx * scale;
      const y = dy * scale;
      movementRef.current.joystick.x = x / JOYSTICK_RADIUS;
      movementRef.current.joystick.y = -y / JOYSTICK_RADIUS;
      if (knobRef.current) {
        knobRef.current.style.transform = `translate3d(calc(-50% + ${x}px), calc(-50% + ${y}px), 0)`;
      }
    },
    [movementRef],
  );

  const resetJoystick = useCallback(() => {
    movementRef.current.joystick.x = 0;
    movementRef.current.joystick.y = 0;
    if (knobRef.current) knobRef.current.style.transform = 'translate3d(-50%, -50%, 0)';
  }, [movementRef]);

  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (disabled || mapOpen) return;
    onManualMovement();
    event.currentTarget.setPointerCapture(event.pointerId);
    const rect = event.currentTarget.getBoundingClientRect();
    originRef.current = { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
    updateJoystick(event.clientX, event.clientY);
  };

  const onPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
    updateJoystick(event.clientX, event.clientY);
  };

  const onPointerEnd = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    resetJoystick();
  };

  const InteractionIcon = nearby?.kind === 'portal'
    ? DoorOpen
    : nearby?.kind === 'workspace'
      ? MonitorCog
      : MessageCircle;

  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-0 z-20 flex items-end justify-between p-4 pb-[max(1rem,env(safe-area-inset-bottom))] md:hidden">
      <div
        className={`pointer-events-auto relative h-28 w-28 touch-none rounded-full border border-white/45 bg-[#17252a]/24 backdrop-blur-[2px] ${mapOpen ? 'invisible' : disabled ? 'opacity-35' : ''}`}
        aria-label="移动摇杆"
        aria-hidden={mapOpen}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerEnd}
        onPointerCancel={onPointerEnd}
      >
        <div className="absolute inset-5 rounded-full border border-white/35" />
        <div
          ref={knobRef}
          className="absolute left-1/2 top-1/2 h-11 w-11 rounded-full border border-white/70 bg-paper/82 shadow-lift"
          style={{ transform: 'translate3d(-50%, -50%, 0)' }}
        />
      </div>

      <div className="pointer-events-auto flex items-end gap-3">
        <button
          type="button"
          onClick={onToggleMap}
          className="grid h-12 w-12 place-items-center rounded-full border border-white/60 bg-[#17252a]/78 text-white shadow-lift backdrop-blur-sm"
          aria-label={mapOpen ? '关闭世界地图' : '打开世界地图'}
          title={mapOpen ? '关闭世界地图' : '打开世界地图'}
        >
          <Map size={20} aria-hidden="true" />
        </button>
        {!mapOpen && (
          <button
            type="button"
            onClick={onInteract}
            disabled={!nearby || disabled}
            className="grid h-16 w-16 place-items-center rounded-full border border-white/70 bg-[#f7b23b]/92 text-[#17252a] shadow-lift disabled:bg-[#17252a]/35 disabled:text-white/45"
            aria-label={nearby?.label ?? '附近没有可交互对象'}
            title={nearby?.label ?? '附近没有可交互对象'}
          >
            <InteractionIcon size={25} aria-hidden="true" />
          </button>
        )}
      </div>
    </div>
  );
}
