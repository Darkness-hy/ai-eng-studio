import { avatarPresets } from './worldContent';
import type { AvatarPresetId } from './types';

export function AvatarSetup({
  selected,
  displayName,
  confirmLabel = '进入科研公社',
  onSelect,
  onConfirm,
}: {
  selected: AvatarPresetId;
  displayName: string;
  confirmLabel?: string;
  onSelect: (preset: AvatarPresetId) => void;
  onConfirm: () => void;
}) {
  return (
    <div className="research-game-ui absolute inset-0 z-50 overflow-y-auto bg-canvas/96 px-4 py-8 backdrop-blur-sm">
      <div className="mx-auto max-w-5xl">
        <div className="border-b border-hairline pb-5">
          <p className="font-mono text-[10px] tracking-[0.18em] text-faint">PLAYER IDENTITY</p>
          <h1 className="mt-2 font-serif text-[30px] font-semibold sm:text-[38px]">选择你的科研角色</h1>
          <p className="mt-2 max-w-2xl text-[13.5px] leading-relaxed text-faint">
            {displayName} 将以这个身份进入科研公社。身份只影响外观标记和社交展示，不限制课程或研究方向。
          </p>
        </div>

        <div className="grid gap-px bg-hairline sm:grid-cols-2 lg:grid-cols-3">
          {avatarPresets.map((preset) => {
            const active = selected === preset.id;
            return (
              <button
                key={preset.id}
                type="button"
                onClick={() => onSelect(preset.id)}
                className={`group grid min-h-[290px] grid-cols-[128px_minmax(0,1fr)] items-center gap-5 bg-canvas px-5 py-5 text-left transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-ink-blue sm:grid-cols-[144px_minmax(0,1fr)] ${
                  active ? 'bg-paper' : 'hover:bg-bone'
                }`}
                aria-pressed={active}
              >
                <div
                  className="relative h-60 w-32 overflow-hidden rounded-md border sm:h-64 sm:w-36"
                  style={{ borderColor: active ? preset.cssColor : 'var(--color-hairline)' }}
                >
                  <div
                    className="absolute inset-x-0 bottom-0 h-20 opacity-20"
                    style={{ backgroundColor: preset.cssColor }}
                  />
                  <img
                    src={preset.asset}
                    alt={`${preset.name}角色`}
                    className="absolute inset-0 h-full w-full object-contain object-bottom"
                    draggable={false}
                  />
                </div>
                <div className="min-w-0">
                  <span
                    className="block h-3 w-3 rounded-full"
                    style={{ backgroundColor: preset.cssColor }}
                  />
                  <h2 className="mt-4 font-serif text-[20px] font-semibold">{preset.name}</h2>
                  <p className="mt-2 text-[12.5px] leading-relaxed text-faint">{preset.field}</p>
                  <p className="mt-4 font-mono text-[9.5px] tracking-[0.12em] text-faint">
                    {active ? 'SELECTED' : 'CHOOSE ROLE'}
                  </p>
                </div>
              </button>
            );
          })}
        </div>

        <div className="mt-6 flex justify-end border-t border-hairline pt-5">
          <button
            type="button"
            onClick={onConfirm}
            className="min-h-11 w-full rounded-md bg-ink px-6 py-2 text-[13.5px] font-medium text-paper transition-colors hover:bg-[#303942] sm:w-auto"
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
