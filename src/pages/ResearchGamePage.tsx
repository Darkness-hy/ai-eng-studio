import { Component, lazy, Suspense, type ErrorInfo, type ReactNode } from 'react';
import { useParams, useSearchParams } from 'react-router-dom';
import { ResearchGameShell } from '../game/ResearchGameShell';
import { isGameSceneId } from '../game/worldContent';
import type { GameSceneId } from '../game/types';
import { isThreeSceneId } from '../game3d/world3d';

const ResearchGame3DShell = lazy(() =>
  import('../game3d/ResearchGame3DShell').then((module) => ({
    default: module.ResearchGame3DShell,
  })),
);

export function ResearchGamePage() {
  const { sceneId: routeSceneId } = useParams<{ sceneId?: string }>();
  const [searchParams] = useSearchParams();
  const sceneId = sceneFromRoute(routeSceneId);
  const force2D = searchParams.get('renderer') === '2d';
  const use3D = !force2D && isThreeSceneId(sceneId) && supportsWebGL();

  if (!use3D) return <ResearchGameShell />;
  return (
    <Game3DErrorBoundary key={sceneId} sceneId={sceneId}>
      <Suspense
        fallback={
          <div className="research-game-ui flex h-dvh min-h-[520px] items-center justify-center bg-[#6f7d82] text-[16px] text-[#e8eef0]">
            正在加载 3D 引擎……
          </div>
        }
      >
        <ResearchGame3DShell sceneId={sceneId} />
      </Suspense>
    </Game3DErrorBoundary>
  );
}

class Game3DErrorBoundary extends Component<
  { children: ReactNode; sceneId: GameSceneId },
  { error: Error | null }
> {
  state = { error: null } as { error: Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[research-game-3d] render failed', error, info);
  }

  render() {
    if (!this.state.error) return this.props.children;
    const base = import.meta.env.BASE_URL.replace(/\/$/, '');
    const href = `${base}${scenePath(this.props.sceneId)}?renderer=2d`;
    return (
      <div className="research-game-ui flex h-dvh min-h-[520px] items-center justify-center bg-canvas p-5">
        <div className="max-w-lg border-l-2 border-ink-red pl-4">
          <h1 className="font-serif text-[24px] font-semibold">3D 场景加载失败</h1>
          <p className="mt-2 text-[13px] leading-relaxed text-faint">
            当前浏览器或图形环境没有完成 3D 初始化。可以先回到保留的 2D 版本继续使用。
          </p>
          <a href={href} className="mt-4 inline-flex min-h-10 items-center rounded-md bg-ink px-4 text-[12.5px] text-paper">
            打开 2D 版本
          </a>
        </div>
      </div>
    );
  }
}

function supportsWebGL(): boolean {
  try {
    const canvas = document.createElement('canvas');
    return Boolean(canvas.getContext('webgl2') || canvas.getContext('webgl'));
  } catch {
    return false;
  }
}

function sceneFromRoute(value: string | undefined): GameSceneId {
  if (value && isGameSceneId(value)) return value;
  if (value === 'observatory' || value === 'lab') return 'research';
  return 'campus';
}

function scenePath(scene: GameSceneId): string {
  return scene === 'campus' ? '/research-game' : `/research-game/${scene}`;
}
