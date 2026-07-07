import { useEffect, useState } from 'react';

/** Tutor mascot states, derived from the widget's live status in TutorWidget. */
export type AvatarMode =
  | 'sleeping' // collapsed / dozing
  | 'greeting' // just opened, no conversation yet
  | 'listening' // user is composing a question
  | 'thinking' // request sent, waiting for first token
  | 'talking' // streaming the answer
  | 'happy' // just finished answering
  | 'confused' // error
  | 'idle'; // open, conversation exists, nothing happening

export type AvatarAvailability = 'unknown' | 'checking' | 'online' | 'offline';

/**
 * Phase 2 switch: flip to true once same-character expression variants exist at
 * public/tutor/avatar-<name>.png (see the generation prompts in the handoff plan).
 * Until then every mode reuses the neutral head-shoulders cutout and emotion is
 * carried entirely by motion + the overlay layer.
 */
const USE_EXPRESSIONS = true;

/** mode -> texture file (without the avatar- prefix / .png suffix). */
const MODE_TEX: Record<AvatarMode, string> = {
  sleeping: 'sleeping',
  greeting: 'greeting',
  listening: 'neutral',
  thinking: 'thinking',
  talking: 'talking',
  happy: 'happy',
  confused: 'confused',
  idle: 'neutral',
};

const ALL_TEX = ['neutral', 'blink', 'greeting', 'thinking', 'talking', 'happy', 'confused', 'sleeping'];

function url(name: string): string {
  return `${import.meta.env.BASE_URL}tutor/avatar-${name}.png`;
}

/** Detect prefers-reduced-motion without setting state in an effect body. */
function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () => typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches,
  );
  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const on = () => setReduced(mq.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, []);
  return !!reduced;
}

const HEART = 'M8 14s-5-3.3-5-7a3 3 0 0 1 5-2 3 3 0 0 1 5 2c0 3.7-5 7-5 7z';
const STAR = 'M8 0l1.6 6.4L16 8l-6.4 1.6L8 16l-1.6-6.4L0 8l6.4-1.6z';

// Deterministic burst layout (no Math.random — stable across renders).
const PARTICLES = [
  { px: '-150%', delay: 0, type: 'heart', color: 'var(--color-ink-red)' },
  { px: '120%', delay: 90, type: 'star', color: 'var(--color-ink-yellow)' },
  { px: '-60%', delay: 170, type: 'star', color: 'var(--color-ink-blue)' },
  { px: '70%', delay: 240, type: 'heart', color: 'var(--color-ink-red)' },
  { px: '-110%', delay: 320, type: 'star', color: 'var(--color-ink-green)' },
];

export function TutorAvatar({
  mode,
  availability,
  size = 36,
  reducedMotion: forced,
}: {
  mode: AvatarMode;
  availability?: AvatarAvailability;
  size?: number;
  reducedMotion?: boolean;
}) {
  const prefersReduced = usePrefersReducedMotion();
  const reduced = forced ?? prefersReduced;
  const animClass = reduced ? '' : `tav-${mode}`;

  // Preload every expression once so mode changes never flash a blank frame.
  useEffect(() => {
    if (!USE_EXPRESSIONS) return;
    ALL_TEX.forEach((n) => {
      const im = new Image();
      im.src = url(n);
    });
  }, []);

  // Ambient blink — only while showing the calm neutral texture (idle/listening),
  // so we never swap a different pose mid-blink. Pure timers, no effect-body setState.
  const [blink, setBlink] = useState(false);
  const calm = mode === 'idle' || mode === 'listening';
  useEffect(() => {
    if (reduced || !calm) return;
    let alive = true;
    let close: ReturnType<typeof setTimeout>;
    let open: ReturnType<typeof setTimeout>;
    const loop = (delay: number) => {
      open = setTimeout(() => {
        if (!alive) return;
        setBlink(true);
        close = setTimeout(() => {
          setBlink(false);
          loop(2600 + (delay % 1800)); // vary cadence without Math.random
        }, 150);
      }, delay);
    };
    loop(1600);
    return () => {
      alive = false;
      clearTimeout(open);
      clearTimeout(close);
    };
  }, [calm, reduced]);

  const baseTex = USE_EXPRESSIONS ? MODE_TEX[mode] : 'neutral';
  const texName = blink && calm && baseTex === 'neutral' ? 'blink' : baseTex;
  const ring = mode === 'listening' || mode === 'talking';
  const ringColor = mode === 'listening' ? 'var(--color-pale-blue)' : 'var(--color-pale-green)';
  const statusSize = Math.max(8, Math.round(size * 0.22));
  const statusColor =
    availability === 'online'
      ? 'var(--color-ink-green)'
      : availability === 'offline'
        ? 'var(--color-ink-red)'
        : 'var(--color-faint)';

  return (
    <div className="relative shrink-0" style={{ width: size, height: size }} aria-hidden>
      {/* pulsing attention ring (behind the frame) */}
      {ring && (
        <span
          className="absolute inset-0 rounded-full"
          style={{
            boxShadow: `0 0 0 2px ${ringColor}`,
            animation: reduced ? 'none' : 'tav-ring 1.4s ease-in-out infinite',
          }}
        />
      )}

      {/* circular portrait frame */}
      <div className="absolute inset-0 overflow-hidden rounded-full border border-hairline bg-bone">
        <img
          src={url(texName)}
          alt=""
          draggable={false}
          className={`tav-img h-full w-full select-none object-cover ${animClass}`}
          style={{ objectPosition: '50% 50%' }}
        />
      </div>

      {availability && (
        <span
          className="absolute rounded-full border-2 border-canvas shadow-sm"
          style={{
            width: statusSize,
            height: statusSize,
            right: Math.max(0, Math.round(size * 0.02)),
            bottom: Math.max(0, Math.round(size * 0.02)),
            backgroundColor: statusColor,
          }}
        />
      )}

      {/* ── overlay layer (glyphs above the frame) ───────────────── */}
      {/* thinking: orbiting dots */}
      {mode === 'thinking' && (
        <span className="absolute -right-1 -top-1 flex gap-0.5">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="h-1 w-1 rounded-full bg-faint"
              style={{ animation: reduced ? 'none' : `tav-ring 1s ease-in-out ${i * 0.18}s infinite` }}
            />
          ))}
        </span>
      )}

      {/* sleeping: drifting z's */}
      {mode === 'sleeping' && !reduced && (
        <span className="pointer-events-none absolute -right-0.5 -top-1 font-mono text-faint" style={{ fontSize: size * 0.3 }}>
          <span className="absolute" style={{ animation: 'tav-zzz 2.8s ease-in-out infinite' }}>z</span>
          <span className="absolute" style={{ animation: 'tav-zzz 2.8s ease-in-out 1.4s infinite', fontSize: size * 0.22 }}>z</span>
        </span>
      )}

      {/* confused: question mark */}
      {mode === 'confused' && (
        <span
          className="absolute -right-1 -top-1 flex h-3.5 w-3.5 items-center justify-center rounded-full bg-pale-red font-mono font-bold text-ink-red"
          style={{ fontSize: size * 0.26 }}
        >
          ?
        </span>
      )}

      {/* happy: pastel heart/star burst */}
      {mode === 'happy' && !reduced && (
        <span className="pointer-events-none absolute inset-0">
          {PARTICLES.map((p, i) => (
            <svg
              key={i}
              viewBox="0 0 16 16"
              width={size * 0.26}
              height={size * 0.26}
              className="absolute left-1/2 top-0"
              style={{
                color: p.color,
                ['--px' as string]: p.px,
                animation: `tav-particle 1s ease-out ${p.delay}ms both`,
              }}
            >
              <path d={p.type === 'heart' ? HEART : STAR} fill="currentColor" />
            </svg>
          ))}
        </span>
      )}
    </div>
  );
}
