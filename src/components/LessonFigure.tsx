import { useEffect, useRef, useState } from 'react';
import { ensureFigures, mountFigures } from '../lib/figures';

/**
 * Renders an upstream interactive lesson figure for a ```figure``` block.
 * The empty `.lesson-figure` div is hydrated by the vanilla scripts; React
 * owns only the host element (no children), so the two don't fight over DOM.
 *
 * The host is only mounted once it nears the viewport: an empty div hydrating
 * into a full-height SVG above the fold would shove the page and make scrolling
 * jump. Deferring keeps that growth at/below the fold, and `mountLessonFigures`
 * (which scans the whole document) then only hydrates the hosts already in view.
 */
export function LessonFigure({ name }: { name: string }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const hostRef = useRef<HTMLDivElement>(null);
  const [near, setNear] = useState(() => typeof IntersectionObserver === 'undefined');
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (near) return;
    const el = wrapRef.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setNear(true);
          io.disconnect();
        }
      },
      { rootMargin: '0px 0px 800px 0px' },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [near]);

  useEffect(() => {
    if (!near) return;
    let live = true;
    ensureFigures()
      .then(() => {
        if (!live) return;
        mountFigures();
        // If the figure name isn't registered, the host stays empty — surface
        // a small caption rather than a blank box.
        requestAnimationFrame(() => {
          if (live && hostRef.current && !hostRef.current.dataset.lfMounted && hostRef.current.childElementCount === 0) {
            setFailed(true);
          }
        });
      })
      .catch(() => live && setFailed(true));
    return () => {
      live = false;
    };
  }, [name, near]);

  return (
    <div ref={wrapRef} className="my-6">
      {near ? (
        <>
          <div ref={hostRef} className="lesson-figure" data-figure={name} />
          {failed && (
            <p className="rounded-lg border border-hairline bg-bone px-4 py-3 font-mono text-[12px] text-faint">
              figure: {name}
            </p>
          )}
        </>
      ) : (
        <div className="h-40 animate-pulse rounded-lg border border-hairline bg-bone" />
      )}
    </div>
  );
}
