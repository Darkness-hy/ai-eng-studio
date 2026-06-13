import { useEffect, useRef, useState } from 'react';
import { ensureFigures, mountFigures } from '../lib/figures';

/**
 * Renders an upstream interactive lesson figure for a ```figure``` block.
 * The empty `.lesson-figure` div is hydrated by the vanilla scripts; React
 * owns only the host element (no children), so the two don't fight over DOM.
 */
export function LessonFigure({ name }: { name: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let live = true;
    ensureFigures()
      .then(() => {
        if (!live) return;
        mountFigures();
        // If the figure name isn't registered, the host stays empty — surface
        // a small caption rather than a blank box.
        requestAnimationFrame(() => {
          if (live && ref.current && !ref.current.dataset.lfMounted && ref.current.childElementCount === 0) {
            setFailed(true);
          }
        });
      })
      .catch(() => live && setFailed(true));
    return () => {
      live = false;
    };
  }, [name]);

  return (
    <div className="my-6">
      <div ref={ref} className="lesson-figure" data-figure={name} />
      {failed && (
        <p className="rounded-lg border border-hairline bg-bone px-4 py-3 font-mono text-[12px] text-faint">
          figure: {name}
        </p>
      )}
    </div>
  );
}
