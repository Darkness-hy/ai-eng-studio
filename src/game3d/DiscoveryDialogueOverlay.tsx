import { ArrowRight, BookOpen, Check } from 'lucide-react';
import type { LandmarkInteraction3D } from './types3d';

export function DiscoveryDialogueOverlay({
  interaction,
  isNew,
  discoveredCount,
  total,
  onClose,
}: {
  interaction: LandmarkInteraction3D;
  isNew: boolean;
  discoveredCount: number;
  total: number;
  onClose: () => void;
}) {
  return (
    <div className="discovery-dialogue-layer absolute inset-0 z-40" role="dialog" aria-modal="true" aria-label={interaction.title}>
      <div className="npc-dialogue-vignette pointer-events-none absolute inset-0" />
      <section className="discovery-dialogue-box absolute inset-x-3 bottom-3 sm:inset-x-[12vw] sm:bottom-7 lg:inset-x-[20vw]">
        <div className="discovery-dialogue-sigil" aria-hidden="true">
          {isNew ? <BookOpen size={28} /> : <Check size={28} />}
        </div>
        <div className="min-w-0">
          <p className="discovery-dialogue-kicker">{isNew ? 'NEW FIELD NOTE' : 'FIELD NOTE RECALLED'}</p>
          <h2>{interaction.title}</h2>
          <p className="discovery-dialogue-note">{interaction.note}</p>
          <div className="discovery-dialogue-footer">
            <div className="discovery-dialogue-progress" aria-label={`科研手记进度 ${discoveredCount}/${total}`}>
              <span style={{ width: `${total > 0 ? discoveredCount / total * 100 : 0}%` }} />
            </div>
            <strong>{discoveredCount}/{total}</strong>
            <button type="button" onClick={onClose}>
              继续探索
              <ArrowRight size={16} aria-hidden="true" />
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
