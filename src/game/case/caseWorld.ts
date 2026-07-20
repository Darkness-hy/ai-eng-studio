import type { Game3DInteraction, Vec3 } from '../../game3d/types3d';
import { DEMO_POIS } from './demoCase';

/** World positions for case POIs (3D scenes). */
export const CASE_POI_POSITIONS: Record<string, Vec3> = {
  poi_poster_a: [2.2, 0, 1.5], // plaza
  poi_repro_log: [-3.4, 0, 2.2], // compute
  poi_split_doc: [-5.2, 0, -3.1], // research
  poi_bench: [3.6, 0, 2.0], // compute
  // mentor uses existing academy-mentor agent position
};

/** Extra interactions injected into each 3D scene for the active demo case. */
export function caseInteractionsForScene(sceneId: string): Game3DInteraction[] {
  return DEMO_POIS.filter((poi) => poi.sceneId === sceneId && poi.id !== 'poi_mentor').map((poi) => {
    const position = CASE_POI_POSITIONS[poi.id] ?? [0, 0, 0];
    return {
      id: `case-${poi.id}`,
      kind: 'landmark' as const,
      label: `检视案件线索：${poi.label.zh}`,
      title: poi.label.zh,
      note: poi.visitHint.zh,
      position,
      radius: 2.5,
      casePoiId: poi.id,
    };
  });
}

export function isCaseMentorAgent(agentId: string): boolean {
  return agentId === 'research-mentor';
}
