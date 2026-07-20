/** Research Island Case Detective v0 — logical types (see docs/research-game-case-detective-v0.md). */

export type CaseId = string;

export type CaseStatus =
  | 'briefed'
  | 'investigating'
  | 'ready_for_review'
  | 'revision'
  | 'closed';

export type EvidenceKind =
  | 'citation'
  | 'testimony'
  | 'observation'
  | 'log'
  | 'experiment_result'
  | 'note';

export type EdgeType = 'supports' | 'contradicts' | 'refines' | 'instances';

export type ClaimState = 'active' | 'retracted' | 'accepted';

export type HypothesisMaturity = 'wild_guess' | 'plausible' | 'operationalized';

export type ReviewFindingCode =
  | 'no_baseline'
  | 'dangling_claim'
  | 'no_limitation'
  | 'overclaim'
  | 'no_repro'
  | 'no_question'
  | 'no_hypothesis'
  | 'no_experiment'
  | 'no_results';

export interface Localized {
  zh: string;
  en: string;
}

export interface EvidenceNode {
  id: string;
  kind: EvidenceKind;
  title: Localized;
  body: Localized;
  sourcePoiId?: string;
  sourceNpcId?: 'mentor' | 'reviewer' | null;
  discoveredAt: string;
  tags: string[];
}

export interface ClaimNode {
  id: string;
  text: string;
  state: ClaimState;
}

export interface CaseEdge {
  id: string;
  from: string;
  to: string;
  type: EdgeType;
}

export interface ReviewFinding {
  code: ReviewFindingCode;
  message: Localized;
  fixHint: Localized;
}

export interface ReviewRound {
  id: string;
  index: number;
  status: 'pending' | 'failed' | 'passed';
  findings: ReviewFinding[];
  submittedAt: string;
  resolvedAt?: string;
}

export interface ExperimentLink {
  id: string;
  label: string;
  summary: string;
  linkedAt: string;
  /** When true, result came from local mock (ResearchRun unavailable). */
  simulated: boolean;
}

export interface ResearchCase {
  id: CaseId;
  title: Localized;
  bulletin: Localized;
  status: CaseStatus;
  researchQuestion: string;
  hypothesis: {
    text: string;
    maturity: HypothesisMaturity;
  };
  resultsSummary: string;
  limitations: string[];
  claims: ClaimNode[];
  evidence: EvidenceNode[];
  edges: CaseEdge[];
  reviewRounds: ReviewRound[];
  experimentRuns: ExperimentLink[];
  /** Mentor dialogue / soft-gate flags, e.g. knows_leak */
  flags: string[];
  completeness: number;
  createdAt: string;
  updatedAt: string;
}

export interface CasePoi {
  id: string;
  caseId: CaseId;
  sceneId: string;
  label: Localized;
  attractor: string;
  grantsEvidenceIds: string[];
  requires?: { evidenceIds?: string[]; flags?: string[] };
  inspectOnce: boolean;
  /** Desk-mode blurb when player "visits" without 3D. */
  visitHint: Localized;
}
