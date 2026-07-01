import { getSupabase } from './supabase';

export type SparkStatus = 'requested' | 'approved' | 'provisioning' | 'ready' | 'failed' | 'revoked';

// The "Spark 使用班级" id (invite code FRB2XC). Not secret; must match the agent's
// SPARK_CLASS_ID. Used to tell the tutor whether the learner is eligible.
export const SPARK_CLASS_ID =
  (import.meta.env.VITE_SPARK_CLASS_ID as string | undefined) ?? '913b4815-41e9-40f2-8e9c-d8459c750a05';
export const SPARK_CLASS_INVITE_CODE =
  (import.meta.env.VITE_SPARK_CLASS_INVITE_CODE as string | undefined) ?? 'FRB2XC';

interface SparkClassRow {
  id: string;
  name: string | null;
  invite_code: string | null;
}

const normalizeClassName = (name: string | null | undefined) => (name ?? '').replace(/\s+/g, '').toLowerCase();

function isSparkClassRow(c: SparkClassRow): boolean {
  return (
    c.id === SPARK_CLASS_ID ||
    c.invite_code?.toUpperCase() === SPARK_CLASS_INVITE_CODE.toUpperCase() ||
    normalizeClassName(c.name) === 'spark使用班级'
  );
}

/** Is the signed-in learner a member of the Spark class? (RLS lets them read their own rows.) */
export async function isInSparkClass(userId: string): Promise<boolean> {
  const supabase = getSupabase();
  const { data: memberships, error: membershipError } = await supabase
    .from('class_members')
    .select('class_id')
    .eq('user_id', userId);
  if (membershipError) throw membershipError;

  const classIds = ((memberships ?? []) as { class_id: string }[]).map((m) => m.class_id);
  if (classIds.includes(SPARK_CLASS_ID)) return true;
  if (!classIds.length) return false;

  const { data: classes, error: classesError } = await supabase.from('classes').select('id,name,invite_code').in('id', classIds);
  if (classesError) throw classesError;
  return ((classes ?? []) as SparkClassRow[]).some(isSparkClassRow);
}

export interface SparkAccountRow {
  user_id: string;
  status: SparkStatus;
  ssh_username: string | null;
  temp_password: string | null;
  host: string | null;
  ssh_port: number | null;
  requested_username: string | null;
  note: string | null;
  error: string | null;
  requested_at: string;
  approved_at: string | null;
  provisioned_at: string | null;
}

/** The signed-in learner's own request (or null). */
export async function getMySparkAccount(userId: string): Promise<SparkAccountRow | null> {
  const supabase = getSupabase();
  const { data, error } = await supabase
    .from('spark_accounts')
    .select('*')
    .eq('user_id', userId)
    .maybeSingle();
  if (error) throw error;
  return (data as SparkAccountRow) ?? null;
}

/** Normalize a learner-supplied pinyin name into a safe linux username, or null. */
export function normalizeUsername(raw?: string): string | null {
  const u = (raw ?? '').toLowerCase().replace(/[^a-z0-9]/g, '').slice(0, 32);
  return /^[a-z][a-z0-9]{1,31}$/.test(u) ? u : null;
}

export async function requestSparkAccount(userId: string, username?: string): Promise<SparkAccountRow> {
  const supabase = getSupabase();
  const { data, error } = await supabase
    .from('spark_accounts')
    .insert({ user_id: userId, requested_username: normalizeUsername(username) })
    .select()
    .single();
  if (error) throw error;
  return data as SparkAccountRow;
}

/** Withdraw an un-approved request (RLS only allows delete while status='requested'). */
export async function cancelSparkRequest(userId: string): Promise<void> {
  const supabase = getSupabase();
  const { error } = await supabase.from('spark_accounts').delete().eq('user_id', userId);
  if (error) throw error;
}

// ── admin ──────────────────────────────────────────────────────────────────
export interface SparkRequestRow extends SparkAccountRow {
  profiles?: { email: string | null; display_name: string | null } | null;
}

export async function listSparkRequests(): Promise<SparkRequestRow[]> {
  const supabase = getSupabase();
  const { data, error } = await supabase
    .from('spark_accounts')
    .select('*, profiles(email, display_name)')
    .order('requested_at', { ascending: false });
  if (error) throw error;
  return (data as SparkRequestRow[]) ?? [];
}

export async function approveSparkAccount(userId: string, adminId: string): Promise<void> {
  const supabase = getSupabase();
  const { error } = await supabase
    .from('spark_accounts')
    .update({ status: 'approved', approved_by: adminId, approved_at: new Date().toISOString() })
    .eq('user_id', userId)
    .eq('status', 'requested');
  if (error) throw error;
}
