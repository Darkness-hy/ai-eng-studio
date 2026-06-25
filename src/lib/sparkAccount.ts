import { getSupabase } from './supabase';

export type SparkStatus = 'requested' | 'approved' | 'provisioning' | 'ready' | 'failed' | 'revoked';

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
