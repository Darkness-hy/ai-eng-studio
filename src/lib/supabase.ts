import { createClient, type SupabaseClient } from '@supabase/supabase-js';

const url = import.meta.env.VITE_SUPABASE_URL as string | undefined;
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined;

/** Cloud sync is optional: without env config the site stays fully local. */
export const cloudEnabled = Boolean(url && anonKey);

let client: SupabaseClient | null = null;

export function getSupabase(): SupabaseClient {
  if (!cloudEnabled) throw new Error('Supabase 未配置（缺少 VITE_SUPABASE_URL / VITE_SUPABASE_ANON_KEY）');
  if (!client) client = createClient(url!, anonKey!);
  return client;
}

export interface ProfileRow {
  id: string;
  email: string;
  display_name: string | null;
  role: 'student' | 'admin';
  created_at: string;
}

export interface ProgressRow {
  user_id: string;
  lesson_id: string;
  done: boolean;
  pre_score: number | null;
  pre_total: number | null;
  post_score: number | null;
  post_total: number | null;
  completed_at: string | null;
  updated_at: string;
}

export interface ActivityRow {
  user_id: string;
  day: string;
}
