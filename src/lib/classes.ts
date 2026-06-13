import { getSupabase, type ProfileRow } from './supabase';

export interface ClassRow {
  id: string;
  name: string;
  owner_id: string;
  invite_code: string;
  created_at: string;
}

const CODE_ALPHABET = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789'; // no easily-confused chars

function randomCode(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(6));
  let s = '';
  for (const b of bytes) s += CODE_ALPHABET[b % CODE_ALPHABET.length];
  return s;
}

export async function createClass(name: string, ownerId: string): Promise<ClassRow> {
  const supabase = getSupabase();
  for (let i = 0; i < 5; i++) {
    const invite_code = randomCode();
    const { data, error } = await supabase
      .from('classes')
      .insert({ name, owner_id: ownerId, invite_code })
      .select()
      .single();
    if (!error) return data as ClassRow;
    if (!/duplicate|unique/i.test(error.message)) throw error;
  }
  throw new Error('could not generate a unique invite code');
}

export async function joinClass(code: string): Promise<string> {
  const { data, error } = await getSupabase().rpc('join_class', { code: code.trim().toUpperCase() });
  if (error) throw error;
  return data as string;
}

export async function leaveClass(classId: string, userId: string): Promise<void> {
  const { error } = await getSupabase()
    .from('class_members')
    .delete()
    .eq('class_id', classId)
    .eq('user_id', userId);
  if (error) throw error;
}

export async function deleteClass(classId: string): Promise<void> {
  const { error } = await getSupabase().from('classes').delete().eq('id', classId);
  if (error) throw error;
}

export async function myClasses(userId: string): Promise<{ owned: ClassRow[]; joined: ClassRow[] }> {
  const supabase = getSupabase();
  const [{ data: owned }, { data: memberships }] = await Promise.all([
    supabase.from('classes').select('*').eq('owner_id', userId).order('created_at'),
    supabase.from('class_members').select('class_id').eq('user_id', userId),
  ]);
  const ids = (memberships ?? []).map((m) => (m as { class_id: string }).class_id);
  let joined: ClassRow[] = [];
  if (ids.length) {
    const { data } = await supabase.from('classes').select('*').in('id', ids);
    joined = ((data ?? []) as ClassRow[]).filter((c) => c.owner_id !== userId);
  }
  return { owned: (owned ?? []) as ClassRow[], joined };
}

export async function getClass(classId: string): Promise<ClassRow | null> {
  const { data } = await getSupabase().from('classes').select('*').eq('id', classId).maybeSingle();
  return (data as ClassRow | null) ?? null;
}

export async function classMembers(classId: string): Promise<ProfileRow[]> {
  const supabase = getSupabase();
  const { data: mem } = await supabase.from('class_members').select('user_id').eq('class_id', classId);
  const ids = (mem ?? []).map((m) => (m as { user_id: string }).user_id);
  if (!ids.length) return [];
  const { data } = await supabase.from('profiles').select('*').in('id', ids).order('email');
  return (data ?? []) as ProfileRow[];
}
