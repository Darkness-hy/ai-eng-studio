import { getSupabase } from './supabase';

export interface Assignment {
  id: string;
  class_id: string;
  title: string;
  lesson_ids: string[];
  due_date: string | null;
  created_at: string;
}

export async function createAssignment(
  classId: string,
  title: string,
  lessonIds: string[],
  dueDate: string | null,
): Promise<void> {
  const { error } = await getSupabase()
    .from('assignments')
    .insert({ class_id: classId, title, lesson_ids: lessonIds, due_date: dueDate });
  if (error) throw error;
}

export async function listAssignments(classId: string): Promise<Assignment[]> {
  const { data } = await getSupabase()
    .from('assignments')
    .select('*')
    .eq('class_id', classId)
    .order('created_at', { ascending: false });
  return (data ?? []) as Assignment[];
}

export async function deleteAssignment(id: string): Promise<void> {
  const { error } = await getSupabase().from('assignments').delete().eq('id', id);
  if (error) throw error;
}
