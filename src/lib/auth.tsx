import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react';
import { cloudEnabled, getSupabase, type ProfileRow } from './supabase';
import { flushNow, initialSync, startSync, stopSync } from './sync';

interface AuthCtx {
  enabled: boolean;
  loading: boolean;
  profile: ProfileRow | null;
  signIn: (email: string, password: string) => Promise<string | null>;
  signUp: (email: string, password: string, displayName: string) => Promise<string | null>;
  signOut: () => Promise<void>;
}

const Ctx = createContext<AuthCtx>(null!);

async function fetchProfile(userId: string): Promise<ProfileRow | null> {
  const { data, error } = await getSupabase()
    .from('profiles')
    .select('*')
    .eq('id', userId)
    .maybeSingle();
  if (error) {
    console.warn('[auth] profile fetch failed', error);
    return null;
  }
  return data as ProfileRow | null;
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [profile, setProfile] = useState<ProfileRow | null>(null);
  const [loading, setLoading] = useState(cloudEnabled);

  useEffect(() => {
    if (!cloudEnabled) return;
    const supabase = getSupabase();
    let live = true;

    const adopt = async (userId: string | undefined) => {
      if (!userId) {
        await flushNow(); // push the tail before tearing down sync
        stopSync();
        if (live) {
          setProfile(null);
          setLoading(false);
        }
        return;
      }
      const p = await fetchProfile(userId);
      if (!live) return;
      setProfile(p);
      setLoading(false);
      try {
        await initialSync(userId);
        startSync(userId);
      } catch (err) {
        console.warn('[auth] initial sync failed', err);
        startSync(userId); // keep collecting local changes; flush retries later
      }
    };

    supabase.auth.getSession().then(({ data }) => void adopt(data.session?.user?.id));
    const { data: sub } = supabase.auth.onAuthStateChange((_event, session) => {
      void adopt(session?.user?.id);
    });
    return () => {
      live = false;
      sub.subscription.unsubscribe();
    };
  }, []);

  const signIn = useCallback(async (email: string, password: string) => {
    try {
      const { error } = await getSupabase().auth.signInWithPassword({ email, password });
      return error ? error.message : null;
    } catch (err) {
      return err instanceof Error ? err.message : String(err);
    }
  }, []);

  const signUp = useCallback(async (email: string, password: string, displayName: string) => {
    try {
      const { error } = await getSupabase().auth.signUp({
        email,
        password,
        options: { data: { display_name: displayName } },
      });
      return error ? error.message : null;
    } catch (err) {
      return err instanceof Error ? err.message : String(err);
    }
  }, []);

  const signOut = useCallback(async () => {
    await flushNow(); // push the tail before tearing down sync
    stopSync();
    await getSupabase().auth.signOut();
    setProfile(null);
  }, []);

  return (
    <Ctx.Provider value={{ enabled: cloudEnabled, loading, profile, signIn, signUp, signOut }}>
      {children}
    </Ctx.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useAuth() {
  return useContext(Ctx);
}
