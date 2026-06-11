import { useState, type FormEvent } from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../lib/auth';
import { useLang } from '../lib/i18n';

export function LoginPage() {
  const { lang } = useLang();
  const zh = lang === 'zh';
  const { enabled, profile, signIn, signUp } = useAuth();
  const [mode, setMode] = useState<'login' | 'signup'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [nickname, setNickname] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (!enabled) {
    return (
      <div className="py-32 text-center text-[14px] text-faint">
        {zh ? '云同步尚未配置，当前进度保存在本机浏览器中。' : 'Cloud sync is not configured; progress is stored locally.'}
      </div>
    );
  }

  // Once the profile lands (after sign-in/up), enter the site.
  if (profile) return <Navigate to="/" replace />;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const err =
      mode === 'login'
        ? await signIn(email.trim(), password)
        : await signUp(email.trim(), password, nickname.trim() || email.split('@')[0]);
    setBusy(false);
    if (err) setError(translateAuthError(err, zh));
    // on success the auth listener sets profile and <Navigate> takes over
  };

  const field =
    'w-full rounded-md border border-hairline bg-paper px-4 py-2.5 text-[14.5px] outline-none transition-colors placeholder:text-faint focus:border-faint';

  return (
    <div className="mx-auto max-w-sm px-5 py-20">
      <h1 className="font-serif text-[32px] font-semibold tracking-tight">
        {mode === 'login' ? (zh ? '登录' : 'Sign in') : zh ? '创建账户' : 'Create account'}
      </h1>
      <p className="mt-2 text-[13.5px] leading-relaxed text-faint">
        {zh
          ? '同一账户在任意设备登录，学习进度与测验成绩自动同步。'
          : 'Sign in on any device and your progress follows you.'}
      </p>

      <form onSubmit={submit} className="mt-8 space-y-3">
        {mode === 'signup' && (
          <input
            value={nickname}
            onChange={(e) => setNickname(e.target.value)}
            placeholder={zh ? '昵称（显示用）' : 'Nickname'}
            className={field}
            maxLength={24}
          />
        )}
        <input
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder={zh ? '邮箱' : 'Email'}
          type="email"
          required
          className={field}
        />
        <input
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder={zh ? '密码（至少 6 位）' : 'Password (min 6 chars)'}
          type="password"
          required
          minLength={6}
          className={field}
        />
        {error && (
          <p className="rounded-md bg-pale-red px-4 py-2.5 text-[13px] text-ink-red">{error}</p>
        )}
        <button
          type="submit"
          disabled={busy}
          className="w-full rounded-md bg-ink px-4 py-2.5 text-[14px] text-white transition-colors hover:bg-ink/85 disabled:opacity-50"
        >
          {busy ? '…' : mode === 'login' ? (zh ? '登录' : 'Sign in') : zh ? '注册并登录' : 'Sign up'}
        </button>
      </form>

      <p className="mt-6 text-center text-[13px] text-faint">
        {mode === 'login' ? (
          <>
            {zh ? '还没有账户？' : 'No account yet?'}{' '}
            <button
              type="button"
              onClick={() => {
                setMode('signup');
                setError(null);
              }}
              className="text-ink underline decoration-hairline underline-offset-2"
            >
              {zh ? '创建一个' : 'Create one'}
            </button>
          </>
        ) : (
          <>
            {zh ? '已有账户？' : 'Have an account?'}{' '}
            <button
              type="button"
              onClick={() => {
                setMode('login');
                setError(null);
              }}
              className="text-ink underline decoration-hairline underline-offset-2"
            >
              {zh ? '直接登录' : 'Sign in'}
            </button>
          </>
        )}
      </p>
    </div>
  );
}

function translateAuthError(message: string, zh: boolean): string {
  if (!zh) return message;
  if (/invalid login credentials/i.test(message)) return '邮箱或密码不正确';
  if (/already registered/i.test(message)) return '该邮箱已注册，请直接登录';
  if (/password should be at least/i.test(message)) return '密码至少需要 6 位';
  if (/rate limit/i.test(message)) return '操作过于频繁，请稍后再试';
  if (/network|fetch/i.test(message)) return '网络错误，请检查网络后重试';
  return message;
}
