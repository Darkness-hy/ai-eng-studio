import { useEffect, useState, type FormEvent } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../lib/auth';
import {
  createClass,
  deleteClass,
  joinClass,
  leaveClass,
  myClasses,
  type ClassRow,
} from '../lib/classes';
import { useLang } from '../lib/i18n';

export function ClassesPage() {
  const { lang } = useLang();
  const zh = lang === 'zh';
  const { enabled, loading, profile } = useAuth();
  const [data, setData] = useState<{ owned: ClassRow[]; joined: ClassRow[] } | null>(null);
  const [name, setName] = useState('');
  const [code, setCode] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);

  const reload = () => {
    if (profile) myClasses(profile.id).then(setData).catch(() => setData({ owned: [], joined: [] }));
  };
  useEffect(reload, [profile]);

  if (!enabled) {
    return <Notice text={zh ? '云同步未配置,班级功能不可用' : 'Cloud sync not configured'} />;
  }
  if (loading || !profile) return <Notice text={zh ? '加载中…' : 'Loading…'} />;

  const create = async (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await createClass(name.trim(), profile.id);
      setName('');
      reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
    setBusy(false);
  };

  const join = async (e: FormEvent) => {
    e.preventDefault();
    if (!code.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await joinClass(code);
      setCode('');
      reload();
    } catch {
      setError(zh ? '邀请码无效' : 'Invalid invite code');
    }
    setBusy(false);
  };

  const copy = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(text);
      setTimeout(() => setCopied(null), 1500);
    });
  };

  const field =
    'rounded-md border border-hairline bg-paper px-3 py-2 text-[14px] outline-none transition-colors placeholder:text-faint focus:border-faint';

  return (
    <div className="mx-auto max-w-3xl px-5 py-14">
      <h1 className="font-serif text-[38px] font-semibold tracking-tight">{zh ? '班级' : 'Classes'}</h1>
      <p className="mt-2 text-[14px] text-faint">
        {zh
          ? '创建你自己的班级,把邀请码发给同学;或用别人给你的邀请码加入班级。在班级详情页查看全班学习进度与答题情况。'
          : 'Create your own class and share its invite code, or join one with a code. Track the whole class on the class page.'}
      </p>

      <div className="mt-8 grid gap-3 sm:grid-cols-2">
        <form onSubmit={create} className="rounded-lg border border-hairline bg-paper p-5">
          <div className="font-mono text-[10.5px] uppercase tracking-[0.16em] text-faint">
            {zh ? '创建班级' : 'Create a class'}
          </div>
          <div className="mt-3 flex gap-2">
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={zh ? '班级名称' : 'Class name'}
              className={`${field} flex-1`}
              maxLength={40}
            />
            <button type="submit" disabled={busy} className="rounded-md bg-ink px-4 text-[13px] text-white hover:bg-ink/85 disabled:opacity-50">
              {zh ? '创建' : 'Create'}
            </button>
          </div>
        </form>
        <form onSubmit={join} className="rounded-lg border border-hairline bg-paper p-5">
          <div className="font-mono text-[10.5px] uppercase tracking-[0.16em] text-faint">
            {zh ? '加入班级' : 'Join a class'}
          </div>
          <div className="mt-3 flex gap-2">
            <input
              value={code}
              onChange={(e) => setCode(e.target.value.toUpperCase())}
              placeholder={zh ? '邀请码' : 'Invite code'}
              className={`${field} flex-1 font-mono tracking-widest`}
              maxLength={6}
            />
            <button type="submit" disabled={busy} className="rounded-md border border-hairline px-4 text-[13px] text-faint hover:bg-bone hover:text-ink disabled:opacity-50">
              {zh ? '加入' : 'Join'}
            </button>
          </div>
        </form>
      </div>
      {error && <p className="mt-3 rounded-md bg-pale-red px-4 py-2 text-[13px] text-ink-red">{error}</p>}

      {data && (
        <>
          <Section title={zh ? '我创建的班级' : 'Classes I teach'}>
            {data.owned.length === 0 ? (
              <Empty text={zh ? '还没有创建班级' : 'No classes yet'} />
            ) : (
              data.owned.map((c) => (
                <div key={c.id} className="flex items-center justify-between rounded-lg border border-hairline bg-paper px-5 py-3.5">
                  <Link to={`/class/${c.id}`} className="min-w-0 flex-1">
                    <div className="truncate text-[15px] font-medium hover:underline">{c.name}</div>
                    <div className="mt-0.5 font-mono text-[11.5px] text-faint">
                      {zh ? '邀请码' : 'code'} {c.invite_code}
                    </div>
                  </Link>
                  <div className="flex shrink-0 items-center gap-3">
                    <button
                      type="button"
                      onClick={() => copy(c.invite_code)}
                      className="font-mono text-[11.5px] text-faint hover:text-ink"
                    >
                      {copied === c.invite_code ? (zh ? '已复制' : 'copied') : zh ? '复制码' : 'copy'}
                    </button>
                    <button
                      type="button"
                      onClick={() => deleteClass(c.id).then(reload)}
                      className="font-mono text-[11.5px] text-faint hover:text-ink-red"
                    >
                      {zh ? '删除' : 'delete'}
                    </button>
                  </div>
                </div>
              ))
            )}
          </Section>
          <Section title={zh ? '我加入的班级' : 'Classes I joined'}>
            {data.joined.length === 0 ? (
              <Empty text={zh ? '还没有加入班级' : 'Not in any class'} />
            ) : (
              data.joined.map((c) => (
                <div key={c.id} className="flex items-center justify-between rounded-lg border border-hairline bg-paper px-5 py-3.5">
                  <Link to={`/class/${c.id}`} className="min-w-0 flex-1 truncate text-[15px] font-medium hover:underline">
                    {c.name}
                  </Link>
                  <button
                    type="button"
                    onClick={() => leaveClass(c.id, profile.id).then(reload)}
                    className="font-mono text-[11.5px] text-faint hover:text-ink-red"
                  >
                    {zh ? '退出' : 'leave'}
                  </button>
                </div>
              ))
            )}
          </Section>
        </>
      )}
    </div>
  );
}

function Notice({ text }: { text: string }) {
  return <div className="py-32 text-center text-[14px] text-faint">{text}</div>;
}
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mt-10">
      <div className="mb-3 font-mono text-[10.5px] uppercase tracking-[0.16em] text-faint">{title}</div>
      <div className="space-y-2">{children}</div>
    </section>
  );
}
function Empty({ text }: { text: string }) {
  return <div className="rounded-lg border border-dashed border-hairline px-5 py-6 text-center text-[13px] text-faint">{text}</div>;
}
