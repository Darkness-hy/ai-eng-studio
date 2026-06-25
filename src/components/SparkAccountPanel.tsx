import { useEffect, useRef, useState } from 'react';
import { useAuth } from '../lib/auth';
import { useLang } from '../lib/i18n';
import {
  cancelSparkRequest,
  getMySparkAccount,
  normalizeUsername,
  requestSparkAccount,
  type SparkAccountRow,
} from '../lib/sparkAccount';

const PENDING: SparkAccountRow['status'][] = ['requested', 'approved', 'provisioning'];
const VPN_URL = 'https://itsc.nju.edu.cn/21601/listm.htm';

// Supabase PostgrestError is a plain object (not an Error), so String(e) would
// render "[object Object]"; pull its message out instead.
const errText = (e: unknown): string =>
  e instanceof Error
    ? e.message
    : e && typeof e === 'object' && 'message' in e
      ? String((e as { message: unknown }).message)
      : String(e);

export function SparkAccountPanel() {
  const { lang } = useLang();
  const zh = lang === 'zh';
  const { profile } = useAuth();
  const [row, setRow] = useState<SparkAccountRow | null | undefined>(undefined); // undefined = loading
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [username, setUsername] = useState('');
  const [copied, setCopied] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const uid = profile?.id;

  useEffect(() => {
    if (!uid) return;
    getMySparkAccount(uid).then(setRow).catch(() => setRow(null));
  }, [uid]);

  // poll while the request is still being processed by the Spark agent
  useEffect(() => {
    if (timer.current) {
      clearInterval(timer.current);
      timer.current = null;
    }
    if (uid && row && PENDING.includes(row.status)) {
      timer.current = setInterval(() => {
        getMySparkAccount(uid).then(setRow).catch(() => {});
      }, 6000);
    }
    return () => {
      if (timer.current) clearInterval(timer.current);
    };
  }, [row, uid]);

  if (!profile) return null;

  const submit = () => {
    if (!uid) return;
    setBusy(true);
    setError(null);
    requestSparkAccount(uid, username)
      .then(setRow)
      .catch((e) => setError(errText(e)))
      .finally(() => setBusy(false));
  };
  const cancel = () => {
    if (!uid) return;
    setBusy(true);
    cancelSparkRequest(uid)
      .then(() => setRow(null))
      .catch((e) => setError(errText(e)))
      .finally(() => setBusy(false));
  };
  const copy = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(text);
      setTimeout(() => setCopied(null), 1500);
    });
  };

  const card = 'rounded-lg border border-hairline bg-paper p-5';
  const label = 'font-mono text-[10.5px] uppercase tracking-[0.16em] text-faint';
  const kv = 'flex items-center justify-between gap-3 rounded-md bg-bone px-3 py-2 font-mono text-[13px]';

  return (
    <section className={card}>
      <div className="flex items-center justify-between">
        <div className={label}>{zh ? '毕业设计 · Spark 账户' : 'Capstone · Spark account'}</div>
        {row && row.status !== 'ready' && (
          <span className="font-mono text-[10.5px] text-faint">{statusLabel(row.status, zh)}</span>
        )}
      </div>

      {row === undefined ? (
        <p className="mt-3 text-[13px] text-faint">{zh ? '加载中…' : 'Loading…'}</p>
      ) : row === null ? (
        <div className="mt-3 space-y-3">
          <p className="text-[13px] leading-relaxed text-faint">
            {zh
              ? '毕业设计在校园网内的 Spark 机器上进行。请先加入「Spark 使用班级」,然后在下方申请(或直接对 AI 助教说「申请spark账号」),系统会自动为你开通 Linux 账户,凭据显示在这里。'
              : 'The capstone runs on the campus Spark machine. First join the Spark class, then request below (or just tell the AI tutor “申请spark账号”). A Linux account is provisioned automatically and the credentials appear here.'}
          </p>
          <label className="block text-[12px] text-faint">
            {zh ? '账户名 = 你名字的小写全拼' : 'Username = your name in lowercase pinyin'}
          </label>
          <input
            value={username}
            onChange={(e) => setUsername(e.target.value.toLowerCase().replace(/[^a-z0-9]/g, ''))}
            placeholder={zh ? '如 dinghongyu' : 'e.g. dinghongyu'}
            maxLength={32}
            className="w-full rounded-md border border-hairline bg-canvas px-3 py-2 font-mono text-[13.5px] outline-none placeholder:text-faint focus:border-faint"
          />
          <button
            type="button"
            onClick={submit}
            disabled={busy || normalizeUsername(username) === null}
            className="rounded-md bg-ink px-4 py-2 text-[13px] text-white hover:bg-ink/85 disabled:opacity-50"
          >
            {zh ? '申请毕业设计账户' : 'Request capstone account'}
          </button>
        </div>
      ) : row.status === 'ready' ? (
        (() => {
          const sshCmd = `ssh -p ${row.ssh_port ?? 22} ${row.ssh_username}@${row.host}`;
          return (
            <div className="mt-3 space-y-2">
              <p className="text-[13px] text-ink-green">{zh ? '账户已开通 🎉 按以下步骤登录:' : 'Account ready — log in:'}</p>
              <ol className="space-y-2 text-[13px]">
                <li>
                  <span className="text-faint">{zh ? '① 先登录南大 VPN:' : '① Sign in to the NJU VPN:'}</span>{' '}
                  <a href={VPN_URL} target="_blank" rel="noreferrer" className="break-all underline decoration-faint underline-offset-2 hover:text-ink">
                    {VPN_URL}
                  </a>
                </li>
                <li>
                  <span className="text-faint">{zh ? '② 再 SSH 登录:' : '② Then SSH in:'}</span>
                  <div className={`${kv} mt-1`}>
                    <button type="button" onClick={() => copy(sshCmd)} className="truncate text-ink hover:underline">{sshCmd}</button>
                  </div>
                </li>
                <li>
                  <span className="text-faint">{zh ? '③ 临时密码:' : '③ Temp password:'}</span>
                  <div className={`${kv} mt-1`}>
                    <button type="button" onClick={() => copy(row.temp_password ?? '')} className="text-ink hover:underline">{row.temp_password}</button>
                  </div>
                </li>
              </ol>
              {copied && <p className="text-[11px] text-faint">{zh ? '已复制' : 'copied'}</p>}
              <p className="rounded-md bg-pale-yellow px-3 py-2 text-[12px] leading-relaxed text-ink-yellow">
                {zh
                  ? '首次登录会要求立即修改密码。这串临时密码仅用于第一次登录,请妥善保管、勿外传。'
                  : 'You must change this password on first login. It is one-time — keep it private.'}
              </p>
            </div>
          );
        })()
      ) : row.status === 'failed' ? (
        <div className="mt-3 space-y-3">
          <p className="rounded-md bg-pale-red px-3 py-2 text-[12.5px] text-ink-red">
            {row.error || (zh ? '开户失败,请联系管理员。' : 'Provisioning failed — contact an admin.')}
          </p>
          <button type="button" onClick={cancel} disabled={busy} className="rounded-md border border-hairline px-4 py-2 text-[13px] text-faint hover:text-ink disabled:opacity-50">
            {zh ? '撤回并重试' : 'Withdraw & retry'}
          </button>
        </div>
      ) : (
        <div className="mt-3 space-y-3">
          <p className="text-[13px] leading-relaxed text-faint">
            {row.status === 'requested'
              ? zh
                ? '申请已提交,系统正在校验班级并为你开通账户,这里会实时更新。'
                : 'Request submitted — verifying your class and provisioning. Updates here automatically.'
              : zh
                ? '正在 Spark 上为你开通账户…'
                : 'Provisioning your account on Spark…'}
          </p>
          {row.status === 'requested' && (
            <button type="button" onClick={cancel} disabled={busy} className="rounded-md border border-hairline px-4 py-2 text-[13px] text-faint hover:text-ink disabled:opacity-50">
              {zh ? '撤回申请' : 'Withdraw request'}
            </button>
          )}
        </div>
      )}

      {error && <p className="mt-3 rounded-md bg-pale-red px-3 py-2 text-[12.5px] text-ink-red">{error}</p>}
    </section>
  );
}

function statusLabel(s: SparkAccountRow['status'], zh: boolean): string {
  const map: Record<SparkAccountRow['status'], [string, string]> = {
    requested: ['待审批', 'pending approval'],
    approved: ['开通中', 'provisioning'],
    provisioning: ['开通中', 'provisioning'],
    ready: ['已开通', 'ready'],
    failed: ['失败', 'failed'],
    revoked: ['已回收', 'revoked'],
  };
  return map[s][zh ? 0 : 1];
}
