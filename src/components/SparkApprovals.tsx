import { useEffect, useState } from 'react';
import { useAuth } from '../lib/auth';
import { useLang } from '../lib/i18n';
import { approveSparkAccount, listSparkRequests, type SparkRequestRow } from '../lib/sparkAccount';

/** Admin-only: review and approve capstone Spark account requests. */
export function SparkApprovals() {
  const { lang } = useLang();
  const zh = lang === 'zh';
  const { profile } = useAuth();
  const [rows, setRows] = useState<SparkRequestRow[] | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const reload = () => listSparkRequests().then(setRows).catch(() => setRows([]));
  useEffect(() => {
    reload();
  }, []);

  if (profile?.role !== 'admin') return null;

  const approve = (userId: string) => {
    if (!profile) return;
    setBusy(userId);
    approveSparkAccount(userId, profile.id)
      .then(reload)
      .catch(() => {})
      .finally(() => setBusy(null));
  };

  const badge = (s: SparkRequestRow['status']) => {
    const tone =
      s === 'ready' ? 'bg-pale-green text-ink-green'
      : s === 'failed' ? 'bg-pale-red text-ink-red'
      : s === 'requested' ? 'bg-pale-yellow text-ink-yellow'
      : 'bg-bone text-faint';
    return <span className={`rounded px-1.5 py-0.5 font-mono text-[10px] ${tone}`}>{s}</span>;
  };

  return (
    <section className="mt-8 rounded-lg border border-hairline bg-paper">
      <div className="flex items-center justify-between border-b border-hairline px-6 py-4">
        <div className="font-mono text-[10.5px] uppercase tracking-[0.16em] text-faint">
          {zh ? '毕业设计 Spark 账户申请' : 'Capstone Spark requests'}
        </div>
        {rows && rows.some((r) => r.status === 'requested') && (
          <span className="font-mono text-[10.5px] text-ink-yellow">
            {rows.filter((r) => r.status === 'requested').length} {zh ? '待审批' : 'pending'}
          </span>
        )}
      </div>
      {rows == null ? (
        <p className="px-6 py-4 text-[13px] text-faint">{zh ? '加载中…' : 'Loading…'}</p>
      ) : rows.length === 0 ? (
        <p className="px-6 py-4 text-[13px] text-faint">{zh ? '暂无申请' : 'No requests yet'}</p>
      ) : (
        <table className="w-full text-[13px]">
          <thead>
            <tr className="border-b border-hairline text-left font-mono text-[10.5px] uppercase tracking-[0.12em] text-faint">
              <th className="px-6 py-3 font-normal">{zh ? '学员' : 'Learner'}</th>
              <th className="px-3 py-3 font-normal">{zh ? '状态' : 'Status'}</th>
              <th className="px-3 py-3 font-normal">{zh ? '账户' : 'Account'}</th>
              <th className="px-6 py-3 font-normal" />
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.user_id} className="border-b border-hairline last:border-0">
                <td className="px-6 py-3">
                  <div className="text-ink">{r.profiles?.display_name ?? r.profiles?.email ?? r.user_id.slice(0, 8)}</div>
                  {r.note && <div className="font-mono text-[11px] text-faint">{r.note}</div>}
                </td>
                <td className="px-3 py-3">{badge(r.status)}</td>
                <td className="px-3 py-3 font-mono text-[12px] text-faint">{r.ssh_username ?? '—'}</td>
                <td className="px-6 py-3 text-right">
                  {r.status === 'requested' && (
                    <button
                      type="button"
                      onClick={() => approve(r.user_id)}
                      disabled={busy === r.user_id}
                      className="rounded-md bg-ink px-3 py-1.5 text-[12px] text-white hover:bg-ink/85 disabled:opacity-50"
                    >
                      {zh ? '批准' : 'Approve'}
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
