import { useMemo, useState, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { caseChecklist } from '../game/case/checklist';
import { buildL1Markdown } from '../game/case/L1Export';
import {
  addEdge,
  canVisitPoi,
  canRunCaseExperiment,
  caseStatusLabel,
  fillDemoScaffoldTexts,
  getDemoPois,
  hasPassedReview,
  markClosed,
  playDemoScript,
  prepareBoardForReview,
  previewReview,
  removeEdge,
  resetCase,
  runCaseExperiment,
  setClaimState,
  setHypothesis,
  setLimitations,
  setResearchQuestion,
  setResultsSummary,
  startDemoCase,
  submitReview,
  talkToMentor,
  useCase,
  visitPoi,
} from '../game/case/caseStore';
import type { CasePoi, EdgeType, HypothesisMaturity, ResearchCase } from '../game/case/types';
import { useLang } from '../lib/i18n';

export function CaseWorkbenchPage() {
  const { lang } = useLang();
  const zh = lang === 'zh';
  const c = useCase();
  const pois = useMemo(() => getDemoPois(), []);
  const [mentorMsg, setMentorMsg] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [edgeFrom, setEdgeFrom] = useState('');
  const [edgeTo, setEdgeTo] = useState('');
  const [edgeType, setEdgeType] = useState<EdgeType>('supports');
  const [report, setReport] = useState<string | null>(null);
  const [limText, setLimText] = useState('');
  const [expBusy, setExpBusy] = useState(false);
  const [expDetail, setExpDetail] = useState<string | null>(null);
  const [scriptBusy, setScriptBusy] = useState(false);

  const flash = (msg: string) => {
    setToast(msg);
    window.setTimeout(() => setToast(null), 2200);
  };

  if (!c) {
    return (
      <div className="research-game-ui mx-auto max-w-3xl px-5 py-16">
        <p className="font-mono text-[11px] tracking-[0.18em] text-faint">CASE DESK · V0</p>
        <h1 className="mt-3 font-serif text-[32px] font-semibold">
          {zh ? '案件工作台' : 'Case workbench'}
        </h1>
        <p className="mt-3 max-w-xl text-[15px] leading-relaxed text-faint">
          {zh
            ? '探案主循环：接案 → 取证 → 导师 → 实验 → 组板 → 审稿 → 导出 L1 训练论文。也可先回岛上实地检视。'
            : 'Detective loop: brief → evidence → mentor → experiment → board → review → L1 export.'}
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <button
            type="button"
            className="rounded-md bg-ink px-5 py-2.5 text-[14px] text-paper hover:bg-ink/90"
            onClick={() => {
              startDemoCase();
              flash(zh ? '案件已立案' : 'Case opened');
            }}
          >
            {zh ? '开始示范案件：被夸大的准确率' : 'Start demo case: Inflated Accuracy'}
          </button>
          <button
            type="button"
            className="rounded-md border border-hairline px-5 py-2.5 text-[14px] hover:bg-bone disabled:opacity-50"
            disabled={scriptBusy}
            onClick={() => {
              setScriptBusy(true);
              void playDemoScript({ submit: true, exportL1: true })
                .then((r) => {
                  flash(
                    r.ok
                      ? zh
                        ? '剧本跑通：已结案'
                        : 'Script OK: case closed'
                      : zh
                        ? `剧本未过审（${r.findingsCount ?? '?'}）`
                        : `Script failed review (${r.findingsCount ?? '?'})`,
                  );
                })
                .finally(() => setScriptBusy(false));
            }}
          >
            {scriptBusy ? (zh ? '剧本执行中…' : 'Running…') : zh ? '一键跑通 30 分钟剧本' : 'Run 30‑min script'}
          </button>
        </div>
        <p className="mt-6 text-[13px] text-faint">
          <Link to="/research-game" className="underline decoration-hairline underline-offset-2">
            {zh ? '返回科研岛' : 'Back to island'}
          </Link>
        </p>
      </div>
    );
  }

  const lastReview = c.reviewRounds[c.reviewRounds.length - 1];
  const checklist = caseChecklist(c);
  const preview = previewReview();
  const passed = hasPassedReview(c);
  const loc = (x: { zh: string; en: string }) => (zh ? x.zh : x.en);

  return (
    <div className="research-game-ui min-h-dvh bg-canvas pb-16">
      <header className="border-b border-hairline bg-paper/90">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-5 py-4">
          <div className="min-w-0">
            <p className="font-mono text-[10.5px] tracking-[0.16em] text-faint">
              CASE · {c.id} · {caseStatusLabel(c.status, zh)}
            </p>
            <h1 className="truncate font-serif text-[22px] font-semibold sm:text-[26px]">{loc(c.title)}</h1>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <CompletenessBar value={c.completeness} zh={zh} />
            <button
              type="button"
              className="rounded-md border border-hairline px-3 py-1.5 text-[12.5px] text-faint hover:text-ink disabled:opacity-50"
              disabled={scriptBusy}
              onClick={() => {
                setScriptBusy(true);
                void playDemoScript({ submit: true, exportL1: false })
                  .then((r) =>
                    flash(
                      r.ok
                        ? zh
                          ? '剧本已跑到可结案'
                          : 'Script ready to export'
                        : zh
                          ? '剧本未过审'
                          : 'Script failed review',
                    ),
                  )
                  .finally(() => setScriptBusy(false));
              }}
            >
              {zh ? '一键剧本' : 'Script'}
            </button>
            <button
              type="button"
              className="rounded-md border border-hairline px-3 py-1.5 text-[12.5px] text-faint hover:text-ink"
              onClick={() => {
                if (window.confirm(zh ? '重置案件？本地卷宗将清空。' : 'Reset case? Local dossier will be cleared.')) {
                  resetCase();
                  setReport(null);
                  setMentorMsg(null);
                  setExpDetail(null);
                  setLimText('');
                }
              }}
            >
              {zh ? '重置' : 'Reset'}
            </button>
            <Link
              to="/research-game"
              className="rounded-md border border-hairline px-3 py-1.5 text-[12.5px] text-faint hover:text-ink"
            >
              {zh ? '回岛探案' : 'Island'}
            </Link>
          </div>
        </div>
      </header>

      {toast && (
        <div className="fixed bottom-6 left-1/2 z-50 -translate-x-1/2 rounded-md bg-ink px-4 py-2 text-[13px] text-paper shadow-lift">
          {toast}
        </div>
      )}

      <div className="mx-auto grid max-w-6xl gap-6 px-5 py-6 lg:grid-cols-[1fr_1.1fr]">
        {/* Left: world desk actions */}
        <section className="space-y-4">
          <Panel title={zh ? '异常通报' : 'Bulletin'}>
            <p className="text-[14px] leading-relaxed text-ink">{loc(c.bulletin)}</p>
          </Panel>

          <Panel title={zh ? '现场取证（书桌模式）' : 'Field POIs (desk mode)'}>
            <ul className="space-y-2">
              {pois.map((poi) => (
                <PoiRow
                  key={poi.id}
                  poi={poi}
                  c={c}
                  zh={zh}
                  onVisit={() => {
                    if (poi.id === 'poi_mentor') return;
                    const r = visitPoi(poi.id);
                    if (!r.ok) {
                      flash(
                        r.reason === 'soft_gate_flag'
                          ? zh
                            ? '信息门：先与导师说清矛盾'
                            : 'Soft gate: name the contradiction to the mentor first'
                          : r.reason === 'soft_gate_evidence'
                            ? zh
                              ? '信息门：还缺前置证据'
                              : 'Soft gate: missing prerequisite evidence'
                            : r.reason === 'already_inspected'
                              ? zh
                                ? '已检视过'
                                : 'Already inspected'
                              : zh
                                ? '无法检视'
                                : 'Cannot inspect',
                      );
                      return;
                    }
                    flash(
                      r.gained.length
                        ? zh
                          ? `获得证据 ${r.gained.length} 条`
                          : `Gained ${r.gained.length} evidence`
                        : zh
                          ? '已检视'
                          : 'Inspected',
                    );
                  }}
                />
              ))}
            </ul>
          </Panel>

          <Panel title={zh ? '导师' : 'Mentor'}>
            <p className="text-[13px] text-faint">
              {zh
                ? `状态机：${c.mentorStage}。带着「海报 vs 复现」矛盾来；说不清不给关键提示。`
                : `Stage: ${c.mentorStage}. Bring the poster-vs-repro contradiction.`}
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              <button
                type="button"
                className="rounded-md border border-hairline px-3 py-1.5 text-[13px] hover:bg-bone"
                onClick={() => {
                  const r = talkToMentor(false);
                  setMentorMsg(loc(r.message));
                }}
              >
                {zh ? '含糊请教' : 'Vague ask'}
              </button>
              <button
                type="button"
                className="rounded-md bg-ink px-3 py-1.5 text-[13px] text-paper hover:bg-ink/90"
                onClick={() => {
                  const r = talkToMentor(true);
                  setMentorMsg(loc(r.message));
                  if (r.ok && r.stage === 'taught_controls') {
                    flash(zh ? '已解锁：协议/泄漏线索' : 'Unlocked: protocol/leak lead');
                  }
                }}
              >
                {zh ? '说清矛盾 / 再听要点' : 'State contradiction / recap'}
              </button>
            </div>
            {mentorMsg && <p className="mt-3 border-l-2 border-hairline pl-3 text-[13.5px] leading-relaxed">{mentorMsg}</p>}
            {c.flags.includes('knows_leak') && (
              <p className="mt-2 font-mono text-[11px] text-ink-green">flag: knows_leak · stage {c.mentorStage}</p>
            )}
          </Panel>

          <Panel title={zh ? '机房实验' : 'Experiment'}>
            <p className="text-[13px] text-faint">
              {zh
                ? '优先尝试 Research API 创建真实 run；失败或未配置则回退本地模拟。需先取得切分证据。'
                : 'Prefer live ResearchRun create; fall back to simulation. Need split evidence first.'}
            </p>
            <button
              type="button"
              className="mt-3 rounded-md bg-ink px-3 py-1.5 text-[13px] text-paper disabled:opacity-40"
              disabled={!canRunCaseExperiment(c) || expBusy}
              onClick={() => {
                setExpBusy(true);
                void runCaseExperiment()
                  .then((r) => {
                    setExpDetail(
                      r.mode === 'live'
                        ? zh
                          ? `已挂真实 run${r.link.researchRunId ? ` ${r.link.researchRunId.slice(0, 8)}…` : ''}`
                          : `Live run linked${r.link.researchRunId ? ` ${r.link.researchRunId.slice(0, 8)}…` : ''}`
                        : r.detail || (zh ? '已用模拟结果' : 'Used simulation'),
                    );
                    flash(
                      r.mode === 'live'
                        ? zh
                          ? '真实 ResearchRun 已挂上卷宗'
                          : 'Live ResearchRun linked'
                        : zh
                          ? '模拟对照已挂上卷宗'
                          : 'Simulated control linked',
                    );
                  })
                  .catch(() => flash(zh ? '无法开跑：还缺前置证据' : 'Cannot run: missing prereq'))
                  .finally(() => setExpBusy(false));
              }}
            >
              {expBusy ? (zh ? '运行中…' : 'Running…') : zh ? '跑清洁切分对照' : 'Run clean-split control'}
            </button>
            {expDetail && <p className="mt-2 font-mono text-[11px] text-faint">{expDetail}</p>}
            {c.experimentRuns.length > 0 && (
              <ul className="mt-3 space-y-1 text-[12.5px] text-faint">
                {c.experimentRuns.map((r) => (
                  <li key={r.id}>
                    {r.label}
                    {r.simulated ? (zh ? ' · 模拟' : ' · sim') : r.researchRunId ? ' · live' : ''}
                  </li>
                ))}
              </ul>
            )}
          </Panel>
        </section>

        {/* Right: dossier board */}
        <section className="space-y-4">
          <Panel title={zh ? '卷宗正文' : 'Dossier fields'}>
            <label className="block text-[12px] text-faint">{zh ? '研究问题' : 'Question'}</label>
            <textarea
              className="mt-1 w-full rounded-md border border-hairline bg-canvas px-3 py-2 text-[13.5px] outline-none focus:border-faint"
              rows={3}
              value={c.researchQuestion}
              onChange={(e) => setResearchQuestion(e.target.value)}
            />
            <label className="mt-3 block text-[12px] text-faint">{zh ? '假设' : 'Hypothesis'}</label>
            <textarea
              className="mt-1 w-full rounded-md border border-hairline bg-canvas px-3 py-2 text-[13.5px] outline-none focus:border-faint"
              rows={3}
              value={c.hypothesis.text}
              onChange={(e) => setHypothesis(e.target.value, c.hypothesis.maturity)}
            />
            <label className="mt-2 block text-[12px] text-faint">{zh ? '假设成熟度' : 'Maturity'}</label>
            <select
              className="mt-1 rounded-md border border-hairline bg-canvas px-2 py-1.5 text-[13px]"
              value={c.hypothesis.maturity}
              onChange={(e) => setHypothesis(c.hypothesis.text, e.target.value as HypothesisMaturity)}
            >
              <option value="wild_guess">{zh ? '猜测' : 'wild guess'}</option>
              <option value="plausible">{zh ? '可检验' : 'plausible'}</option>
              <option value="operationalized">{zh ? '可操作' : 'operationalized'}</option>
            </select>
            <label className="mt-3 block text-[12px] text-faint">{zh ? '结果摘要' : 'Results'}</label>
            <textarea
              className="mt-1 w-full rounded-md border border-hairline bg-canvas px-3 py-2 text-[13.5px] outline-none focus:border-faint"
              rows={2}
              value={c.resultsSummary}
              onChange={(e) => setResultsSummary(e.target.value)}
            />
            <label className="mt-3 block text-[12px] text-faint">{zh ? '局限（一行一条，用换行）' : 'Limitations (one per line)'}</label>
            <textarea
              className="mt-1 w-full rounded-md border border-hairline bg-canvas px-3 py-2 text-[13.5px] outline-none focus:border-faint"
              rows={2}
              value={limText || c.limitations.join('\n')}
              onChange={(e) => {
                setLimText(e.target.value);
                setLimitations(e.target.value.split('\n'));
              }}
            />
            <button
              type="button"
              className="mt-3 rounded-md border border-hairline px-3 py-1.5 text-[12.5px] text-faint hover:text-ink"
              onClick={() => {
                fillDemoScaffoldTexts();
                setLimText('');
                flash(zh ? '已填入示范正文骨架' : 'Filled scaffold texts');
              }}
            >
              {zh ? '填入示范正文（省时）' : 'Fill scaffold texts'}
            </button>
          </Panel>

          <Panel title={zh ? '证据板' : 'Evidence board'}>
            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <p className="font-mono text-[10.5px] text-faint">CLAIMS</p>
                <ul className="mt-2 space-y-2">
                  {c.claims.map((cl) => (
                    <li key={cl.id} className="rounded-md border border-hairline bg-bone/60 px-2 py-2 text-[12.5px]">
                      <p>{cl.text}</p>
                      <div className="mt-1 flex flex-wrap gap-1">
                        {(['active', 'retracted', 'accepted'] as const).map((s) => (
                          <button
                            key={s}
                            type="button"
                            className={`rounded px-1.5 py-0.5 font-mono text-[10px] ${
                              cl.state === s ? 'bg-ink text-paper' : 'bg-paper text-faint'
                            }`}
                            onClick={() => setClaimState(cl.id, s)}
                          >
                            {s}
                          </button>
                        ))}
                      </div>
                      <p className="mt-1 font-mono text-[10px] text-faint">{cl.id}</p>
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <p className="font-mono text-[10.5px] text-faint">EVIDENCE</p>
                <ul className="mt-2 max-h-64 space-y-2 overflow-y-auto">
                  {c.evidence.length === 0 && (
                    <li className="text-[12.5px] text-faint">{zh ? '尚未取证' : 'No evidence yet'}</li>
                  )}
                  {c.evidence.map((e) => (
                    <li key={e.id} className="rounded-md border border-hairline bg-paper px-2 py-2 text-[12.5px]">
                      <p className="font-medium">
                        <span className="font-mono text-[10px] text-faint">[{e.kind}] </span>
                        {loc(e.title)}
                      </p>
                      <p className="mt-1 text-[12px] leading-relaxed text-faint">{loc(e.body)}</p>
                      <p className="mt-1 font-mono text-[10px] text-faint">{e.id}</p>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="mt-4 border-t border-hairline pt-3">
              <p className="font-mono text-[10.5px] text-faint">EDGES</p>
              <div className="mt-2 flex flex-wrap items-end gap-2">
                <label className="text-[11px] text-faint">
                  from
                  <input
                    className="mt-0.5 block w-36 rounded border border-hairline bg-canvas px-2 py-1 font-mono text-[11px]"
                    placeholder="ev_…"
                    value={edgeFrom}
                    onChange={(e) => setEdgeFrom(e.target.value.trim())}
                    list="case-node-ids"
                  />
                </label>
                <label className="text-[11px] text-faint">
                  to
                  <input
                    className="mt-0.5 block w-36 rounded border border-hairline bg-canvas px-2 py-1 font-mono text-[11px]"
                    placeholder="cl_…"
                    value={edgeTo}
                    onChange={(e) => setEdgeTo(e.target.value.trim())}
                    list="case-node-ids"
                  />
                </label>
                <select
                  className="rounded border border-hairline bg-canvas px-2 py-1 text-[12px]"
                  value={edgeType}
                  onChange={(e) => setEdgeType(e.target.value as EdgeType)}
                >
                  <option value="supports">supports</option>
                  <option value="contradicts">contradicts</option>
                  <option value="refines">refines</option>
                </select>
                <button
                  type="button"
                  className="rounded-md bg-ink px-3 py-1 text-[12px] text-paper"
                  onClick={() => {
                    if (!edgeFrom || !edgeTo) return;
                    addEdge(edgeFrom, edgeTo, edgeType);
                    flash(zh ? '已连边' : 'Edge added');
                  }}
                >
                  {zh ? '连边' : 'Link'}
                </button>
                <datalist id="case-node-ids">
                  {c.evidence.map((e) => (
                    <option key={e.id} value={e.id} />
                  ))}
                  {c.claims.map((cl) => (
                    <option key={cl.id} value={cl.id} />
                  ))}
                </datalist>
              </div>
              <ul className="mt-2 space-y-1">
                {c.edges.map((e) => (
                  <li key={e.id} className="flex items-center justify-between font-mono text-[11px] text-faint">
                    <span>
                      {e.from} —{e.type}→ {e.to}
                    </span>
                    <button type="button" className="text-ink-red" onClick={() => removeEdge(e.id)}>
                      ×
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          </Panel>

          <Panel title={zh ? '结案清单 / 审稿庭' : 'Checklist / Review'}>
            <ul className="space-y-1.5">
              {checklist.map((item) => (
                <li
                  key={item.id}
                  className={`flex items-start gap-2 text-[12.5px] ${item.ok ? 'text-ink-green' : 'text-faint'}`}
                >
                  <span className="font-mono text-[11px]">{item.ok ? '✓' : '·'}</span>
                  <span>{zh ? item.zh : item.en}</span>
                </li>
              ))}
            </ul>
            <p className="mt-2 font-mono text-[11px] text-faint">
              {zh ? '预检洞' : 'Preview findings'}: {preview.length}
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              <button
                type="button"
                className="rounded-md border border-hairline px-3 py-1.5 text-[12.5px] hover:bg-bone"
                onClick={() => {
                  prepareBoardForReview();
                  setLimText('');
                  flash(zh ? '已整理卷宗（含撤回过度主张）' : 'Board prepared (overclaim retracted)');
                }}
              >
                {zh ? '一键整理卷宗' : 'Prepare board'}
              </button>
              <button
                type="button"
                className="rounded-md bg-ink px-3 py-1.5 text-[13px] text-paper hover:bg-ink/90 disabled:opacity-40"
                disabled={c.status === 'closed'}
                onClick={() => {
                  const r = submitReview();
                  flash(
                    r.passed
                      ? zh
                        ? '审稿通过'
                        : 'Review passed'
                      : zh
                        ? `未通过（${r.findingsCount} 项）`
                        : `Failed (${r.findingsCount})`,
                  );
                }}
              >
                {zh ? '送审' : 'Submit for review'}
              </button>
            </div>
            {lastReview && (
              <div className="mt-3 space-y-2">
                <p className="font-mono text-[11px] text-faint">
                  round {lastReview.index} · {lastReview.status}
                </p>
                {lastReview.findings.map((f) => (
                  <div key={f.code + f.message.zh} className="rounded-md bg-pale-red/80 px-3 py-2 text-[12.5px] text-ink-red">
                    <p>{loc(f.message)}</p>
                    <p className="mt-1 text-ink-red/80">{loc(f.fixHint)}</p>
                  </div>
                ))}
                {lastReview.status === 'passed' && (
                  <p className="text-[13px] text-ink-green">{zh ? '可以导出 L1 结案。' : 'You may export the L1 report.'}</p>
                )}
              </div>
            )}
          </Panel>

          <Panel title={zh ? '导出 L1' : 'Export L1'}>
            <button
              type="button"
              className="rounded-md border border-hairline px-3 py-1.5 text-[13px] hover:bg-bone disabled:opacity-40"
              disabled={!passed}
              onClick={() => {
                const md = buildL1Markdown(c, zh ? 'zh' : 'en');
                setReport(md);
                markClosed();
                flash(zh ? 'L1 已生成并结案' : 'L1 generated · case closed');
              }}
            >
              {zh ? '生成训练论文并结案' : 'Generate L1 & close'}
            </button>
            {c.status === 'closed' && (
              <p className="mt-2 text-[12.5px] text-ink-green">{zh ? '本案已结案。' : 'Case closed.'}</p>
            )}
            {report && (
              <div className="mt-3">
                <pre className="max-h-80 overflow-auto whitespace-pre-wrap rounded-md border border-hairline bg-bone p-3 text-[12px] leading-relaxed">
                  {report}
                </pre>
                <div className="mt-2 flex flex-wrap gap-3">
                  <button
                    type="button"
                    className="text-[12.5px] text-faint underline"
                    onClick={() => {
                      void navigator.clipboard.writeText(report);
                      flash(zh ? '已复制' : 'Copied');
                    }}
                  >
                    {zh ? '复制全文' : 'Copy'}
                  </button>
                  <button
                    type="button"
                    className="text-[12.5px] text-faint underline"
                    onClick={() => {
                      const blob = new Blob([report], { type: 'text/markdown;charset=utf-8' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `${c.id}-L1.md`;
                      a.click();
                      URL.revokeObjectURL(url);
                      flash(zh ? '已下载 .md' : 'Downloaded .md');
                    }}
                  >
                    {zh ? '下载 Markdown' : 'Download .md'}
                  </button>
                </div>
              </div>
            )}
          </Panel>
        </section>
      </div>
    </div>
  );
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="rounded-lg border border-hairline bg-paper p-4 shadow-lift">
      <h2 className="font-mono text-[10.5px] tracking-[0.14em] text-faint">{title}</h2>
      <div className="mt-3">{children}</div>
    </section>
  );
}

function CompletenessBar({ value, zh }: { value: number; zh: boolean }) {
  const pct = Math.round(value * 100);
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-28 overflow-hidden rounded-full bg-bone">
        <div className="h-full bg-ink transition-all" style={{ width: `${pct}%` }} />
      </div>
      <span className="font-mono text-[11px] text-faint">
        {pct}% {zh ? '完整度' : 'complete'}
      </span>
    </div>
  );
}

function PoiRow({
  poi,
  c,
  zh,
  onVisit,
}: {
  poi: CasePoi;
  c: ResearchCase;
  zh: boolean;
  onVisit: () => void;
}) {
  const gate = canVisitPoi(poi, c);
  const isMentor = poi.id === 'poi_mentor';
  const label = zh ? poi.label.zh : poi.label.en;
  const hint = zh ? poi.visitHint.zh : poi.visitHint.en;
  return (
    <li className="flex items-start justify-between gap-3 rounded-md border border-hairline bg-canvas px-3 py-2">
      <div className="min-w-0">
        <p className="text-[13.5px] font-medium">{label}</p>
        <p className="mt-0.5 text-[12px] text-faint">{hint}</p>
        {!gate.ok && !isMentor && (
          <p className="mt-1 font-mono text-[10px] text-ink-yellow">
            {gate.reason === 'already_inspected' ? (zh ? '已取证' : 'done') : zh ? '未解锁' : 'locked'}
          </p>
        )}
      </div>
      {!isMentor && (
        <button
          type="button"
          className="shrink-0 rounded-md border border-hairline px-2.5 py-1 text-[12px] hover:bg-bone disabled:opacity-40"
          disabled={gate.reason === 'already_inspected'}
          onClick={onVisit}
        >
          {zh ? '检视' : 'Inspect'}
        </button>
      )}
      {isMentor && <span className="shrink-0 font-mono text-[10px] text-faint">→ mentor</span>}
    </li>
  );
}
