import { useState } from 'react';
import { isChallengePassed, markChallengePassed, useChallenges } from '../lib/challenges';
import { useLang } from '../lib/i18n';
import type { Challenge } from '../lib/types';

export function ChallengeWidget({ challenge, lessonId }: { challenge: Challenge; lessonId: string }) {
  const { lang } = useLang();
  const zh = lang === 'zh';
  useChallenges(); // re-render when pass state changes
  const alreadyPassed = isChallengePassed(lessonId);

  const [code, setCode] = useState(challenge.starter);
  const [status, setStatus] = useState<'idle' | 'running' | 'pass' | 'fail'>(
    alreadyPassed ? 'pass' : 'idle',
  );
  const [output, setOutput] = useState('');
  const [showSolution, setShowSolution] = useState(false);

  const run = async () => {
    setStatus('running');
    setOutput('');
    const { runPython } = await import('../lib/pyodide');
    const program = `${code}\n\n# ── hidden tests ──\n${challenge.tests}\nprint("ALL_TESTS_PASSED")`;
    const res = await runPython(program);
    if (res.error || !res.output.includes('ALL_TESTS_PASSED')) {
      setStatus('fail');
      setOutput((res.output + (res.error ?? '')).replace('ALL_TESTS_PASSED', '').trim());
    } else {
      setStatus('pass');
      setOutput('');
      markChallengePassed(lessonId);
    }
  };

  return (
    <section className="my-10 overflow-hidden rounded-lg border border-hairline bg-paper">
      <header className="flex items-baseline justify-between border-b border-hairline px-6 py-4">
        <div className="flex items-baseline gap-3">
          <span className="rounded-full bg-pale-blue px-2.5 py-0.5 text-[11px] font-medium uppercase tracking-[0.05em] text-ink-blue">
            {zh ? '编码挑战' : 'Challenge'}
          </span>
          <span className="font-serif text-[18px] font-semibold">
            {zh ? challenge.titleZh : challenge.titleEn}
          </span>
        </div>
        {status === 'pass' && (
          <span className="rounded-full bg-pale-green px-2.5 py-0.5 text-[11px] text-ink-green">
            {zh ? '已通过' : 'Passed'}
          </span>
        )}
      </header>

      <div className="px-6 py-5">
        <p className="text-[14.5px] leading-relaxed text-ink">{zh ? challenge.promptZh : challenge.promptEn}</p>

        <textarea
          value={code}
          onChange={(e) => setCode(e.target.value)}
          spellCheck={false}
          onKeyDown={(e) => {
            if (e.key === 'Tab') {
              e.preventDefault();
              const t = e.currentTarget;
              const s = t.selectionStart;
              const v = t.value;
              const next = v.slice(0, s) + '    ' + v.slice(t.selectionEnd);
              setCode(next);
              requestAnimationFrame(() => {
                t.selectionStart = t.selectionEnd = s + 4;
              });
            }
          }}
          className="mt-4 h-64 w-full resize-y rounded-md border border-hairline bg-bone/40 p-4 font-mono text-[13px] leading-relaxed text-ink outline-none focus:border-faint"
        />

        <div className="mt-3 flex items-center gap-3">
          <button
            type="button"
            disabled={status === 'running'}
            onClick={run}
            className="rounded-md bg-ink px-4 py-2 font-mono text-[12px] text-white transition-colors hover:bg-ink/85 disabled:opacity-50"
          >
            {status === 'running' ? (zh ? '运行中…' : 'Running…') : zh ? '▸ 运行测试' : '▸ Run tests'}
          </button>
          <button
            type="button"
            onClick={() => setShowSolution((v) => !v)}
            className="font-mono text-[11.5px] text-faint underline decoration-hairline underline-offset-2 hover:text-ink"
          >
            {showSolution ? (zh ? '隐藏参考答案' : 'Hide solution') : zh ? '查看参考答案' : 'Show solution'}
          </button>
        </div>

        {status === 'running' && !output && (
          <p className="mt-3 font-mono text-[12px] text-faint">
            {zh ? '加载 Python 运行时（首次约 10 MB）…' : 'Loading Python runtime…'}
          </p>
        )}
        {status === 'pass' && (
          <p className="mt-3 rounded-md bg-pale-green px-4 py-3 text-[13.5px] text-ink-green">
            {zh ? '全部测试通过，挑战完成。' : 'All tests passed — challenge complete.'}
          </p>
        )}
        {status === 'fail' && (
          <div className="mt-3 rounded-md bg-pale-red px-4 py-3">
            <p className="text-[13px] font-medium text-ink-red">{zh ? '还没通过，看看错误：' : 'Not yet — check the error:'}</p>
            <pre className="mt-2 overflow-x-auto font-mono text-[12px] whitespace-pre-wrap text-ink-red">{output}</pre>
          </div>
        )}

        {showSolution && (
          <pre className="mt-3 overflow-x-auto rounded-md border border-hairline bg-bone/40 p-4 font-mono text-[12.5px] leading-relaxed">
            {challenge.solution}
          </pre>
        )}
      </div>
    </section>
  );
}
