import { useEffect, useRef, useState } from 'react';
import { useLang } from '../../lib/i18n';

const STAGES = ['think', 'act', 'observe', 'check'] as const;
type Stage = (typeof STAGES)[number];

interface ScriptEntry {
  kind: Stage | 'final';
  iter: 1 | 2 | 3;
  zh: string;
  en: string;
}

const SCRIPT: readonly ScriptEntry[] = [
  // ---- iteration 1: get the weather ----
  {
    kind: 'think',
    iter: 1,
    zh: '任务要查天气、换算华氏度、写入文件。第一步：先拿到北京今天的天气。',
    en: 'The task needs weather, a °F conversion, and a file write. Step one: fetch today’s weather in Beijing.',
  },
  {
    kind: 'act',
    iter: 1,
    zh: 'weather.get(city="北京")',
    en: 'weather.get(city="Beijing")',
  },
  {
    kind: 'observe',
    iter: 1,
    zh: '{"temp_c": 31, "sky": "晴"}',
    en: '{"temp_c": 31, "sky": "clear"}',
  },
  {
    kind: 'check',
    iter: 1,
    zh: '目标完成？还没——温度还没换算',
    en: 'Done? Not yet — the temperature isn’t converted',
  },
  // ---- iteration 2: convert to °F ----
  {
    kind: 'think',
    iter: 2,
    zh: '拿到 31°C。换算公式：F = C × 9/5 + 32，交给计算器算。',
    en: 'Got 31°C. Conversion: F = C × 9/5 + 32 — hand it to the calculator.',
  },
  {
    kind: 'act',
    iter: 2,
    zh: 'calc.eval("31*9/5+32")',
    en: 'calc.eval("31*9/5+32")',
  },
  {
    kind: 'observe',
    iter: 2,
    zh: '{"result": 87.8}',
    en: '{"result": 87.8}',
  },
  {
    kind: 'check',
    iter: 2,
    zh: '目标完成？还没——结果还没写入 report.txt',
    en: 'Done? Not yet — nothing written to report.txt',
  },
  // ---- iteration 3: write the file ----
  {
    kind: 'think',
    iter: 3,
    zh: '87.8°F 已算出。最后一步：把结果写入 report.txt。',
    en: '87.8°F computed. Last step: write the result to report.txt.',
  },
  {
    kind: 'act',
    iter: 3,
    zh: 'fs.write("report.txt", "北京：晴，31°C / 87.8°F")',
    en: 'fs.write("report.txt", "Beijing: clear, 31°C / 87.8°F")',
  },
  {
    kind: 'observe',
    iter: 3,
    zh: '{"ok": true}',
    en: '{"ok": true}',
  },
  {
    kind: 'check',
    iter: 3,
    zh: '目标完成？是——已查询、已换算、已写入',
    en: 'Done? Yes — fetched, converted, written',
  },
  {
    kind: 'final',
    iter: 3,
    zh: '北京今天晴，31°C（87.8°F），已写入 report.txt',
    en: 'Beijing is clear today, 31°C (87.8°F) — written to report.txt',
  },
];

const STAGE_LABELS: Record<Stage, { zh: string; en: string }> = {
  think: { zh: '思考 Think', en: 'Think' },
  act: { zh: '行动 Act', en: 'Act' },
  observe: { zh: '观察 Observe', en: 'Observe' },
  check: { zh: '判断 Done?', en: 'Done?' },
};

function CheckIcon() {
  return (
    <svg viewBox="0 0 12 12" className="h-3 w-3 shrink-0" aria-hidden="true">
      <path d="M2 6.5 5 9.5 10 3" fill="none" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

function DownArrow() {
  return (
    <div className="flex justify-center py-1 text-faint">
      <svg viewBox="0 0 12 12" className="h-3 w-3" aria-hidden="true">
        <path d="M6 1v8M3 6.5 6 9.5l3-3" fill="none" stroke="currentColor" strokeWidth="1.2" />
      </svg>
    </div>
  );
}

export function AgentLoopSim() {
  const { lang } = useLang();
  const zh = lang === 'zh';

  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);

  const finished = step >= SCRIPT.length;
  const activePlaying = playing && !finished;
  const current = step > 0 ? SCRIPT[step - 1] : null;

  useEffect(() => {
    if (!activePlaying) return;
    const id = window.setInterval(() => {
      setStep((s) => (s < SCRIPT.length ? s + 1 : s));
    }, 600);
    return () => window.clearInterval(id);
  }, [activePlaying]);


  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [step]);

  const stageState = (stage: Stage): 'current' | 'visited' | 'idle' => {
    if (!current) return 'idle';
    if (finished || current.kind === 'final') return 'visited';
    const cur = STAGES.indexOf(current.kind);
    const own = STAGES.indexOf(stage);
    if (own === cur) return 'current';
    return own < cur ? 'visited' : 'idle';
  };

  const stageWord = current
    ? current.kind === 'final'
      ? 'done'
      : current.kind === 'check'
        ? 'done?'
        : current.kind
    : '–';
  const counter = `iter ${current ? current.iter : 0}/3 · ${stageWord}`;

  const reset = () => {
    setPlaying(false);
    setStep(0);
  };

  return (
    <section className="my-10 overflow-hidden rounded-lg border border-hairline bg-paper">
      {/* header */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-hairline px-6 py-4">
        <span className="rounded-full bg-pale-blue px-2 py-0.5 font-mono text-[11px] text-ink-blue">
          {zh ? '交互实验' : 'Interactive'}
        </span>
        <h3 className="font-serif text-[15px] text-ink">
          {zh ? 'Agent 循环：一步一步看' : 'The agent loop, step by step'}
        </h3>
        <span className="text-[12px] text-faint">
          {zh ? '点「下一步」看它循环到完成' : 'Click “Next” and watch it loop to done'}
        </span>
        <div className="ml-auto flex items-center gap-2">
          <span className="font-mono text-[11px] text-faint">{counter}</span>
          {finished && (
            <span className="rounded-full bg-pale-green px-2 py-0.5 font-mono text-[11px] text-ink-green">
              {zh ? '完成' : 'Done'}
            </span>
          )}
          <button
            type="button"
            onClick={() => setStep((s) => Math.min(s + 1, SCRIPT.length))}
            disabled={finished}
            className="rounded-md bg-ink px-3 py-1 font-mono text-[11px] text-white hover:bg-ink/80 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {zh ? '下一步' : 'Next'}
          </button>
          <button
            type="button"
            onClick={() => setPlaying((p) => !p)}
            disabled={finished}
            className="rounded-md border border-hairline px-3 py-1 font-mono text-[11px] text-faint hover:bg-bone disabled:cursor-not-allowed disabled:opacity-40"
          >
            {activePlaying ? (zh ? '暂停' : 'Pause') : zh ? '自动播放' : 'Auto'}
          </button>
          <button
            type="button"
            onClick={reset}
            className="rounded-md border border-hairline px-3 py-1 font-mono text-[11px] text-faint hover:bg-bone"
          >
            {zh ? '重置' : 'Reset'}
          </button>
        </div>
      </div>

      {/* body */}
      <div className="grid md:grid-cols-[200px_1fr]">
        {/* left: loop diagram */}
        <div className="border-b border-hairline px-4 py-5 md:border-b-0 md:border-r">
          <p className="mb-3 pl-5 font-mono text-[11px] text-faint">while not done:</p>
          <div className="relative">
            {/* curved return arrow: Done? -> Think */}
            <svg
              className="absolute inset-y-2 left-0 w-4 text-faint"
              viewBox="0 0 16 100"
              preserveAspectRatio="none"
              fill="none"
              aria-hidden="true"
            >
              <path
                d="M14 90 C 2 72, 2 28, 11 11"
                stroke="currentColor"
                strokeWidth="1"
                vectorEffect="non-scaling-stroke"
              />
              <path
                d="M6 16 L11 11 L10 19"
                stroke="currentColor"
                strokeWidth="1"
                vectorEffect="non-scaling-stroke"
              />
            </svg>
            <div className="pl-5">
              {STAGES.map((stage, i) => {
                const state = stageState(stage);
                const cls =
                  state === 'current'
                    ? 'border-ink-blue bg-pale-blue text-ink-blue'
                    : state === 'visited'
                      ? 'border-hairline bg-bone text-faint'
                      : 'border-hairline bg-paper text-faint';
                return (
                  <div key={stage}>
                    {i > 0 && <DownArrow />}
                    <div
                      className={`flex items-center justify-between gap-2 rounded-md border px-3 py-2 font-mono text-[12px] ${cls}`}
                    >
                      <span>{zh ? STAGE_LABELS[stage].zh : STAGE_LABELS[stage].en}</span>
                      {state === 'visited' && <CheckIcon />}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* right: transcript */}
        <div>
          <div ref={logRef} className="flex max-h-[340px] flex-col gap-2 overflow-y-auto px-6 py-4">
            <p className="font-mono text-[11px] text-faint">
              {zh
                ? 'task: 查询北京今天的天气，换算成华氏度后写入 report.txt'
                : 'task: Get today’s weather in Beijing, convert to °F, write it to report.txt'}
            </p>
            {step === 0 && (
              <p className="text-[12.5px] text-faint">
                {zh
                  ? '还没有任何动作。点「下一步」开始第一轮思考。'
                  : 'Nothing yet. Click “Next” to start the first Think step.'}
              </p>
            )}
            {SCRIPT.slice(0, step).map((entry, i) => {
              const text = zh ? entry.zh : entry.en;
              switch (entry.kind) {
                case 'think':
                  return (
                    <p key={i} className="font-serif text-[13px] italic text-faint">
                      {text}
                    </p>
                  );
                case 'act':
                  return (
                    <div
                      key={i}
                      className="rounded-md bg-bone px-3 py-2 font-mono text-[12px] text-ink"
                    >
                      tool: {text}
                    </div>
                  );
                case 'observe':
                  return (
                    <div
                      key={i}
                      className="rounded-md bg-pale-yellow px-3 py-2 font-mono text-[12px] text-ink-yellow"
                    >
                      {text}
                    </div>
                  );
                case 'check':
                  return (
                    <p key={i} className="font-mono text-[11px] text-faint">
                      {text}
                    </p>
                  );
                case 'final':
                  return (
                    <div
                      key={i}
                      className="rounded-md bg-pale-green px-3 py-2 font-serif text-[13px] text-ink-green"
                    >
                      {text}
                    </div>
                  );
              }
            })}
          </div>

          {/* memory strip */}
          <div className="border-t border-hairline px-6 py-3 font-mono text-[11px] text-faint">
            {`messages: ${1 + step} · tools: weather.get, calc.eval, fs.write`}
          </div>
        </div>
      </div>

      {/* footer */}
      <div className="border-t border-hairline bg-bone/50 px-6 py-3 text-[12.5px] text-faint">
        {zh
          ? '这就是所有 Agent 框架的全部秘密：一个 while 循环，LLM 决定调用哪个工具，把结果塞回上下文，直到它说『完成』。'
          : 'That is the entire secret of every agent framework: a while loop — the LLM picks a tool, the result is stuffed back into context, until it says “done”.'}
      </div>
    </section>
  );
}
