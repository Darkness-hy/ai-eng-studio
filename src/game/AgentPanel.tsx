import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type ReactNode,
} from 'react';
import { Link } from 'react-router-dom';
import { fetchIndex } from '../lib/data';
import {
  askTutor,
  saveTutorMessages,
  tutorEnabled,
  type ChatMessage,
  type TutorContext,
} from '../lib/tutor';
import { useAuth } from '../lib/auth';
import type { CourseIndex } from '../lib/types';
import { agents } from './worldContent';
import {
  buildResearchMentorContext,
  lessonsForPath,
  recommendResearchLessons,
  researchLearningPaths,
  type CurriculumLessonRef,
} from './researchCurriculum';
import type { AgentId, GameSceneId } from './types';
import type { ZoneId } from '../lib/researchGame';

type Turn = { role: 'user' | 'assistant'; content: string };

export function AgentPanel({
  agentId,
  onClose,
  onWorkspace,
  onTravel,
}: {
  agentId: AgentId;
  onClose: () => void;
  onWorkspace: (zoneId: ZoneId) => void;
  onTravel: (sceneId: GameSceneId) => void;
}) {
  const agent = agents[agentId];
  if (agentId === 'research-mentor') {
    return <ResearchMentorPanel key={agentId} onClose={onClose} />;
  }
  return (
    <PanelFrame agentId={agentId} onClose={onClose}>
      <div className="grid gap-7 lg:grid-cols-[minmax(0,1.15fr)_minmax(280px,0.85fr)]">
        <section>
          <p className="border-l-2 border-ink-blue pl-3 text-[15px] leading-relaxed">
            {agent.greeting}
          </p>
          <h3 className="mt-6 font-serif text-[19px] font-semibold">这个 Agent 负责什么</h3>
          <p className="mt-2 text-[13px] leading-relaxed text-faint">{agent.focus}</p>
          <SimpleAgentChat key={agentId} agentId={agentId} />
        </section>
        <aside className="border-t border-hairline pt-5 lg:border-l lg:border-t-0 lg:pl-7 lg:pt-0">
          <div className="font-mono text-[9.5px] tracking-[0.16em] text-faint">STRUCTURED ACTIONS</div>
          <div className="mt-4 grid gap-2">
            {agentId === 'campus-guide' && (
              <>
                <ActionButton label="去科研训练馆" onClick={() => onTravel('academy')} />
                <ActionButton label="去模型与系统实验室" onClick={() => onTravel('research')} />
              </>
            )}
            {agentId === 'methods-mentor' && (
              <>
                <ActionButton label="打开项目与数据终端" onClick={() => onWorkspace('observatory')} />
                <ActionButton label="打开假设白板" onClick={() => onWorkspace('lab')} />
              </>
            )}
            {agent.workspaceZone && (
              <ActionButton label="打开当前科研工作台" onClick={() => onWorkspace(agent.workspaceZone!)} />
            )}
          </div>
          <p className="mt-4 text-[11.5px] leading-relaxed text-faint">
            Agent 只提出建议和结构化动作。接受假设、运行实验和公开成果仍需要玩家确认。
          </p>
        </aside>
      </div>
    </PanelFrame>
  );
}

function ResearchMentorPanel({ onClose }: { onClose: () => void }) {
  const { profile } = useAuth();
  const [index, setIndex] = useState<CourseIndex | null>(null);
  const [pathId, setPathId] = useState(researchLearningPaths[0].id);
  const [query, setQuery] = useState('我没有系统做过科研，应该从哪里开始？');
  const [recommendations, setRecommendations] = useState<CurriculumLessonRef[]>([]);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [stream, setStream] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let live = true;
    fetchIndex()
      .then((value) => {
        if (!live) return;
        setIndex(value);
        setRecommendations(lessonsForPath(value, researchLearningPaths[0]));
      })
      .catch(() => {
        if (live) setError('课程目录读取失败');
      });
    return () => {
      live = false;
      abortRef.current?.abort();
    };
  }, []);

  const path = researchLearningPaths.find((item) => item.id === pathId) ?? researchLearningPaths[0];
  const pathLessons = useMemo(() => (index ? lessonsForPath(index, path) : []), [index, path]);

  const search = () => {
    if (!index) return;
    setRecommendations(recommendResearchLessons(index, query));
  };

  const ask = async (event: FormEvent) => {
    event.preventDefault();
    const question = query.trim();
    if (!question || !index || busy) return;
    const found = recommendResearchLessons(index, question);
    setRecommendations(found);
    setTurns((current) => [...current, { role: 'user', content: question }]);
    setBusy(true);
    setError(null);
    setStream('');
    if (!tutorEnabled) {
      const names = found.slice(0, 3).map((item) => `《${item.title}》`).join('、');
      const answer = `先把问题缩小到一个可验证动作。根据 503 节课程目录，建议先看 ${names || '第一次完整科研循环'}，然后写下：研究对象、可改变的变量、主指标，以及什么结果会推翻你的判断。`;
      setTurns((current) => [...current, { role: 'assistant', content: answer }]);
      setBusy(false);
      return;
    }
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const context = await buildResearchMentorContext(index, question, found);
      const history: ChatMessage[] = turns.map((turn) => ({ ...turn }));
      let full = '';
      await askTutor(question, history, context, null, 'zh', {
        assistantId: 'research-mentor',
        signal: controller.signal,
        onDelta: (delta) => {
          full += delta;
          setStream(full);
        },
      });
      setTurns((current) => [...current, { role: 'assistant', content: full }]);
      setStream('');
      if (profile && full) {
        void saveTutorMessages(profile.id, context.lessonId, [
          { role: 'user', content: question },
          { role: 'assistant', content: full },
        ]);
      }
    } catch (caught) {
      if (!(caught instanceof DOMException && caught.name === 'AbortError')) {
        setError(caught instanceof Error ? caught.message : '导师暂时无法回答');
      }
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  };

  return (
    <PanelFrame agentId="research-mentor" onClose={onClose} wide>
      <div className="grid gap-7 xl:grid-cols-[minmax(300px,0.8fr)_minmax(0,1.2fr)]">
        <section>
          <div className="font-mono text-[9.5px] tracking-[0.16em] text-faint">RESEARCH TRAINING PATH</div>
          <div className="mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-1">
            {researchLearningPaths.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setPathId(item.id)}
                className={`rounded-md border px-3 py-3 text-left transition-colors ${
                  item.id === path.id
                    ? 'border-ink bg-ink text-paper'
                    : 'border-hairline bg-bone text-ink hover:bg-paper'
                }`}
              >
                <span className="block text-[13px] font-medium">{item.name}</span>
                <span className={`mt-1 block text-[11.5px] leading-relaxed ${item.id === path.id ? 'text-paper/70' : 'text-faint'}`}>
                  {item.description}
                </span>
              </button>
            ))}
          </div>
          <div className="mt-5 border-t border-hairline">
            {pathLessons.map((lesson, indexInPath) => (
              <Link
                key={lesson.id}
                to={`/lesson/${lesson.phaseSlug}/${lesson.lessonSlug}`}
                className="flex items-start gap-3 border-b border-hairline py-3 text-[12.5px] transition-colors hover:text-ink-blue"
              >
                <span className="font-mono text-[9.5px] text-faint">{indexInPath + 1}</span>
                <span className="min-w-0">
                  <span className="block font-medium">{lesson.title}</span>
                  <span className="mt-0.5 block text-[10.5px] text-faint">{lesson.phaseTitle}</span>
                </span>
              </Link>
            ))}
          </div>
        </section>

        <section className="min-w-0 border-t border-hairline pt-6 xl:border-l xl:border-t-0 xl:pl-7 xl:pt-0">
          <div className="flex items-baseline justify-between gap-3">
            <div className="font-mono text-[9.5px] tracking-[0.16em] text-faint">CURRICULUM-GROUNDED MENTOR</div>
            <span className="font-mono text-[9.5px] text-faint">{index?.stats.lessons ?? '—'} LESSONS</span>
          </div>
          <div className="mt-4 max-h-52 overflow-y-auto border-y border-hairline py-3">
            {turns.length === 0 && !stream ? (
              <p className="text-[13px] leading-relaxed text-faint">
                苏老师会先判断你卡在科研循环的哪一步，再从课程目录检索相关内容。没有配置 AI 助教时仍会给出本地学习建议。
              </p>
            ) : (
              <div className="space-y-3 text-[13px] leading-relaxed">
                {turns.map((turn, indexInChat) => (
                  <div key={`${turn.role}-${indexInChat}`} className={turn.role === 'user' ? 'text-ink' : 'border-l-2 border-ink-blue pl-3 text-faint'}>
                    {turn.content}
                  </div>
                ))}
                {stream && <div className="border-l-2 border-ink-blue pl-3 text-faint">{stream}</div>}
              </div>
            )}
          </div>
          <form onSubmit={(event) => void ask(event)} className="mt-4">
            <label htmlFor="mentor-question" className="text-[12px] font-medium">
              你现在卡在哪一步？
            </label>
            <textarea
              id="mentor-question"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              minLength={2}
              maxLength={1000}
              className="mt-2 min-h-24 w-full resize-y rounded-md border border-hairline bg-paper px-3 py-3 text-[13px] leading-relaxed outline-none focus:border-ink-blue"
            />
            <div className="mt-3 flex flex-col gap-2 sm:flex-row">
              <button
                type="button"
                onClick={search}
                disabled={!index}
                className="min-h-10 rounded-md border border-hairline bg-bone px-4 text-[12.5px] font-medium hover:bg-paper disabled:opacity-50"
              >
                检索 503 节课程
              </button>
              <button
                type="submit"
                disabled={!index || busy || query.trim().length < 2}
                className="min-h-10 flex-1 rounded-md bg-ink px-4 text-[12.5px] font-medium text-paper disabled:bg-hairline disabled:text-faint"
              >
                {busy ? '导师正在思考……' : tutorEnabled ? '向导师 Agent 提问' : '获得本地导师建议'}
              </button>
            </div>
          </form>
          {error && <p className="mt-3 text-[12px] text-ink-red">{error}</p>}

          {recommendations.length > 0 && (
            <div className="mt-5">
              <h3 className="font-serif text-[17px] font-semibold">当前推荐课程</h3>
              <div className="mt-2 grid gap-px bg-hairline sm:grid-cols-2">
                {recommendations.slice(0, 6).map((lesson) => (
                  <Link
                    key={lesson.id}
                    to={`/lesson/${lesson.phaseSlug}/${lesson.lessonSlug}`}
                    className="min-w-0 bg-canvas px-3 py-3 text-[12px] hover:bg-bone"
                  >
                    <span className="block truncate font-medium">{lesson.title}</span>
                    <span className="mt-1 block truncate text-[10.5px] text-faint">{lesson.phaseTitle}</span>
                  </Link>
                ))}
              </div>
            </div>
          )}
        </section>
      </div>
    </PanelFrame>
  );
}

function SimpleAgentChat({ agentId }: { agentId: AgentId }) {
  const agent = agents[agentId];
  const [input, setInput] = useState('');
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const question = input.trim();
    if (!question || busy) return;
    setInput('');
    setTurns((current) => [...current, { role: 'user', content: question }]);
    if (!tutorEnabled) {
      setTurns((current) => [
        ...current,
        { role: 'assistant', content: `${agent.greeting} ${agent.focus}` },
      ]);
      return;
    }
    setBusy(true);
    setError(null);
    const context: TutorContext = {
      lessonId: `research-game/${agent.id}`,
      title: agent.role,
      text: [
        `你是${agent.name}，身份是${agent.role}。`,
        agent.focus,
        '回答保持简短、具体。不得直接接受假设、启动实验、修改测量值或公开产物。',
        '你只能提出建议，任何 ResearchRun 写操作必须由玩家明确确认。',
      ].join('\n'),
    };
    try {
      let full = '';
      await askTutor(question, turns as ChatMessage[], context, null, 'zh', {
        assistantId: agent.id,
        onDelta: (delta) => {
          full += delta;
        },
      });
      setTurns((current) => [...current, { role: 'assistant', content: full }]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Agent 暂时无法回答');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mt-6 border-t border-hairline pt-5">
      <div className="max-h-40 space-y-2 overflow-y-auto text-[12.5px] leading-relaxed">
        {turns.map((turn, index) => (
          <p key={`${turn.role}-${index}`} className={turn.role === 'assistant' ? 'border-l-2 border-hairline pl-3 text-faint' : ''}>
            {turn.content}
          </p>
        ))}
      </div>
      <form onSubmit={(event) => void submit(event)} className="mt-3 flex gap-2">
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="继续追问这个 Agent"
          className="min-h-10 min-w-0 flex-1 rounded-md border border-hairline bg-paper px-3 text-[12.5px] outline-none focus:border-ink-blue"
        />
        <button
          type="submit"
          disabled={busy || !input.trim()}
          className="min-h-10 rounded-md bg-ink px-4 text-[12px] font-medium text-paper disabled:opacity-40"
        >
          {busy ? '…' : '发送'}
        </button>
      </form>
      {error && <p className="mt-2 text-[11.5px] text-ink-red">{error}</p>}
    </div>
  );
}

function PanelFrame({
  agentId,
  onClose,
  wide = false,
  children,
}: {
  agentId: AgentId;
  onClose: () => void;
  wide?: boolean;
  children: ReactNode;
}) {
  const agent = agents[agentId];
  return (
    <div className="absolute inset-0 z-40 flex items-end justify-center bg-ink/35 p-0 backdrop-blur-[2px] sm:items-center sm:p-5">
      <div className={`max-h-[92dvh] w-full overflow-y-auto rounded-t-md bg-canvas shadow-lift sm:rounded-md ${wide ? 'max-w-6xl' : 'max-w-4xl'}`}>
        <header className="sticky top-0 z-10 flex min-h-20 items-center justify-between gap-4 overflow-visible border-b border-hairline bg-canvas/95 px-4 py-3 backdrop-blur-sm sm:px-6">
          <div className="flex min-w-0 items-center gap-4">
            <img src={agent.asset} alt="" className="-mt-8 h-28 w-20 shrink-0 self-end object-contain object-bottom drop-shadow-md" />
            <div className="min-w-0">
              <h2 className="truncate font-serif text-[19px] font-semibold">{agent.name}</h2>
              <p className="truncate text-[11.5px] text-faint">{agent.role}</p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md border border-hairline bg-paper text-[20px] text-faint hover:text-ink"
            aria-label="关闭对话"
          >
            ×
          </button>
        </header>
        <div className="px-4 py-6 sm:px-6">{children}</div>
      </div>
    </div>
  );
}

function ActionButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="min-h-10 rounded-md border border-hairline bg-bone px-3 text-left text-[12.5px] font-medium transition-colors hover:bg-paper"
    >
      {label}
    </button>
  );
}
