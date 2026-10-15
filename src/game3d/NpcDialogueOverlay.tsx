import {
  ArrowRight,
  BookOpen,
  DoorOpen,
  LoaderCircle,
  MonitorCog,
  Send,
  X,
} from 'lucide-react';
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from 'react';
import { Link } from 'react-router-dom';
import {
  buildResearchMentorContext,
  lessonsForPath,
  recommendResearchLessons,
  researchLearningPaths,
  type CurriculumLessonRef,
} from '../game/researchCurriculum';
import type { AgentId, GameSceneId } from '../game/types';
import { agents } from '../game/worldContent';
import { useAuth } from '../lib/auth';
import { fetchIndex } from '../lib/data';
import type { ZoneId } from '../lib/researchGame';
import {
  askTutor,
  saveTutorMessages,
  tutorEnabled,
  type ChatMessage,
  type TutorContext,
} from '../lib/tutor';
import type { CourseIndex } from '../lib/types';

type Turn = { role: 'user' | 'assistant'; content: string };
type DialogueMode = 'intro' | 'conversation';

export function NpcDialogueOverlay({
  agentId,
  reducedMotion,
  onWorkspace,
  onTravel,
  onClose,
}: {
  agentId: AgentId;
  reducedMotion: boolean;
  onWorkspace: (zoneId: ZoneId) => void;
  onTravel: (sceneId: GameSceneId) => void;
  onClose: () => void;
}) {
  const agent = agents[agentId];
  const [mode, setMode] = useState<DialogueMode>('intro');
  const [visibleCharacters, setVisibleCharacters] = useState(
    reducedMotion ? agent.greeting.length : 0,
  );

  useEffect(() => {
    if (reducedMotion || mode !== 'intro') return;
    const timer = window.setInterval(() => {
      setVisibleCharacters((current) => {
        if (current >= agent.greeting.length) {
          window.clearInterval(timer);
          return current;
        }
        return Math.min(agent.greeting.length, current + 2);
      });
    }, 24);
    return () => window.clearInterval(timer);
  }, [agent.greeting, mode, reducedMotion]);

  return (
    <div
      className="npc-dialogue-layer absolute inset-0 z-40"
      role="dialog"
      aria-modal="true"
      aria-label={`与${agent.name}交谈`}
      data-dialogue-mode={mode}
    >
      <div className="npc-dialogue-vignette pointer-events-none absolute inset-0" />
      <section className="npc-dialogue-box absolute inset-x-3 bottom-3 sm:inset-x-[5vw] sm:bottom-7 lg:inset-x-[9vw]">
        <img className="npc-dialogue-portrait" src={agent.asset} alt="" />
        <div className="npc-dialogue-content">
          <div className="npc-dialogue-nameplate">
            <div className="min-w-0">
              <strong>{agent.name}</strong>
              <span>{agent.role}</span>
            </div>
            <button
              type="button"
              className="npc-dialogue-close"
              onClick={onClose}
              aria-label="结束交谈"
              title="结束交谈"
            >
              <X size={17} aria-hidden="true" />
            </button>
          </div>

          {mode === 'intro' ? (
            <>
              <p className="npc-dialogue-line">
                {agent.greeting.slice(0, visibleCharacters)}
                {visibleCharacters < agent.greeting.length && <span className="npc-dialogue-caret" />}
              </p>
              <div className="npc-dialogue-actions">
                <button type="button" className="npc-dialogue-secondary" onClick={onClose}>
                  <X size={16} aria-hidden="true" />
                  离开
                </button>
                <button
                  type="button"
                  className="npc-dialogue-primary"
                  onClick={() => setMode('conversation')}
                >
                  {continueLabel(agentId)}
                  <ArrowRight size={17} aria-hidden="true" />
                </button>
              </div>
            </>
          ) : agentId === 'research-mentor' ? (
            <ResearchMentorDialogue key={agentId} />
          ) : (
            <AgentConversation
              key={agentId}
              agentId={agentId}
              onWorkspace={onWorkspace}
              onTravel={onTravel}
            />
          )}
        </div>
      </section>
    </div>
  );
}

function AgentConversation({
  agentId,
  onWorkspace,
  onTravel,
}: {
  agentId: Exclude<AgentId, 'research-mentor'>;
  onWorkspace: (zoneId: ZoneId) => void;
  onTravel: (sceneId: GameSceneId) => void;
}) {
  const { profile } = useAuth();
  const agent = agents[agentId];
  const [input, setInput] = useState('');
  const [turns, setTurns] = useState<Turn[]>([
    { role: 'assistant', content: agent.greeting },
  ]);
  const [stream, setStream] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const transcriptRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const transcript = transcriptRef.current;
    if (transcript) transcript.scrollTop = transcript.scrollHeight;
  }, [stream, turns]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const question = input.trim();
    if (!question || busy) return;
    const history: ChatMessage[] = turns.map((turn) => ({ ...turn }));
    setInput('');
    setTurns((current) => [...current, { role: 'user', content: question }]);
    setBusy(true);
    setError(null);
    setStream('');

    if (!tutorEnabled) {
      const answer = `${agent.focus} 先把你现在的目标写成一个可以确认的下一步，我会据此继续追问。`;
      setTurns((current) => [...current, { role: 'assistant', content: answer }]);
      setBusy(false);
      return;
    }

    const context: TutorContext = {
      lessonId: `research-game/${agent.id}`,
      title: agent.role,
      text: [
        `你是${agent.name}，身份是${agent.role}。`,
        agent.focus,
        '回答保持简短、具体，并延续当前对话。',
        '不得直接接受假设、启动实验、修改测量值或公开产物。',
        '任何 ResearchRun 写操作都必须由玩家明确确认。',
      ].join('\n'),
    };
    try {
      let full = '';
      await askTutor(question, history, context, null, 'zh', {
        assistantId: agent.id,
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
      setError(caught instanceof Error ? caught.message : 'Agent 暂时无法回答');
      setStream('');
    } finally {
      setBusy(false);
    }
  };

  const actions = dialogueActions(agentId, onWorkspace, onTravel);
  return (
    <div className="npc-dialogue-conversation">
      <div ref={transcriptRef} className="npc-dialogue-transcript" aria-live="polite">
        {turns.map((turn, index) => (
          <DialogueMessage key={`${turn.role}-${index}`} turn={turn} />
        ))}
        {stream && <DialogueMessage turn={{ role: 'assistant', content: stream }} streaming />}
        {busy && !stream && (
          <div className="npc-dialogue-thinking">
            <LoaderCircle size={14} className="animate-spin" aria-hidden="true" />
            {agent.name}正在思考
          </div>
        )}
      </div>

      {actions.length > 0 && (
        <div className="npc-dialogue-choices" aria-label="对话行动">
          {actions.map((action) => (
            <button key={action.label} type="button" onClick={action.onClick}>
              {action.kind === 'travel' ? <DoorOpen size={15} /> : <MonitorCog size={15} />}
              {action.label}
            </button>
          ))}
        </div>
      )}

      <form onSubmit={(event) => void submit(event)} className="npc-dialogue-form">
        <label htmlFor={`npc-question-${agentId}`} className="sr-only">
          向{agent.name}提问
        </label>
        <input
          id={`npc-question-${agentId}`}
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder={`向${agent.name}继续提问…`}
          maxLength={1000}
        />
        <button type="submit" disabled={busy || input.trim().length < 2} aria-label="发送问题">
          <Send size={17} aria-hidden="true" />
        </button>
      </form>
      {error && <p className="npc-dialogue-error">{error}</p>}
    </div>
  );
}

function ResearchMentorDialogue() {
  const { profile } = useAuth();
  const [index, setIndex] = useState<CourseIndex | null>(null);
  const [pathId, setPathId] = useState(researchLearningPaths[0].id);
  const [query, setQuery] = useState('');
  const [recommendations, setRecommendations] = useState<CurriculumLessonRef[]>([]);
  const [turns, setTurns] = useState<Turn[]>([
    { role: 'assistant', content: agents['research-mentor'].greeting },
  ]);
  const [stream, setStream] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const transcriptRef = useRef<HTMLDivElement | null>(null);
  const path = researchLearningPaths.find((item) => item.id === pathId) ?? researchLearningPaths[0];
  const pathLessons = useMemo(() => (index ? lessonsForPath(index, path) : []), [index, path]);

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

  useEffect(() => {
    const transcript = transcriptRef.current;
    if (transcript) transcript.scrollTop = transcript.scrollHeight;
  }, [stream, turns]);

  const selectPath = (nextPathId: string) => {
    const nextPath = researchLearningPaths.find((item) => item.id === nextPathId);
    if (!nextPath) return;
    setPathId(nextPathId);
    if (index) setRecommendations(lessonsForPath(index, nextPath));
  };

  const ask = async (event: FormEvent) => {
    event.preventDefault();
    const question = query.trim();
    if (!question || !index || busy) return;
    const found = recommendResearchLessons(index, question);
    const history: ChatMessage[] = turns.map((turn) => ({ ...turn }));
    setRecommendations(found);
    setTurns((current) => [...current, { role: 'user', content: question }]);
    setQuery('');
    setBusy(true);
    setError(null);
    setStream('');

    if (!tutorEnabled) {
      const names = found.slice(0, 3).map((item) => `《${item.title}》`).join('、');
      const answer = `先把问题缩小到一个可验证动作。建议先看${names || '第一次完整科研循环'}，再写下研究对象、可改变的变量、主指标，以及什么结果会推翻你的判断。`;
      setTurns((current) => [...current, { role: 'assistant', content: answer }]);
      setBusy(false);
      return;
    }

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const context = await buildResearchMentorContext(index, question, found);
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
      setStream('');
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  };

  return (
    <div className="npc-dialogue-mentor">
      <div className="npc-dialogue-mentor-main">
        <div ref={transcriptRef} className="npc-dialogue-transcript" aria-live="polite">
          {turns.map((turn, turnIndex) => (
            <DialogueMessage key={`${turn.role}-${turnIndex}`} turn={turn} />
          ))}
          {stream && <DialogueMessage turn={{ role: 'assistant', content: stream }} streaming />}
          {busy && !stream && (
            <div className="npc-dialogue-thinking">
              <LoaderCircle size={14} className="animate-spin" aria-hidden="true" />
              苏老师正在检索课程并组织回答
            </div>
          )}
        </div>
        <form onSubmit={(event) => void ask(event)} className="npc-dialogue-form">
          <label htmlFor="mentor-dialogue-question" className="sr-only">向苏老师提问</label>
          <input
            id="mentor-dialogue-question"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="告诉苏老师你卡在科研的哪一步…"
            maxLength={1000}
          />
          <button
            type="submit"
            disabled={!index || busy || query.trim().length < 2}
            aria-label="发送问题"
          >
            <Send size={17} aria-hidden="true" />
          </button>
        </form>
        {error && <p className="npc-dialogue-error">{error}</p>}
      </div>

      <aside className="npc-dialogue-curriculum" aria-label="科研训练课程建议">
        <div className="npc-dialogue-curriculum-heading">
          <span><BookOpen size={14} />科研训练路径</span>
          <small>{index?.stats.lessons ?? '—'} 节课程</small>
        </div>
        <div className="npc-dialogue-paths">
          {researchLearningPaths.map((item) => (
            <button
              key={item.id}
              type="button"
              aria-pressed={item.id === path.id}
              onClick={() => selectPath(item.id)}
            >
              {item.name}
            </button>
          ))}
        </div>
        <p className="npc-dialogue-path-description">{path.description}</p>
        <div className="npc-dialogue-lessons">
          {(recommendations.length > 0 ? recommendations : pathLessons).slice(0, 4).map((lesson) => (
            <Link key={lesson.id} to={`/lesson/${lesson.phaseSlug}/${lesson.lessonSlug}`}>
              <span>{lesson.title}</span>
              <small>{lesson.phaseTitle}</small>
            </Link>
          ))}
        </div>
      </aside>
    </div>
  );
}

function DialogueMessage({ turn, streaming = false }: { turn: Turn; streaming?: boolean }) {
  return (
    <div className={`npc-dialogue-message npc-dialogue-message--${turn.role}`}>
      <span>{turn.role === 'assistant' ? 'NPC' : '你'}</span>
      <p>
        {turn.content}
        {streaming && <span className="npc-dialogue-caret" />}
      </p>
    </div>
  );
}

function dialogueActions(
  agentId: Exclude<AgentId, 'research-mentor'>,
  onWorkspace: (zoneId: ZoneId) => void,
  onTravel: (sceneId: GameSceneId) => void,
) {
  if (agentId === 'campus-guide') {
    return [
      ['科研训练馆', 'academy'],
      ['模型与系统实验室', 'research'],
      ['算力工坊', 'compute'],
      ['论文与评审工坊', 'paper'],
      ['学术交流广场', 'plaza'],
    ].map(([label, sceneId]) => ({
      kind: 'travel' as const,
      label,
      onClick: () => onTravel(sceneId as GameSceneId),
    }));
  }
  if (agentId === 'methods-mentor') {
    return [
      { kind: 'workspace' as const, label: '项目与数据终端', onClick: () => onWorkspace('observatory') },
      { kind: 'workspace' as const, label: '研究假设白板', onClick: () => onWorkspace('lab') },
    ];
  }
  const zone = agents[agentId].workspaceZone;
  return zone
    ? [{ kind: 'workspace' as const, label: workspaceLabel(zone), onClick: () => onWorkspace(zone) }]
    : [];
}

function workspaceLabel(zoneId: ZoneId): string {
  const labels: Record<ZoneId, string> = {
    observatory: '项目与数据终端',
    lab: '研究假设白板',
    compute: '离线实验集群',
    paper: '分析与复现稿件',
    plaza: '验证研究海报板',
  };
  return labels[zoneId];
}

function continueLabel(agentId: AgentId): string {
  if (agentId === 'research-mentor') return '开始科研训练';
  if (agentId === 'campus-guide') return '讨论科研路线';
  if (agentId === 'reviewer') return '接受审稿质询';
  return '继续交谈';
}
