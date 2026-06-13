#!/usr/bin/env node
/**
 * Content pipeline for ai-eng-studio.
 * Scans the upstream ai-engineering-from-scratch repo plus local zh overlays
 * and emits static JSON consumed by the SPA:
 *   public/data/index.json                      global catalog + phase DAG
 *   public/data/lessons/<phase>/<lesson>.json   full lesson payloads
 *   public/data/glossary.json                   glossary terms (en + zh overlay)
 *
 * Run: node scripts/build-content.mjs
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const APP_ROOT = path.resolve(__dirname, '..');
const UPSTREAM = path.resolve(APP_ROOT, '..', 'ai-engineering-from-scratch');
const PHASES_DIR = path.join(UPSTREAM, 'phases');
const ROADMAP_PATH = path.join(UPSTREAM, 'ROADMAP.md');
const GLOSSARY_PATH = path.join(UPSTREAM, 'glossary', 'terms.md');
const ZH_DIR = path.join(APP_ROOT, 'content', 'zh');
const CHALLENGE_DIR = path.join(APP_ROOT, 'content', 'challenges');
const OUT_DIR = path.join(APP_ROOT, 'public', 'data');

const CODE_EXTS = new Set(['.py', '.ts', '.tsx', '.js', '.rs', '.jl', '.json', '.sh', '.yml', '.yaml', '.md', '.toml']);
const MAX_CODE_BYTES = 60_000;

// Capstones excluded because they may not run on the target machine —
// NVIDIA DGX Spark (aarch64 Linux, single Blackwell GB10 GPU). Audited
// 2026-06-12 against each lesson's Stack/code (see plan/task_plan.md):
// faster-whisper/CTranslate2 has no aarch64 CUDA wheels, Flash-Attention 3
// is Hopper-only, TensorFlow GPU on ARM is container-only, and FP8/Marlin
// kernels for sm_121 are not reliably packaged yet.
const EXCLUDED_LESSONS = new Set([
  '19-capstone-projects/03-realtime-voice-assistant',
  '19-capstone-projects/07-end-to-end-fine-tuning-pipeline',
  '19-capstone-projects/12-video-understanding-pipeline',
  '19-capstone-projects/14-speculative-decoding-server',
]);

// Phase metadata: zh titles/descriptions plus the dependency DAG from README.md.
const PHASE_META = {
  '00-setup-and-tooling':        { num: 0,  zh: '环境与工具链',        en: 'Setup & Tooling',              zhDesc: '开发环境、Python 生态、Docker 与实验工具链', deps: [] },
  '01-math-foundations':         { num: 1,  zh: '数学基础',            en: 'Math Foundations',             zhDesc: '线性代数、微积分、概率与优化——AI 的地基', deps: [0] },
  '02-ml-fundamentals':          { num: 2,  zh: '机器学习基础',        en: 'ML Fundamentals',              zhDesc: '从线性回归到集成学习，手写经典算法', deps: [1] },
  '03-deep-learning-core':       { num: 3,  zh: '深度学习核心',        en: 'Deep Learning Core',           zhDesc: '神经网络、反向传播、训练技巧，全部从零实现', deps: [2] },
  '04-computer-vision':          { num: 4,  zh: '计算机视觉',          en: 'Computer Vision',              zhDesc: '卷积网络、检测、分割与现代视觉模型', deps: [3] },
  '05-nlp-foundations-to-advanced': { num: 5, zh: '自然语言处理',      en: 'NLP Foundations to Advanced',  zhDesc: '从词向量到序列模型的语言处理全路径', deps: [3] },
  '06-speech-and-audio':         { num: 6,  zh: '语音与音频',          en: 'Speech & Audio',               zhDesc: '语音识别、合成与音频信号处理', deps: [3] },
  '07-transformers-deep-dive':   { num: 7,  zh: 'Transformer 深入',    en: 'Transformers Deep Dive',       zhDesc: '注意力机制、位置编码与 Transformer 全family解剖', deps: [5] },
  '08-generative-ai':            { num: 8,  zh: '生成式 AI',           en: 'Generative AI',                zhDesc: '扩散模型、VAE、GAN 与多模态生成', deps: [7] },
  '09-reinforcement-learning':   { num: 9,  zh: '强化学习',            en: 'Reinforcement Learning',       zhDesc: '从 Bandit 到 PPO，决策与策略学习', deps: [3] },
  '10-llms-from-scratch':        { num: 10, zh: '从零构建 LLM',        en: 'LLMs from Scratch',            zhDesc: '分词器、预训练、SFT、RLHF——亲手造一个 GPT', deps: [7] },
  '11-llm-engineering':          { num: 11, zh: 'LLM 工程',            en: 'LLM Engineering',              zhDesc: '提示工程、RAG、微调与推理优化的工程实践', deps: [10] },
  '12-multimodal-ai':            { num: 12, zh: '多模态 AI',           en: 'Multimodal AI',                zhDesc: '视觉-语言模型与跨模态对齐', deps: [10] },
  '13-tools-and-protocols':      { num: 13, zh: '工具与协议',          en: 'Tools & Protocols',            zhDesc: 'Function calling、MCP 与工具生态', deps: [11] },
  '14-agent-engineering':        { num: 14, zh: 'Agent 工程',          en: 'Agent Engineering',            zhDesc: 'Agent 循环、记忆、规划与可靠性工程', deps: [13] },
  '15-autonomous-systems':       { num: 15, zh: '自治系统',            en: 'Autonomous Systems',           zhDesc: '长时程自治、自我改进与安全护栏', deps: [14] },
  '16-multi-agent-and-swarms':   { num: 16, zh: '多智能体与集群',      en: 'Multi-Agent & Swarms',         zhDesc: '协作、竞争与大规模智能体编排', deps: [15] },
  '17-infrastructure-and-production': { num: 17, zh: '基础设施与生产', en: 'Infrastructure & Production',  zhDesc: '部署、监控、扩缩容与成本工程', deps: [14] },
  '18-ethics-safety-alignment':  { num: 18, zh: '伦理、安全与对齐',    en: 'Ethics, Safety & Alignment',   zhDesc: '对齐技术、红队测试与负责任的 AI', deps: [15] },
  '19-capstone-projects':        { num: 19, zh: '毕业项目',            en: 'Capstone Projects',            zhDesc: '端到端综合项目，检验全部所学', deps: [16, 17, 18] },
};

function readIfExists(p) {
  try { return fs.readFileSync(p, 'utf8'); } catch { return null; }
}

function readJsonIfExists(p) {
  const raw = readIfExists(p);
  if (raw == null) return null;
  try { return JSON.parse(raw); } catch (err) {
    console.warn(`  ! bad JSON skipped: ${p} (${err.message})`);
    return null;
  }
}

// ─── Lesson markdown parsing ─────────────────────────────────────────
function parseLessonMd(md) {
  const lines = md.split(/\r?\n/);
  let title = null;
  let quote = null;
  const meta = {};

  for (const line of lines.slice(0, 40)) {
    if (title == null) {
      const m = line.match(/^#\s+(.+)/);
      if (m) { title = m[1].trim(); continue; }
    }
    if (quote == null) {
      const m = line.match(/^>\s*(.+)/);
      if (m) { quote = m[1].trim(); continue; }
    }
    const metaMatch = line.match(/^\*\*(Type|Languages|Prerequisites|Time):\*\*\s*(.+)/);
    if (metaMatch) meta[metaMatch[1].toLowerCase()] = metaMatch[2].trim();
  }

  return { title: title ?? 'Untitled', quote, meta, body: md };
}

function listCodeFiles(codeDir) {
  if (!fs.existsSync(codeDir)) return [];
  const files = [];
  const walk = (dir, rel) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const abs = path.join(dir, entry.name);
      const relPath = rel ? `${rel}/${entry.name}` : entry.name;
      if (entry.isDirectory()) { walk(abs, relPath); continue; }
      const ext = path.extname(entry.name);
      if (!CODE_EXTS.has(ext) && entry.name !== 'Dockerfile') continue;
      const stat = fs.statSync(abs);
      if (stat.size > MAX_CODE_BYTES) continue;
      files.push({ name: relPath, lang: langOf(entry.name), content: fs.readFileSync(abs, 'utf8') });
    }
  };
  walk(codeDir, '');
  // Python first, then alphabetical: matches the curriculum's primary language.
  const order = { python: 0, typescript: 1, rust: 2, julia: 3 };
  files.sort((a, b) => (order[a.lang] ?? 9) - (order[b.lang] ?? 9) || a.name.localeCompare(b.name));
  return files;
}

function langOf(name) {
  const ext = path.extname(name);
  return {
    '.py': 'python', '.ts': 'typescript', '.tsx': 'typescript', '.js': 'javascript',
    '.rs': 'rust', '.jl': 'julia', '.json': 'json', '.sh': 'bash',
    '.yml': 'yaml', '.yaml': 'yaml', '.md': 'markdown', '.toml': 'toml',
  }[ext] ?? (name === 'Dockerfile' ? 'docker' : 'text');
}

// Detect whether a python file can plausibly run in Pyodide:
// stdlib + numpy only, no file/network side effects.
const PYODIDE_OK_IMPORTS = new Set([
  'math', 'random', 'itertools', 'functools', 'collections', 'dataclasses', 'typing',
  'json', 're', 'time', 'statistics', 'heapq', 'bisect', 'copy', 'enum', 'abc',
  'string', 'textwrap', 'operator', 'fractions', 'decimal', 'numpy',
]);

function pyodideRunnable(content) {
  const imports = [...content.matchAll(/^\s*(?:from\s+([\w.]+)|import\s+([\w.]+(?:\s*,\s*[\w.]+)*))/gm)]
    .flatMap((m) => (m[1] ? [m[1]] : m[2].split(/\s*,\s*/)))
    .map((s) => s.split('.')[0]);
  if (imports.length === 0) return true;
  return imports.every((mod) => PYODIDE_OK_IMPORTS.has(mod));
}

// ─── Glossary parsing ────────────────────────────────────────────────
function parseGlossary(md) {
  const terms = [];
  let current = null;
  for (const line of md.split(/\r?\n/)) {
    const termMatch = line.match(/^###\s+(.+)/);
    if (termMatch) {
      current = { term: termMatch[1].trim(), saying: '', meaning: '', origin: '' };
      terms.push(current);
      continue;
    }
    if (!current) continue;
    const field = line.match(/^-\s+\*\*(.+?):\*\*\s*(.+)/);
    if (field) {
      const key = field[1].toLowerCase();
      if (key.includes('people say')) current.saying = field[2].trim();
      else if (key.includes('actually means')) current.meaning = field[2].trim();
      else if (key.includes('called that')) current.origin = field[2].trim();
    }
  }
  return terms;
}

// ─── ROADMAP hour estimates ──────────────────────────────────────────
// Headings look like: ## Phase 7: Transformers Deep Dive — ✅ (~14 hours)
function parseRoadmapHours() {
  const md = readIfExists(ROADMAP_PATH);
  const hours = {};
  if (!md) return hours;
  for (const m of md.matchAll(/^##\s+Phase\s+(\d+):.*?\(~(\d+)\s*hours?\)/gm)) {
    hours[Number(m[1])] = Number(m[2]);
  }
  return hours;
}

// ─── Main build ──────────────────────────────────────────────────────
function build() {
  const roadmapHours = parseRoadmapHours();
  fs.rmSync(OUT_DIR, { recursive: true, force: true });
  fs.mkdirSync(path.join(OUT_DIR, 'lessons'), { recursive: true });

  const phaseSlugs = fs.readdirSync(PHASES_DIR).filter((d) => PHASE_META[d]);
  const phases = [];
  let lessonCount = 0;
  let zhCount = 0;
  let quizCount = 0;

  for (const phaseSlug of phaseSlugs.sort()) {
    const phaseDir = path.join(PHASES_DIR, phaseSlug);
    const meta = PHASE_META[phaseSlug];
    const zhTitles = readJsonIfExists(path.join(ZH_DIR, phaseSlug, 'titles.json')) ?? {};
    const lessons = [];

    const lessonSlugs = fs.readdirSync(phaseDir, { withFileTypes: true })
      .filter((e) => e.isDirectory())
      .map((e) => e.name)
      .sort((a, b) => a.localeCompare(b, 'en', { numeric: true }));

    for (const lessonSlug of lessonSlugs) {
      if (EXCLUDED_LESSONS.has(`${phaseSlug}/${lessonSlug}`)) continue;
      const lessonDir = path.join(phaseDir, lessonSlug);
      const enMd = readIfExists(path.join(lessonDir, 'docs', 'en.md'));
      if (enMd == null) continue;

      const parsed = parseLessonMd(enMd);
      const zhMd = readIfExists(path.join(ZH_DIR, phaseSlug, `${lessonSlug}.md`));
      const quizEn = readJsonIfExists(path.join(lessonDir, 'quiz.json'));
      const quizZh = readJsonIfExists(path.join(ZH_DIR, phaseSlug, `${lessonSlug}.quiz.json`));
      const codeFiles = listCodeFiles(path.join(lessonDir, 'code'));

      const id = `${phaseSlug}/${lessonSlug}`;
      lessonCount += 1;
      if (zhMd) zhCount += 1;
      if (quizEn) quizCount += 1;

      lessons.push({
        slug: lessonSlug,
        title: parsed.title,
        titleZh: zhTitles[lessonSlug] ?? null,
        type: parsed.meta.type ?? null,
        time: parsed.meta.time ?? null,
        languages: parsed.meta.languages?.split(/\s*,\s*/) ?? [],
        hasQuiz: Boolean(quizEn),
        hasZh: Boolean(zhMd),
        runnable: codeFiles.some((f) => f.lang === 'python' && pyodideRunnable(f.content)),
      });

      const payload = {
        id,
        phase: phaseSlug,
        slug: lessonSlug,
        title: parsed.title,
        titleZh: zhTitles[lessonSlug] ?? null,
        quote: parsed.quote,
        meta: parsed.meta,
        bodyEn: parsed.body,
        bodyZh: zhMd,
        quizEn: quizEn?.questions ?? null,
        quizZh: quizZh?.questions ?? null,
        code: codeFiles.map((f) => ({
          ...f,
          runnable: f.lang === 'python' ? pyodideRunnable(f.content) : false,
        })),
        challenge: readJsonIfExists(path.join(CHALLENGE_DIR, phaseSlug, `${lessonSlug}.json`)),
      };

      const outPhaseDir = path.join(OUT_DIR, 'lessons', phaseSlug);
      fs.mkdirSync(outPhaseDir, { recursive: true });
      fs.writeFileSync(path.join(outPhaseDir, `${lessonSlug}.json`), JSON.stringify(payload));
    }

    phases.push({
      slug: phaseSlug,
      num: meta.num,
      titleEn: meta.en,
      titleZh: meta.zh,
      descZh: meta.zhDesc,
      deps: meta.deps,
      // ROADMAP is canonical; if a phase is missing there (e.g. Phase 13),
      // fall back to summing the per-lesson "~N minutes" estimates.
      hours:
        roadmapHours[meta.num] ??
        Math.round(
          lessons.reduce((acc, l) => acc + (Number(/(\d+)\s*min/.exec(l.time ?? '')?.[1]) || 60), 0) / 60,
        ),
      lessons,
    });
  }

  phases.sort((a, b) => a.num - b.num);
  const index = {
    generatedAt: new Date().toISOString(),
    stats: { phases: phases.length, lessons: lessonCount, quizzes: quizCount, zhLessons: zhCount },
    phases,
  };
  fs.writeFileSync(path.join(OUT_DIR, 'index.json'), JSON.stringify(index));

  // Glossary
  const glossaryMd = readIfExists(GLOSSARY_PATH);
  const terms = glossaryMd ? parseGlossary(glossaryMd) : [];
  const zhGlossary = readJsonIfExists(path.join(ZH_DIR, 'glossary.json')) ?? {};
  for (const t of terms) {
    const zh = zhGlossary[t.term];
    if (zh) Object.assign(t, { zh });
  }
  fs.writeFileSync(path.join(OUT_DIR, 'glossary.json'), JSON.stringify({ terms }));

  console.log(`index: ${phases.length} phases, ${lessonCount} lessons (${quizCount} quizzes, ${zhCount} zh)`);
  console.log(`glossary: ${terms.length} terms (${Object.keys(zhGlossary).length} zh)`);
}

build();
