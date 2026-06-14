/** Canvas-rendered certificates and shareable achievement cards (PNG download). */
import QRCode from 'qrcode';

const INK = '#2f3437';
const FAINT = '#787774';
const BLUE = '#1f6c9f';
const CANVAS_BG = '#fbfbfa';
const HAIRLINE = '#d9d8d4';
const PINK = '#f2335d';
const GOLD = '#b8922e';
const GOLD_LT = '#d8b65a';

const VERIFY_BASE = 'https://darkness-hy.github.io/ai-eng-studio/';

async function ready(): Promise<void> {
  try {
    await Promise.all([
      document.fonts.load('600 50px Newsreader'),
      document.fonts.load('italic 600 28px Newsreader'),
      document.fonts.load('400 22px Newsreader'),
      document.fonts.load('600 16px "JetBrains Mono"'),
    ]);
    await document.fonts.ready;
  } catch {
    /* fall back to generic fonts */
  }
}

function setup(w: number, h: number): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } {
  const dpr = 2;
  const canvas = document.createElement('canvas');
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext('2d')!;
  ctx.scale(dpr, dpr);
  ctx.fillStyle = CANVAS_BG;
  ctx.fillRect(0, 0, w, h);
  ctx.textAlign = 'center';
  return { canvas, ctx };
}

function toBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((res, rej) =>
    canvas.toBlob((b) => (b ? res(b) : rej(new Error('生成图片失败'))), 'image/png'),
  );
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((res, rej) => {
    const i = new Image();
    i.onload = () => res(i);
    i.onerror = rej;
    i.src = src;
  });
}

/** Stable 8-char id from the certificate fields (for the QR + printed code). */
function certId(...parts: string[]): string {
  let h = 5381;
  const s = parts.join('|');
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) >>> 0;
  return ('AE' + h.toString(36).toUpperCase()).slice(0, 10);
}

function drawMark(ctx: CanvasRenderingContext2D, x: number, y: number, s: number) {
  ctx.save();
  ctx.translate(x, y);
  ctx.fillStyle = '#111111';
  ctx.beginPath();
  ctx.roundRect(0, 0, s, s, s * 0.22);
  ctx.fill();
  ctx.fillStyle = '#f2f1ee';
  const bar = (bx: number, by: number, bw: number, bh: number) => {
    ctx.beginPath();
    ctx.roundRect(bx, by, bw, bh, bh / 2);
    ctx.fill();
  };
  bar(s * 0.18, s * 0.34, s * 0.46, s * 0.07);
  bar(s * 0.18, s * 0.47, s * 0.32, s * 0.07);
  bar(s * 0.18, s * 0.6, s * 0.2, s * 0.07);
  ctx.fillStyle = PINK;
  ctx.beginPath();
  ctx.arc(s * 0.76, s * 0.63, s * 0.05, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawStar(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number) {
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const rad = i % 2 === 0 ? r : r * 0.45;
    const a = (Math.PI / 5) * i - Math.PI / 2;
    const px = cx + Math.cos(a) * rad;
    const py = cy + Math.sin(a) * rad;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
}

/** Gold wax-style seal with concentric rings, a star and circular text. */
function drawSeal(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number, zh: boolean) {
  ctx.save();
  ctx.strokeStyle = GOLD;
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeStyle = GOLD_LT;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(cx, cy, r - 9, 0, Math.PI * 2);
  ctx.stroke();

  // circular text top + bottom
  ctx.fillStyle = GOLD;
  const top = 'AI ENGINEERING FROM SCRATCH';
  const bottom = zh ? '· 已认证 · 已认证 ·' : '· VERIFIED · VERIFIED ·';
  const ring = (text: string, baseAngle: number, flip: boolean) => {
    ctx.font = '600 10px "JetBrains Mono", monospace';
    const step = (Math.PI * 1.25) / text.length;
    for (let i = 0; i < text.length; i++) {
      const a = baseAngle + (i - (text.length - 1) / 2) * step * (flip ? -1 : 1);
      ctx.save();
      ctx.translate(cx + Math.cos(a) * (r - 20), cy + Math.sin(a) * (r - 20));
      ctx.rotate(a + (flip ? -Math.PI / 2 : Math.PI / 2));
      ctx.textAlign = 'center';
      ctx.fillText(text[i], 0, 0);
      ctx.restore();
    }
  };
  ring(top, -Math.PI / 2, false);
  ring(bottom, Math.PI / 2, true);

  // center star + label
  ctx.fillStyle = GOLD;
  drawStar(ctx, cx, cy - 6, 16);
  ctx.fill();
  ctx.textAlign = 'center';
  ctx.font = '600 11px "JetBrains Mono", monospace';
  ctx.fillText(zh ? '结业认证' : 'CERTIFIED', cx, cy + 26);
  ctx.restore();
}

export interface CertOpts {
  name: string;
  achievement: string;
  date: string;
  zh: boolean;
}

/** Portrait certificate: gold seal, signature line, QR verification, cert id. */
export async function generateCertificate(o: CertOpts): Promise<Blob> {
  await ready();
  const W = 880;
  const H = 1240;
  const { canvas, ctx } = setup(W, H);
  const id = certId(o.name, o.achievement, o.date);

  // borders
  ctx.strokeStyle = INK;
  ctx.lineWidth = 2;
  ctx.strokeRect(40, 40, W - 80, H - 80);
  ctx.strokeStyle = HAIRLINE;
  ctx.lineWidth = 1;
  ctx.strokeRect(52, 52, W - 104, H - 104);

  ctx.fillStyle = FAINT;
  ctx.font = '600 15px "JetBrains Mono", monospace';
  ctx.fillText(o.zh ? '结业证书 · CERTIFICATE OF COMPLETION' : 'CERTIFICATE OF COMPLETION', W / 2, 150);

  ctx.fillStyle = INK;
  ctx.font = '600 46px Newsreader, serif';
  ctx.fillText(o.zh ? '从零开始的 AI 工程' : 'AI Engineering from Scratch', W / 2, 250);

  ctx.fillStyle = FAINT;
  ctx.font = '22px Newsreader, serif';
  ctx.fillText(o.zh ? '兹证明' : 'This certifies that', W / 2, 340);

  ctx.fillStyle = BLUE;
  ctx.font = '600 44px Newsreader, serif';
  ctx.fillText(o.name, W / 2, 412);

  ctx.fillStyle = INK;
  ctx.font = '23px Newsreader, serif';
  ctx.fillText(o.achievement, W / 2, 488);

  ctx.fillStyle = FAINT;
  ctx.font = '15px "JetBrains Mono", monospace';
  ctx.fillText(o.date, W / 2, 552);

  // gold seal
  drawSeal(ctx, W / 2, 700, 74, o.zh);

  // signature line (left) + verification QR (right)
  const sy = 920;
  const sigX = 150;
  const sigW = 250;
  ctx.fillStyle = INK;
  ctx.font = 'italic 600 30px Newsreader, serif';
  ctx.textAlign = 'center';
  ctx.fillText('LaViRA', sigX + sigW / 2, sy - 12);
  ctx.strokeStyle = INK;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(sigX, sy);
  ctx.lineTo(sigX + sigW, sy);
  ctx.stroke();
  ctx.fillStyle = FAINT;
  ctx.font = '11px "JetBrains Mono", monospace';
  ctx.fillText(o.zh ? '课程导师 · INSTRUCTOR' : 'COURSE INSTRUCTOR', sigX + sigW / 2, sy + 22);

  // QR (right)
  const qrDataUrl = await QRCode.toDataURL(`${VERIFY_BASE}?cert=${id}`, {
    margin: 0,
    width: 260,
    color: { dark: INK, light: CANVAS_BG },
  });
  const qrImg = await loadImage(qrDataUrl);
  const qx = W - 150 - 110;
  const qy = sy - 110;
  ctx.drawImage(qrImg, qx, qy, 110, 110);
  ctx.fillStyle = FAINT;
  ctx.font = '10px "JetBrains Mono", monospace';
  ctx.textAlign = 'center';
  ctx.fillText(o.zh ? '扫码验证' : 'SCAN TO VERIFY', qx + 55, qy + 128);

  // cert id
  ctx.fillStyle = FAINT;
  ctx.font = '12px "JetBrains Mono", monospace';
  ctx.fillText(`${o.zh ? '证书编号' : 'NO.'} ${id}`, W / 2, 1040);

  // brand footer
  drawMark(ctx, W / 2 - 100, 1090, 38);
  ctx.fillStyle = INK;
  ctx.font = '600 19px Newsreader, serif';
  ctx.textAlign = 'left';
  ctx.fillText('LaViRA', W / 2 - 50, 1115);
  ctx.fillStyle = FAINT;
  ctx.font = '11px "JetBrains Mono", monospace';
  ctx.textAlign = 'center';
  ctx.fillText('Language-Vision-Robot Actions Translation · 独家冠名', W / 2, 1165);

  return toBlob(canvas);
}

export interface ShareOpts {
  name: string;
  doneCount: number;
  totalLessons: number;
  streak: number;
  badges: number;
  zh: boolean;
}

export async function generateShareCard(o: ShareOpts): Promise<Blob> {
  await ready();
  const W = 800;
  const H = 800;
  const { canvas, ctx } = setup(W, H);

  ctx.strokeStyle = HAIRLINE;
  ctx.lineWidth = 1;
  ctx.strokeRect(40, 40, W - 80, H - 80);

  // course branding — kept in sync with the OG social card
  ctx.fillStyle = FAINT;
  ctx.font = '600 14px "JetBrains Mono", monospace';
  ctx.fillText('AI ENGINEERING · FROM SCRATCH · 中文交互版', W / 2, 96);

  ctx.fillStyle = INK;
  ctx.font = '600 34px Newsreader, serif';
  ctx.fillText(o.zh ? '从零开始的 AI 工程' : 'AI Engineering from Scratch', W / 2, 142);

  ctx.fillStyle = FAINT;
  ctx.font = '13px "JetBrains Mono", monospace';
  ctx.fillText(
    o.zh ? '从数学基础到可上线 · 20 阶段 · 503 课' : 'Math foundations → production · 20 phases · 503 lessons',
    W / 2,
    172,
  );

  ctx.strokeStyle = HAIRLINE;
  ctx.beginPath();
  ctx.moveTo(150, 204);
  ctx.lineTo(W - 150, 204);
  ctx.stroke();

  // personal achievement
  ctx.fillStyle = BLUE;
  ctx.font = '600 38px Newsreader, serif';
  ctx.fillText(o.name, W / 2, 268);
  ctx.fillStyle = FAINT;
  ctx.font = '17px Newsreader, serif';
  ctx.fillText(o.zh ? '的学习成就' : "'s learning so far", W / 2, 300);

  const stat = (x: number, big: string, label: string) => {
    ctx.fillStyle = INK;
    ctx.font = '600 50px Newsreader, serif';
    ctx.fillText(big, x, 408);
    ctx.fillStyle = FAINT;
    ctx.font = '15px "JetBrains Mono", monospace';
    ctx.fillText(label, x, 442);
  };
  stat(W / 2 - 230, `${o.doneCount}`, o.zh ? '完成课程' : 'lessons');
  stat(W / 2, `${o.streak}`, o.zh ? '连续天数' : 'day streak');
  stat(W / 2 + 230, `${o.badges}`, o.zh ? '解锁徽章' : 'badges');

  const pct = o.totalLessons ? o.doneCount / o.totalLessons : 0;
  const bx = 160;
  const bw = W - 320;
  ctx.fillStyle = '#eceae6';
  ctx.beginPath();
  ctx.roundRect(bx, 520, bw, 10, 5);
  ctx.fill();
  ctx.fillStyle = '#346538';
  ctx.beginPath();
  ctx.roundRect(bx, 520, Math.max(bw * pct, 6), 10, 5);
  ctx.fill();
  ctx.fillStyle = FAINT;
  ctx.font = '14px "JetBrains Mono", monospace';
  ctx.fillText(`${o.doneCount} / ${o.totalLessons} · ${Math.round(pct * 100)}%`, W / 2, 565);

  // CTA — drives traffic, matches the OG / placement cards
  ctx.fillStyle = INK;
  ctx.font = '600 18px Newsreader, serif';
  ctx.fillText(o.zh ? '一起从零做 AI →' : 'Build AI from scratch →', W / 2, 668);
  ctx.fillStyle = FAINT;
  ctx.font = '13px "JetBrains Mono", monospace';
  ctx.fillText('darkness-hy.github.io/ai-eng-studio', W / 2, 694);

  return toBlob(canvas);
}

export interface PlacementCardOpts {
  entryNum: number;
  entryTitle: string;
  total: number;
  maxTotal: number;
  areas: { label: string; score: number; max: number }[];
  zh: boolean;
}

/** Square share card for a placement result — the big phase number is the hook,
 *  the URL is the call-to-action that drives friends to test their own level. */
export async function generatePlacementCard(o: PlacementCardOpts): Promise<Blob> {
  await ready();
  const W = 800;
  const H = 800;
  const { canvas, ctx } = setup(W, H);

  ctx.strokeStyle = HAIRLINE;
  ctx.lineWidth = 1;
  ctx.strokeRect(40, 40, W - 80, H - 80);

  ctx.fillStyle = FAINT;
  ctx.font = '600 14px "JetBrains Mono", monospace';
  ctx.fillText('AI ENGINEERING · FROM SCRATCH', W / 2, 110);

  ctx.fillStyle = INK;
  ctx.font = '22px Newsreader, serif';
  ctx.fillText(o.zh ? '我的 AI 工程定级' : 'My AI Engineering Level', W / 2, 158);

  ctx.fillStyle = BLUE;
  ctx.font = '600 92px Newsreader, serif';
  ctx.fillText(o.zh ? `第 ${o.entryNum} 阶段` : `Phase ${o.entryNum}`, W / 2, 268);

  ctx.fillStyle = INK;
  ctx.font = '24px Newsreader, serif';
  ctx.fillText(o.entryTitle, W / 2, 310);

  ctx.fillStyle = FAINT;
  ctx.font = '16px "JetBrains Mono", monospace';
  ctx.fillText(`${o.zh ? '总分' : 'Score'} ${o.total} / ${o.maxTotal}`, W / 2, 348);

  // area mini-bars (colour by mastery — same thresholds as the result page)
  const bx0 = 312;
  const bw = 218;
  let by = 410;
  for (const a of o.areas) {
    const pct = a.max ? a.score / a.max : 0;
    ctx.textAlign = 'left';
    ctx.fillStyle = INK;
    ctx.font = '15px Newsreader, serif';
    ctx.fillText(a.label, 120, by + 4);
    ctx.fillStyle = '#eceae6';
    ctx.beginPath();
    ctx.roundRect(bx0, by - 6, bw, 8, 4);
    ctx.fill();
    ctx.fillStyle = a.score >= 8 ? '#346538' : a.score >= 4 ? GOLD : '#c0563f';
    ctx.beginPath();
    ctx.roundRect(bx0, by - 6, Math.max(bw * pct, 6), 8, 4);
    ctx.fill();
    ctx.textAlign = 'right';
    ctx.fillStyle = FAINT;
    ctx.font = '13px "JetBrains Mono", monospace';
    ctx.fillText(`${a.score}/${a.max}`, 660, by + 4);
    by += 42;
  }
  ctx.textAlign = 'center';

  ctx.fillStyle = INK;
  ctx.font = '600 18px Newsreader, serif';
  ctx.fillText(o.zh ? '测测你的起点 →' : 'Find your level →', W / 2, 706);
  ctx.fillStyle = FAINT;
  ctx.font = '13px "JetBrains Mono", monospace';
  ctx.fillText('darkness-hy.github.io/ai-eng-studio', W / 2, 732);

  return toBlob(canvas);
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
