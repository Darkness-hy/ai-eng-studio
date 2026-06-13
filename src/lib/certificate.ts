/** Canvas-rendered certificates and shareable achievement cards (PNG download). */

const INK = '#2f3437';
const FAINT = '#787774';
const BLUE = '#1f6c9f';
const CANVAS_BG = '#fbfbfa';
const HAIRLINE = '#d9d8d4';
const PINK = '#f2335d';

async function ready(): Promise<void> {
  try {
    await Promise.all([
      document.fonts.load('600 56px Newsreader'),
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
  return new Promise((res) => canvas.toBlob((b) => res(b!), 'image/png'));
}

/** The LaViRA mark (rounded dark plate + light bars + pink dot). */
function drawMark(ctx: CanvasRenderingContext2D, x: number, y: number, s: number) {
  ctx.save();
  ctx.translate(x, y);
  const r = s * 0.22;
  ctx.fillStyle = '#111111';
  ctx.beginPath();
  ctx.roundRect(0, 0, s, s, r);
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

export interface CertOpts {
  name: string;
  achievement: string; // e.g. "完成「深度学习核心」阶段（13 课）"
  date: string;
  zh: boolean;
}

export async function generateCertificate(o: CertOpts): Promise<Blob> {
  await ready();
  const W = 1200;
  const H = 849;
  const { canvas, ctx } = setup(W, H);

  ctx.strokeStyle = INK;
  ctx.lineWidth = 2;
  ctx.strokeRect(44, 44, W - 88, H - 88);
  ctx.strokeStyle = HAIRLINE;
  ctx.lineWidth = 1;
  ctx.strokeRect(56, 56, W - 112, H - 112);

  ctx.fillStyle = FAINT;
  ctx.font = '600 15px "JetBrains Mono", monospace';
  ctx.fillText(o.zh ? '结业证书 · CERTIFICATE OF COMPLETION' : 'CERTIFICATE OF COMPLETION', W / 2, 150);

  ctx.fillStyle = INK;
  ctx.font = '600 50px Newsreader, serif';
  ctx.fillText(o.zh ? '从零开始的 AI 工程' : 'AI Engineering from Scratch', W / 2, 270);

  ctx.fillStyle = FAINT;
  ctx.font = '22px Newsreader, serif';
  ctx.fillText(o.zh ? '兹证明' : 'This certifies that', W / 2, 360);

  ctx.fillStyle = BLUE;
  ctx.font = '600 46px Newsreader, serif';
  ctx.fillText(o.name, W / 2, 432);

  ctx.fillStyle = INK;
  ctx.font = '24px Newsreader, serif';
  ctx.fillText(o.achievement, W / 2, 510);

  ctx.fillStyle = FAINT;
  ctx.font = '15px "JetBrains Mono", monospace';
  ctx.fillText(o.date, W / 2, 580);

  drawMark(ctx, W / 2 - 110, 660, 40);
  ctx.fillStyle = INK;
  ctx.font = '600 20px Newsreader, serif';
  ctx.textAlign = 'left';
  ctx.fillText('LaViRA', W / 2 - 58, 688);
  ctx.fillStyle = FAINT;
  ctx.font = '12px "JetBrains Mono", monospace';
  ctx.textAlign = 'center';
  ctx.fillText('Language-Vision-Robot Actions Translation · 独家冠名', W / 2, 740);

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

  ctx.fillStyle = FAINT;
  ctx.font = '600 14px "JetBrains Mono", monospace';
  ctx.fillText('AI ENGINEERING · FROM SCRATCH', W / 2, 120);

  ctx.fillStyle = INK;
  ctx.font = '600 40px Newsreader, serif';
  ctx.fillText(o.name, W / 2, 200);
  ctx.fillStyle = FAINT;
  ctx.font = '18px Newsreader, serif';
  ctx.fillText(o.zh ? '的学习成就' : "'s learning so far", W / 2, 238);

  const stat = (x: number, big: string, label: string) => {
    ctx.fillStyle = BLUE;
    ctx.font = '600 52px Newsreader, serif';
    ctx.fillText(big, x, 380);
    ctx.fillStyle = FAINT;
    ctx.font = '15px "JetBrains Mono", monospace';
    ctx.fillText(label, x, 415);
  };
  stat(W / 2 - 230, `${o.doneCount}`, o.zh ? '完成课程' : 'lessons');
  stat(W / 2, `${o.streak}`, o.zh ? '连续天数' : 'day streak');
  stat(W / 2 + 230, `${o.badges}`, o.zh ? '解锁徽章' : 'badges');

  // progress bar
  const pct = o.totalLessons ? o.doneCount / o.totalLessons : 0;
  const bx = 160;
  const bw = W - 320;
  ctx.fillStyle = '#eceae6';
  ctx.beginPath();
  ctx.roundRect(bx, 500, bw, 10, 5);
  ctx.fill();
  ctx.fillStyle = '#346538';
  ctx.beginPath();
  ctx.roundRect(bx, 500, Math.max(bw * pct, 6), 10, 5);
  ctx.fill();
  ctx.fillStyle = FAINT;
  ctx.font = '14px "JetBrains Mono", monospace';
  ctx.fillText(`${o.doneCount} / ${o.totalLessons} · ${Math.round(pct * 100)}%`, W / 2, 545);

  drawMark(ctx, W / 2 - 100, 650, 36);
  ctx.fillStyle = INK;
  ctx.font = '600 18px Newsreader, serif';
  ctx.textAlign = 'left';
  ctx.fillText('LaViRA', W / 2 - 52, 674);
  ctx.textAlign = 'center';

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
