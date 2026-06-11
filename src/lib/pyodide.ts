/**
 * Lazy Pyodide loader for in-browser Python execution.
 * Loaded from CDN on first run; only stdlib + numpy lessons are marked runnable.
 */

const PYODIDE_VERSION = 'v0.26.4';
const CDN = `https://cdn.jsdelivr.net/pyodide/${PYODIDE_VERSION}/full/`;

interface PyodideApi {
  runPythonAsync(code: string): Promise<unknown>;
  loadPackagesFromImports(code: string): Promise<void>;
  setStdout(opts: { batched: (s: string) => void }): void;
  setStderr(opts: { batched: (s: string) => void }): void;
}

declare global {
  interface Window {
    loadPyodide?: (opts: { indexURL: string }) => Promise<PyodideApi>;
  }
}

let pyodidePromise: Promise<PyodideApi> | null = null;

function getPyodide(): Promise<PyodideApi> {
  if (!pyodidePromise) {
    pyodidePromise = (async () => {
      if (!window.loadPyodide) {
        await new Promise<void>((resolve, reject) => {
          const script = document.createElement('script');
          script.src = `${CDN}pyodide.js`;
          script.onload = () => resolve();
          script.onerror = () => reject(new Error('Pyodide 加载失败'));
          document.head.appendChild(script);
        });
      }
      return window.loadPyodide!({ indexURL: CDN });
    })();
    pyodidePromise.catch(() => {
      pyodidePromise = null; // allow retry after a network failure
    });
  }
  return pyodidePromise;
}

export interface RunResult {
  output: string;
  error: string | null;
  ms: number;
}

export async function runPython(code: string): Promise<RunResult> {
  const started = performance.now();
  const py = await getPyodide();
  let output = '';
  py.setStdout({ batched: (s) => (output += s + '\n') });
  py.setStderr({ batched: (s) => (output += s + '\n') });
  try {
    await py.loadPackagesFromImports(code);
    const result = await py.runPythonAsync(code);
    if (result !== undefined && result !== null) output += String(result) + '\n';
    return { output, error: null, ms: performance.now() - started };
  } catch (err) {
    return {
      output,
      error: err instanceof Error ? err.message : String(err),
      ms: performance.now() - started,
    };
  }
}
