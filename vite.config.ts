import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  // Served from GitHub Pages project path: https://darkness-hy.github.io/ai-eng-studio/
  base: process.env.DEPLOY_BASE ?? '/',
  plugins: [react(), tailwindcss()],
  server: {
    port: 5180,
    proxy: {
      '/research-api': {
        target: process.env.RESEARCH_API_PROXY_TARGET ?? 'http://127.0.0.1:8790',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/research-api/, ''),
      },
      '/tutor-api': {
        target: process.env.TUTOR_API_PROXY_TARGET ?? 'http://127.0.0.1:8787',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/tutor-api/, ''),
      },
    },
  },
});
