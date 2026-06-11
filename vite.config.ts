import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  // Served from GitHub Pages project path: https://darkness-hy.github.io/ai-eng-studio/
  base: process.env.DEPLOY_BASE ?? '/',
  plugins: [react(), tailwindcss()],
  server: { port: 5180 },
});
