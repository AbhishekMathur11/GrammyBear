import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'node:path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, './src'),
    },
  },
  server: {
    // Dev-only: proxies WebSocket calls to the real backend (main.py, port 8003)
    // so `npm run dev` can talk to the live app without any changes there.
    proxy: {
      '/ws': { target: 'ws://localhost:8003', ws: true },
    },
  },
})
