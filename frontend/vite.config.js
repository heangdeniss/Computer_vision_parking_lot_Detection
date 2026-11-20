import { spawn } from 'node:child_process'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const ROOT = fileURLToPath(new URL('..', import.meta.url)) // project root, one level above frontend/
let api

// Start the inference API so `npm run dev` alone gives a working UI.
function startApi() {
  api = spawn(join(ROOT, '.venv/bin/python'),
    ['-m', 'uvicorn', 'server.app:app', '--port', '8000', '--log-level', 'warning'],
    { cwd: ROOT, stdio: 'inherit' })
  api.on('error', () => {}) // ignored if a server is already running on port 8000
}

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'parking-api',
      configureServer(server) {
        server.httpServer.on('close', () => api?.kill())
        fetch('http://127.0.0.1:8000/api/health')
          .then((r) => { if (!r.ok) startApi() })
          .catch(startApi)
      },
    },
  ],
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
