import type { ClientRequest, IncomingMessage, ServerResponse } from 'http';
import { defineConfig, loadEnv } from 'vite'
import macros from 'vite-plugin-babel-macros'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const apiTarget = env.VITE_API_BASE_URL || 'http://localhost:8000';
  return {
    babelMacros: {
      plugins: ['babel-plugin-glsl/macro']
    },
    plugins: [
      macros(),
      react()
    ],
    server: {
      host: 'localhost',
      port: 3000,
      strictPort: true,
      proxy: {
        '/design': {
          target: apiTarget,
          changeOrigin: true,
          secure: false,
          configure: (proxy, _options) => {
            // cast to any so TS won’t complain
            (proxy as any).on('proxyReq', (_proxyReq: ClientRequest, req: IncomingMessage, _res: ServerResponse) => {
              console.debug('[vite proxy] request to:', req.url);
            });
            (proxy as any).on('proxyRes', (proxyRes: IncomingMessage, _req: IncomingMessage, _res: ServerResponse) => {
              console.debug('[vite proxy] response status:', proxyRes.statusCode);
            });
          }
        },
        '/models': {
          target: apiTarget,
          changeOrigin: true,
          secure: false,
        }
      }
    }
  };
});