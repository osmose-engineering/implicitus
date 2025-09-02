import { test, expect } from '@playwright/test';
import { createServer, ViteDevServer } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';

// Update baseline with:
//   npx playwright test tests/voronoi_canvas.visual.spec.ts --update-snapshots

const __dirname = fileURLToPath(new URL('.', import.meta.url));

let server: ViteDevServer | undefined;

test.beforeAll(async () => {
  server = await createServer({
    configFile: path.resolve(__dirname, '../vite.config.ts'),
    root: path.resolve(__dirname, '..'),
    server: { port: 3000, strictPort: true },
  });
  await server.listen();
});

test.afterAll(async () => {
  await server?.close();
});

test('VoronoiCanvas visual regression', async ({ page }) => {
  page.on('pageerror', err => console.error('pageerror:', err));
  page.on('console', msg => console.log('console:', msg.text()));
  await page.addInitScript(() => {
    (window as any).process = { env: { NODE_ENV: 'test' } };
  });
  await page.goto('http://localhost:3000/tests/visual/voronoi_canvas_page.html');
  const canvas = page.locator('canvas');
  await expect(canvas).toHaveScreenshot('voronoi-canvas-baseline.png');
});
