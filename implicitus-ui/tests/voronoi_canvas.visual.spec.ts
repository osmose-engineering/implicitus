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
  page.on('pageerror', err => { throw err; });
  page.on('console', msg => {
    if (
      msg.type() === 'warning' &&
      msg.text().includes('edge z-range below tolerance')
    ) {
      throw new Error(msg.text());
    }
  });
  await page.addInitScript(() => {
    (window as any).process = {
      env: { NODE_ENV: 'test', VORONOI_ASSERT_Z: 'true' }
    };
  });
  await page.goto('http://localhost:3000/tests/visual/voronoi_canvas_page.html');
  const root = page.locator('[data-testid="voronoi-canvas-root"]');
  await expect(root).toHaveAttribute('data-has-flat-edges', 'false');
  const canvas = page.locator('canvas');
  await expect(canvas).toHaveScreenshot('voronoi-canvas-baseline.png');
});
