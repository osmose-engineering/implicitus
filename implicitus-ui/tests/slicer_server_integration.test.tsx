// @vitest-environment jsdom
import React from 'react';
import { beforeAll, afterAll, describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import VoronoiCanvas from '../src/components/VoronoiCanvas';
import { spawn, spawnSync, ChildProcess } from 'child_process';
import path from 'path';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';

// Polyfill ResizeObserver for Three.js
(globalThis as any).ResizeObserver = class {
  observe(){}
  unobserve(){}
  disconnect(){}
};

const PYTHON = process.env.PYTHON || 'python';
const CARGO = process.env.CARGO || 'cargo';
const pythonCheck = spawnSync(PYTHON, ['-c', 'import fastapi, uvicorn'], { stdio: 'ignore' });
const cargoCheck = spawnSync(CARGO, ['--version'], { stdio: 'ignore' });
const hasDeps = pythonCheck.status === 0 && cargoCheck.status === 0;

const contour = Array.from({ length: 40 }, (_, i) => {
  const theta = (2 * Math.PI * i) / 40;
  return [Math.cos(theta), Math.sin(theta)];
});

const mockServer = setupServer(
  http.post('http://127.0.0.1:4000/slice', () =>
    HttpResponse.json({ contours: [contour] })
  )
);

const testBody = async () => {
  const reqBody = { model: {}, layer: 0 };
  const resp = await fetch('http://127.0.0.1:4000/slice', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(reqBody),
  });
  expect(resp.ok).toBe(true);
  const body = await resp.json();
  const contour = body.contours[0];
  expect(contour.length).toBeGreaterThan(0);
  for (const [x, y] of contour) {
    expect(Math.abs(x * x + y * y - 1)).toBeLessThan(0.1);
  }
  const sample = contour.slice(0, 20);
  const seedPoints = sample.map(([x, y]: [number, number]) => [x, y, 0]);
  const edges = sample.map((_, i) => [i, (i + 1) % sample.length]);
  render(
    <VoronoiCanvas seedPoints={seedPoints} edges={edges} bbox={[-1, -1, -1, 1, 1, 1]} />
  );
  expect(screen.queryByTestId('no-edges-warning')).toBeNull();
};

(hasDeps ? describe : describe.skip)('slicer_server integration', () => {
  let designServer: ChildProcess | undefined;
  let slicerServer: ChildProcess | undefined;

  beforeAll(async () => {
    const designUrl = 'http://127.0.0.1:8001/docs';
    let designRunning = false;
    try {
      const res = await fetch(designUrl);
      if (res.ok) designRunning = true;
    } catch {}
    if (!designRunning) {
      const designPath = path.join(__dirname, 'start_design_api.py');
      designServer = spawn(PYTHON, [designPath], { stdio: 'ignore' });
      for (let i = 0; i < 50; i++) {
        try {
          const res = await fetch(designUrl);
          if (res.ok) { designRunning = true; break; }
        } catch {}
        await new Promise(r => setTimeout(r, 100));
      }
      if (!designRunning) throw new Error('design_api server did not start');
    }

    const slicerCwd = path.join(__dirname, '..', '..', 'core_engine');
    slicerServer = spawn(CARGO, ['run', '--bin', 'slicer_server'], {
      cwd: slicerCwd,
      stdio: 'ignore',
    });
    const sliceReq = { model: {}, layer: 0 };
    for (let i = 0; i < 100; i++) {
      try {
        const res = await fetch('http://127.0.0.1:4000/slice', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(sliceReq),
        });
        if (res.ok) break;
      } catch {}
      await new Promise(r => setTimeout(r, 1000));
    }
  }, 180000);

  afterAll(() => {
    designServer?.kill();
    slicerServer?.kill();
  });

  it('returns circular contour and renders without warnings', testBody);
});

describe('slicer_server mocked integration', () => {
  beforeAll(() => mockServer.listen());
  afterAll(() => mockServer.close());
  it('returns circular contour and renders without warnings', testBody);
});
