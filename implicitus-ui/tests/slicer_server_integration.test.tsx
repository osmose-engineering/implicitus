// @vitest-environment jsdom
import React from 'react';
import { beforeAll, afterAll, describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import VoronoiCanvas from '../src/components/VoronoiCanvas';
import { spawn, ChildProcess } from 'child_process';
import path from 'path';

// Polyfill ResizeObserver for Three.js
(globalThis as any).ResizeObserver = class {
  observe(){}
  unobserve(){}
  disconnect(){}
};

let designServer: ChildProcess | undefined;
let slicerServer: ChildProcess | undefined;

beforeAll(async () => {
  const designUrl = 'http://127.0.0.1:8001/docs';
  let designRunning = false;
  try {
    const res = await fetch(designUrl);
    if (res.ok) designRunning = true;
  } catch {
    // not running
  }
  if (!designRunning) {
    const designPath = path.join(__dirname, 'start_design_api.py');
    designServer = spawn('python', [designPath], { stdio: 'ignore' });
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
  slicerServer = spawn('cargo', ['run', '--bin', 'slicer_server'], {
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

describe('slicer_server integration', () => {
  it('returns circular contour and renders without warnings', async () => {
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
  });
});
