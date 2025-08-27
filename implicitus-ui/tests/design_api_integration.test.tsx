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


let server: ChildProcess | undefined;

beforeAll(async () => {
  const serverPath = path.join(__dirname, 'start_design_api.py');
  server = spawn('python', [serverPath], { stdio: 'ignore' });
  // wait until server is responsive
  for (let i = 0; i < 50; i++) {
    try {
      const res = await fetch('http://127.0.0.1:8001/docs');
      if (res.ok) return;
    } catch (err) {
      // ignore until ready
    }
    await new Promise(r => setTimeout(r, 100));
  }
  throw new Error('design_api server did not start');
}, 10000);

afterAll(() => {
  server?.kill();
});

describe('design_api integration with VoronoiCanvas', () => {
  it('renders without warning when backend provides edges', async () => {
    const reqBody = {
      primitives: [
        {
          shape: 'box',
          size_mm: 10,
          infill: {
            pattern: 'voronoi',
            mode: 'organic',
            uniform: false,
            min_dist: 2,
            seed_points: [
              [0, 0, 0],
              [1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
            ],
            bbox_min: [0, 0, 0],
            bbox_max: [1, 1, 1],
          },
        },
      ],
    };

    const resp = await fetch('http://127.0.0.1:8001/design/review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(reqBody),
    });
    expect(resp.ok).toBe(true);
    const body = await resp.json();
    const infill = body.spec[0].modifiers.infill;
    expect(infill.edges.length).toBeGreaterThan(0);

    render(
      <VoronoiCanvas
        seedPoints={infill.seed_points}
        edges={infill.edges}
        bbox={[0, 0, 0, 1, 1, 1]}
      />
    );
    expect(screen.queryByTestId('no-edges-warning')).toBeNull();
  });
});
