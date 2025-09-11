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
const pythonCheck = spawnSync(PYTHON, ['-c', 'import fastapi, uvicorn'], {
  stdio: 'ignore',
});
const hasPython = pythonCheck.status === 0;

const mockHandlers = [
  http.get('http://127.0.0.1:8001/docs', () => HttpResponse.text('ok')),
  http.post('http://127.0.0.1:8001/design/review', async () => {
    const mockBody = {
      spec: [
        {
          modifiers: {
            infill: {
              seed_points: [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
              ],
              edge_list: [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
              ],
              cell_vertices: [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
              ],
            },
          },
        },
      ],
    };
    return HttpResponse.json(mockBody);
  }),
];

const mockServer = setupServer(...mockHandlers);

const runReal = hasPython;

const testBody = async () => {
  const reqBody = {
    primitives: [
      {
        shape: 'box',
        size_mm: 10,
        infill: {
          pattern: 'voronoi',
          mode: 'organic',
          bbox_min: [0, 0, 0],
          bbox_max: [1, 1, 1],
          seed_points: [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
          ],
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
  const vertices = infill.cell_vertices || infill.vertices || [];
  expect(infill.edge_list.length).toBeGreaterThan(0);
  expect(vertices.length).toBeGreaterThan(0);

  render(
    <VoronoiCanvas
      seedPoints={infill.seed_points}
      edges={infill.edge_list}
      vertices={vertices}
      bbox={[0, 0, 0, 1, 1, 1]}
    />
  );
  expect(screen.queryByTestId('no-edges-warning')).toBeNull();
};

(runReal ? describe : describe.skip)('design_api integration with VoronoiCanvas', () => {
  let server: ChildProcess | undefined;

  beforeAll(async () => {
    const serverPath = path.join(__dirname, 'start_design_api.py');
    server = spawn(PYTHON, [serverPath], { stdio: 'ignore' });
    for (let i = 0; i < 50; i++) {
      try {
        const res = await fetch('http://127.0.0.1:8001/docs');
        if (res.ok) return;
      } catch {}
      await new Promise(r => setTimeout(r, 100));
    }
    throw new Error('design_api server did not start');
  }, 10000);

  afterAll(() => {
    server?.kill();
  });

  it('renders without warning when backend provides edges', testBody);
});

describe('design_api mocked integration', () => {
  beforeAll(() => mockServer.listen());
  afterAll(() => mockServer.close());
  it('renders without warning when backend provides edges', testBody);
});
