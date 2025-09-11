// @vitest-environment jsdom
import React from 'react';
import { beforeAll, afterAll, describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import VoronoiCanvas from '../src/components/VoronoiCanvas';
import { spawn, spawnSync, ChildProcess } from 'child_process';
import path from 'path';
import fs from 'fs';

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

(hasDeps ? describe : describe.skip)('full stack integration', () => {
  let designServer: ChildProcess | undefined;
  let slicerServer: ChildProcess | undefined;

  beforeAll(async () => {
    const designPath = path.join(__dirname, 'start_design_api.py');
    designServer = spawn(PYTHON, [designPath], { stdio: 'ignore' });
    for (let i = 0; i < 50; i++) {
      try {
        const res = await fetch('http://127.0.0.1:8001/docs');
        if (res.ok) break;
      } catch {}
      await new Promise(r => setTimeout(r, 100));
    }

    const slicerPath = path.join(__dirname, 'start_slicer_server.js');
    slicerServer = spawn('node', [slicerPath], { stdio: 'ignore' });
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

  it('renders geometry without console warnings or errors', async () => {
    const modelPath = path.join(__dirname, '..', '..', 'examples', 'helmet_shell.json');
    const model = JSON.parse(fs.readFileSync(modelPath, 'utf-8'));
    const resp = await fetch('http://127.0.0.1:8001/design/review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(model),
    });
    expect(resp.ok).toBe(true);
    const body = await resp.json();
    const infill = body.spec[0].modifiers.infill;

    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    render(
      <VoronoiCanvas
        seedPoints={infill.seed_points}
        edges={infill.edge_list}
        vertices={infill.cell_vertices || []}
        bbox={[0, 0, 0, 1, 1, 1]}
      />
    );

    expect(screen.queryByTestId('no-edges-warning')).toBeNull();
    expect(warnSpy).not.toHaveBeenCalled();
    expect(errorSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
    errorSpy.mockRestore();
  });
});
