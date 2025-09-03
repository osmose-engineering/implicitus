// @vitest-environment jsdom
import React, { useState } from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, beforeAll, afterAll, expect } from 'vitest';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import VoronoiCanvas from '../src/components/VoronoiCanvas';

// Polyfill ResizeObserver for Three.js
(globalThis as any).ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};

// Mock backend with delayed completion
let statusHits = 0;
const server = setupServer(
  http.post('http://localhost:4000/voronoi', () =>
    HttpResponse.json({ job_id: 'job-1' }, { status: 202 })
  ),
  http.get('http://localhost:4000/voronoi/status/job-1', () => {
    statusHits++;
    if (statusHits < 3) {
      return HttpResponse.json({ status: 'rendering' }, { status: 202 });
    }
    return HttpResponse.json({
      status: 'complete',
      vertices: [ [0,0,0], [1,0,0] ],
      edges: [ [0,1] ],
    });
  })
);

// Minimal harness that mirrors App's polling behaviour
function Harness({ seeds }: { seeds: [number, number, number][] }) {
  const [verts, setVerts] = useState<[number, number, number][]>([]);
  const [edges, setEdges] = useState<number[][]>([]);
  const [rendering, setRendering] = useState(false);

  React.useEffect(() => {
    let cancelled = false;
    if (seeds.length === 0) return;
    setRendering(true);
    fetch('http://localhost:4000/voronoi', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ seeds })
    })
      .then(res => res.json())
      .then(data => {
        const poll = () => {
          fetch(`http://localhost:4000/voronoi/status/${data.job_id}`)
            .then(res => {
              if (res.status === 202) {
                if (!cancelled) setTimeout(poll, 10);
                return null;
              }
              return res.json();
            })
            .then(data => {
              if (!data) return;
              setVerts(data.vertices || []);
              setEdges(data.edges || []);
              setRendering(false);
            });
        };
        poll();
      });
    return () => {
      cancelled = true;
    };
  }, [seeds]);

  return (
    <div>
      {rendering && <span>Rendering...</span>}
      {!rendering && (
        <VoronoiCanvas
          seedPoints={seeds}
          vertices={verts}
          edges={edges}
          infillPoints={verts}
          infillEdges={edges}
          bbox={[0,0,0,1,1,1]}
          thickness={0.35}
          maxSteps={256}
          epsilon={0.001}
          showSolid
          showInfill
          showRaymarch={false}
          showStruts={false}
        />
      )}
    </div>
  );
}

describe('voronoi status polling', () => {
  beforeAll(() => server.listen());
  afterAll(() => server.close());

  it('shows rendering message then renders mesh when ready', async () => {
    const seeds: [number, number, number][] = [ [0,0,0], [1,0,0] ];
    render(<Harness seeds={seeds} />);
    // Initially show rendering
    expect(screen.getByText('Rendering...')).toBeTruthy();
    // Wait until rendering message disappears
    await waitFor(() => expect(screen.queryByText('Rendering...')).toBeNull());
    // Should render without warnings
    expect(screen.queryByTestId('no-edges-warning')).toBeNull();
  });
});

