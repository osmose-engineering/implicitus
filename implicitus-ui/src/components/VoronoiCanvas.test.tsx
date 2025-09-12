// @vitest-environment jsdom
import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import {
  generateHexTest3D,
  computeFilteredEdges,
  EDGE_Z_VARIATION_TOLERANCE,
} from './VoronoiCanvas';
import VoronoiCanvas from './VoronoiCanvas';

describe('VoronoiCanvas filteredEdges', () => {
  it('produces uniform edge lengths and a single connected component', () => {
    const bboxMin = [0, 0, 0];
    const bboxMax = [3, 3, 3];
    const spacing = 1;
    const [pts, edges] = generateHexTest3D(bboxMin, bboxMax, spacing);
    const filtered = computeFilteredEdges(pts as any, edges as any);
    expect(filtered.length).toBeGreaterThan(0);
    filtered.forEach(([i, j]) => {
      const dz = Math.abs(pts[i][2] - pts[j][2]);
      expect(dz).toBeGreaterThanOrEqual(EDGE_Z_VARIATION_TOLERANCE);
    });
  });

  it('keeps long vertical edges that exceed horizontal lengths', () => {
    // A star-shaped set of short horizontal edges around a central point
    // plus one tall vertical edge. In older implementations using a single
    // length threshold, the vertical edge would be discarded as an outlier.
    const pts = [
      [0, 0, 0],   // center
      [1, 0, 0],   // +x
      [0, 1, 0],   // +y
      [-1, 0, 0],  // -x
      [0, -1, 0],  // -y
      [0, 0, 5],   // tall z
    ];
    const edges = [
      [0, 1],
      [0, 2],
      [0, 3],
      [0, 4],
      [0, 5], // vertical edge length 5
    ];

    const filtered = computeFilteredEdges(pts as any, edges as any);
    // Only the vertical edge should survive filtering.
    expect(filtered).toEqual([[0, 5]]);
  });

  it('drops edges with near-zero z variation', () => {
    const pts = [
      [0, 0, 0],
      [1, 0, EDGE_Z_VARIATION_TOLERANCE / 2],
    ];
    const edges = [[0, 1]] as any;
    const filtered = computeFilteredEdges(pts as any, edges);
    expect(filtered).toHaveLength(0);
  });
});

describe('VoronoiCanvas warning', () => {
  it('renders a warning banner when no edges are provided', () => {
    render(
      <VoronoiCanvas seedPoints={[]} edges={[]} bbox={[0, 0, 0, 1, 1, 1]} />
    );
    expect(screen.getByTestId('no-edges-warning')).toBeTruthy();
  });

  it('logs a warning when vertices are absent', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    render(
      <VoronoiCanvas
        seedPoints={[]}
        vertices={[]}
        edges={[[0, 1]]}
        bbox={[0, 0, 0, 1, 1, 1]}
      />
    );
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining('no vertices provided')
    );
    warn.mockRestore();
  });

  it('warns once when filtered edges are planar', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const vertices: [number, number, number][] = [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
    ];
    const edges: [number, number][] = [
      [0, 1],
      [1, 2],
    ];
    const props = {
      seedPoints: [],
      vertices,
      edges,
      bbox: [0, 0, 0, 1, 1, 1] as [number, number, number, number, number, number],
    };
    const { rerender } = render(<VoronoiCanvas {...props} />);
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining('edge z-range below tolerance')
    );
    const root = screen.getByTestId('voronoi-canvas-root');
    expect(root.dataset.hasFlatEdges).toBe('true');
    rerender(<VoronoiCanvas {...props} />);
    expect(warn).toHaveBeenCalledTimes(1);
    warn.mockRestore();
  });

  it('does not warn when edges span 3D', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const vertices: [number, number, number][] = [
      [0, 0, 0],
      [1, 0, 1],
      [0, 1, 2],
    ];
    const edges: [number, number][] = [
      [0, 1],
      [1, 2],
    ];
    render(
      <VoronoiCanvas
        seedPoints={[]}
        vertices={vertices}
        edges={edges}
        bbox={[0, 0, 0, 1, 1, 1]}
      />
    );
    expect(warn).not.toHaveBeenCalledWith(
      expect.stringContaining('edge z-range below tolerance')
    );
    const root = screen.getByTestId('voronoi-canvas-root');
    expect(root.dataset.hasFlatEdges).toBe('false');
    warn.mockRestore();
  });

  it('throws when strict z-range assertion is enabled', () => {
    process.env.VORONOI_ASSERT_Z = 'true';
    const vertices: [number, number, number][] = [
      [0, 0, 0],
      [1, 0, 0],
    ];
    const edges: [number, number][] = [
      [0, 1],
    ];
    expect(() =>
      render(
        <VoronoiCanvas
          seedPoints={[]}
          vertices={vertices}
          edges={edges}
          bbox={[0, 0, 0, 1, 1, 1]}
        />
      )
    ).toThrow(/edge z-range below tolerance/);
    delete process.env.VORONOI_ASSERT_Z;
  });
});

describe('VoronoiCanvas DOM rendering', () => {
  it('renders the expected number of vertices and edges', async () => {
    vi.resetModules();
    vi.doMock('./VoronoiStruts', () => ({
      VoronoiStruts: ({ vertices, edges }: any) => (
        <svg data-testid="voronoi-svg">
          {vertices.map((_: any, i: number) => (
            <circle key={`v-${i}`} />
          ))}
          {edges.map((_: any, i: number) => (
            <line key={`e-${i}`} />
          ))}
        </svg>
      ),
    }));
    vi.doMock('@react-three/fiber', () => ({
      Canvas: ({ children }: any) => <div>{children}</div>,
      extend: () => {},
    }));
    vi.doMock('@react-three/drei', async () => {
      const actual = await vi.importActual<any>('@react-three/drei');
      return {
        ...actual,
        OrbitControls: () => <div />,
      };
    });
    const { default: MockedVoronoiCanvas } = await import('./VoronoiCanvas');
    const vertices: [number, number, number][] = [
      [0, 0, 0],
      [1, 0, 1],
      [2, 0, 2],
    ];
    const edges: [number, number][] = [
      [0, 1],
      [1, 2],
    ];
    render(
      <MockedVoronoiCanvas
        seedPoints={vertices}
        vertices={vertices}
        edges={edges}
        bbox={[0, 0, 0, 1, 1, 1]}
        showStruts
      />
    );
    const svg = screen.getByTestId('voronoi-svg');
    expect(svg.querySelectorAll('circle')).toHaveLength(vertices.length);
    expect(svg.querySelectorAll('line')).toHaveLength(edges.length);
    vi.resetModules();
  });
});
