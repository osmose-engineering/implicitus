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
import { Graph, alg } from 'graphlib';

describe('VoronoiCanvas filteredEdges', () => {
  it('produces uniform edge lengths and a single connected component', () => {
    const bboxMin = [0, 0, 0];
    const bboxMax = [3, 3, 3];
    const spacing = 1;
    const [pts, edges] = generateHexTest3D(bboxMin, bboxMax, spacing);
    const filtered = computeFilteredEdges(pts as any, edges as any);

    expect(filtered.length).toBeGreaterThan(0);
    const lengths: number[] = [];
    let verticalCount = 0;
    filtered.forEach(([i, j]) => {
      const [xi, yi, zi] = pts[i];
      const [xj, yj, zj] = pts[j];
      const dx = xi - xj;
      const dy = yi - yj;
      const dz = zi - zj;
      lengths.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
      if (Math.abs(dz) > Math.max(Math.abs(dx), Math.abs(dy))) verticalCount++;
    });
    const unique = new Set(lengths.map(l => l.toFixed(6)));
    expect(unique.size).toBeLessThanOrEqual(2);
    expect(verticalCount).toBeGreaterThan(0);

    const g = new Graph();
    pts.forEach((_, idx) => g.setNode(String(idx)));
    filtered.forEach(([i, j]) => g.setEdge(String(i), String(j)));
    const components = alg.components(g);
    expect(components.length).toBe(1);
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
    // All edges, including the vertical one, should survive filtering.
    expect(filtered).toContainEqual([0, 5]);
    expect(filtered.length).toBe(edges.length);
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

  it('warns about edges with near-zero z-range', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const vertices: [number, number, number][] = [
      [0, 0, 0],
      [1, 0, EDGE_Z_VARIATION_TOLERANCE / 2], // practically flat
    ];
    const edges: [number, number][] = [[0, 1]];
    render(
      <VoronoiCanvas
        seedPoints={[]}
        vertices={vertices}
        edges={edges}
        bbox={[0, 0, 0, 1, 1, 1]}
      />
    );
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining('edge z-range below tolerance')
    );
    const root = screen.getByTestId('voronoi-canvas-root');
    expect(root.dataset.hasFlatEdges).toBe('true');
    warn.mockRestore();
  });

  it('throws when strict z-range assertion is enabled', () => {
    process.env.VORONOI_ASSERT_Z = 'true';
    const vertices: [number, number, number][] = [
      [0, 0, 0],
      [1, 0, 0], // identical z
    ];
    const edges: [number, number][] = [[0, 1]];
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
