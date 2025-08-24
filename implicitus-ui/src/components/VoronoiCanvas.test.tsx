// @vitest-environment jsdom
import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { generateHexTest3D, computeFilteredEdges } from './VoronoiCanvas';
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

    const lengths = filtered.map(([i, j]) => {
      const [xi, yi, zi] = pts[i];
      const [xj, yj, zj] = pts[j];
      const dx = xi - xj;
      const dy = yi - yj;
      const dz = zi - zj;
      return Math.sqrt(dx * dx + dy * dy + dz * dz);
    });
    const unique = new Set(lengths.map(l => l.toFixed(6)));
    expect(unique.size).toBeLessThanOrEqual(2);

    const g = new Graph();
    pts.forEach((_, idx) => g.setNode(String(idx)));
    filtered.forEach(([i, j]) => g.setEdge(String(i), String(j)));
    const components = alg.components(g);
    expect(components.length).toBe(1);
  });
});

describe('VoronoiCanvas warning', () => {
  it('renders a warning banner when no edges are provided', () => {
    render(
      <VoronoiCanvas seedPoints={[]} edges={[]} bbox={[0, 0, 0, 1, 1, 1]} />
    );
    expect(screen.getByTestId('no-edges-warning')).toBeTruthy();
  });
});
