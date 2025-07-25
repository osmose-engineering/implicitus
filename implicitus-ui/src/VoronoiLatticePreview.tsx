import React, { useMemo } from 'react';
import { BufferAttribute } from 'three';

interface VoronoiLatticePreviewProps {
  spec: {
    min_dist?: number;
    resolution?: number[];
    periodic?: boolean;
  };
  bounds: [number, number, number][];
  seedPoints: [number, number, number][];
}

const MM_TO_UNIT = 0.1;

/**
 * Voronoi lattice preview using a single THREE.Points.
 */
const VoronoiLatticePreview: React.FC<VoronoiLatticePreviewProps> = ({
  spec,
  bounds,
  seedPoints,
}) => {
  // Convert seedPoints (already scaled to scene units) into a flat Float32Array
  const positions = useMemo(() => new Float32Array(seedPoints.flat()), [
    seedPoints,
  ]);

  // Compute point size from min_dist (in mm), scaled to scene units
  const pointSize = (spec.min_dist ?? 1) * MM_TO_UNIT * 0.5;

  // Compute Voronoi edge segments by connecting neighbors within cutoff distance
  const edgePositions = useMemo(() => {
    const cutoff = (spec.min_dist ?? 1) * MM_TO_UNIT * 2;
    const cutoff2 = cutoff * cutoff;
    const arr: number[] = [];
    for (let i = 0; i < seedPoints.length; i++) {
      const [xi, yi, zi] = seedPoints[i];
      for (let j = i + 1; j < seedPoints.length; j++) {
        const [xj, yj, zj] = seedPoints[j];
        const dx = xi - xj;
        const dy = yi - yj;
        const dz = zi - zj;
        if (dx * dx + dy * dy + dz * dz < cutoff2) {
          arr.push(xi, yi, zi, xj, yj, zj);
        }
      }
    }
    return new Float32Array(arr);
  }, [seedPoints, spec.min_dist]);

  return (
    <>
      <points>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={positions.length / 3}
            array={positions}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial size={pointSize} color="cyan" />
      </points>
      <lineSegments>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={edgePositions.length / 3}
            array={edgePositions}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial linewidth={1} color="white" />
      </lineSegments>
    </>
  );
};

export default VoronoiLatticePreview;