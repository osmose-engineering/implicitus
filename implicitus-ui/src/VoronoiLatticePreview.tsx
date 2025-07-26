import React, { useMemo } from 'react';

interface VoronoiLatticePreviewProps {
  spec: {
    min_dist?: number;
    resolution?: number[];
    periodic?: boolean;
    seed_points?: [number, number, number][];
  };
  bounds: [number, number, number][];
}

const MM_TO_UNIT = 0.1;

// seed sphere constants
const SEED_SPHERE_RADIUS_MM = 1.0; // 1mm

/**
 * Voronoi lattice preview using a single THREE.Points.
 */
const VoronoiLatticePreview: React.FC<VoronoiLatticePreviewProps> = ({
  spec,
  bounds,
}) => {
  const seedPoints = spec.seed_points ?? [];

  const seedPointsUnits = useMemo(() => {
    return seedPoints.map(([x, y, z]) => [x * MM_TO_UNIT, y * MM_TO_UNIT, z * MM_TO_UNIT]);
  }, [seedPoints]);

  // Compute Voronoi edge segments by connecting neighbors within cutoff distance
  const edgePositions = useMemo(() => {
    const cutoff = (spec.min_dist ?? 1) * MM_TO_UNIT * 2;
    const cutoff2 = cutoff * cutoff;
    const arr: number[] = [];
    for (let i = 0; i < seedPointsUnits.length; i++) {
      const [xi, yi, zi] = seedPointsUnits[i];
      for (let j = i + 1; j < seedPointsUnits.length; j++) {
        const [xj, yj, zj] = seedPointsUnits[j];
        const dx = xi - xj;
        const dy = yi - yj;
        const dz = zi - zj;
        if (dx * dx + dy * dy + dz * dz < cutoff2) {
          arr.push(xi, yi, zi, xj, yj, zj);
        }
      }
    }
    return new Float32Array(arr);
  }, [seedPointsUnits, spec.min_dist]);

  // seed point positions
  const pointsPositions = useMemo(() => {
    const arr: number[] = [];
    for (const [x, y, z] of seedPointsUnits) {
      arr.push(x, y, z);
    }
    return new Float32Array(arr);
  }, [seedPointsUnits]);

  return (
    <>
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
      <points>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={pointsPositions.length / 3}
            array={pointsPositions}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial size={(spec.min_dist ?? 1) * MM_TO_UNIT * 0.5} color="cyan" />
      </points>
    </>
  );
};

export default VoronoiLatticePreview;