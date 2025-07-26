import React, { useMemo, useRef, useLayoutEffect } from 'react';
import { InstancedMesh, Object3D } from 'three';

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

// seed sphere constants
const SEED_SPHERE_RADIUS_MM = 1.0; // 1mm
const SEED_SPHERE_SEGMENTS = 8;

/**
 * Voronoi lattice preview using a single THREE.Points.
 */
const VoronoiLatticePreview: React.FC<VoronoiLatticePreviewProps> = ({
  spec,
  bounds,
  seedPoints,
}) => {
  const MAX_INSTANCES = 500;
  const instances = seedPoints.slice(0, MAX_INSTANCES);

  // Sphere radius for each seed point (half of min_dist)
  const meshRef = useRef<InstancedMesh>(null!);
  const tempObj = useMemo(() => new Object3D(), []);
  useLayoutEffect(() => {
    if (!meshRef.current) return;
    const mesh = meshRef.current;
    instances.forEach((pt, i) => {
      tempObj.position.set(pt[0] * MM_TO_UNIT, pt[1] * MM_TO_UNIT, pt[2] * MM_TO_UNIT);
      tempObj.updateMatrix();
      mesh.setMatrixAt(i, tempObj.matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
  }, [instances]);

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
      <instancedMesh
        ref={meshRef}
        count={instances.length}
        castShadow
        receiveShadow
      >
        <sphereGeometry args={[SEED_SPHERE_RADIUS_MM * MM_TO_UNIT, 4, 4]} />
        <meshStandardMaterial color="cyan" />
      </instancedMesh>
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