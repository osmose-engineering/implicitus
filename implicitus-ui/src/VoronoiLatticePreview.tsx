import React, { useMemo, useRef, useEffect } from 'react';
import * as THREE from 'three';

interface VoronoiLatticePreviewProps {
  spec: {
    min_dist?: number;
    periodic?: boolean;
    seed_points?: [number, number, number][];
    wall_thickness?: number;
  };
  bounds?: [number, number, number][];
}

const MM_TO_UNIT = 0.1;

// seed sphere constants
const SEED_SPHERE_RADIUS_MM = 1.0; // 1mm

/**
 * Voronoi lattice preview using a single THREE.Points.
 */
const VoronoiLatticePreview: React.FC<VoronoiLatticePreviewProps> = ({
  spec,
  bounds: _bounds,
}) => {
  const seedPointsUnits = useMemo(
    () =>
      (spec.seed_points ?? []).map(
        ([x, y, z]) => [x * MM_TO_UNIT, y * MM_TO_UNIT, z * MM_TO_UNIT] as [number, number, number]
      ),
    [spec.seed_points]
  );

  // Strut thickness in scene units.  If wall_thickness is omitted, fall back to
  // min_dist, and finally a 1mm default.
  const thicknessUnits = ((spec.wall_thickness ?? spec.min_dist ?? 1) as number) * MM_TO_UNIT;

  // Compute Voronoi edge segments by connecting neighboring seeds.  When
  // min_dist is not provided (e.g. when a fixed seed count is requested), the
  // previous implementation used a constant 1 mm cutoff which was often much
  // smaller than the actual seed spacing.  This resulted in most seeds having
  // no neighbors and the preview showing many disjoint short struts. To make
  // the visualization robust, estimate a reasonable cutoff from the seed
  // distribution by averaging the nearest-neighbor distance.
  const edgePositions = useMemo(() => {
    if (seedPointsUnits.length === 0) return new Float32Array();

    let cutoff = spec.min_dist ? spec.min_dist * MM_TO_UNIT * 2 : 0;
    if (!spec.min_dist) {
      let total = 0;
      for (let i = 0; i < seedPointsUnits.length; i++) {
        let minD2 = Infinity;
        const [xi, yi, zi] = seedPointsUnits[i];
        for (let j = 0; j < seedPointsUnits.length; j++) {
          if (i === j) continue;
          const [xj, yj, zj] = seedPointsUnits[j];
          const dx = xi - xj;
          const dy = yi - yj;
          const dz = zi - zj;
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < minD2) minD2 = d2;
        }
        if (minD2 < Infinity) total += Math.sqrt(minD2);
      }
      const avg = total / seedPointsUnits.length;
      cutoff = avg * 1.5; // generous neighbor radius
    }

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

  const meshRef = useRef<THREE.InstancedMesh>(null!);

  // Build a 4x4 matrix per edge to position, orient, and scale a unit cylinder
  const edgeMatrices = useMemo(() => {
    const mats: THREE.Matrix4[] = [];
    const thickness = thicknessUnits;
    const up = new THREE.Vector3(0, 1, 0);
    for (let i = 0; i < edgePositions.length; i += 6) {
      const p1 = new THREE.Vector3(edgePositions[i], edgePositions[i+1], edgePositions[i+2]);
      const p2 = new THREE.Vector3(edgePositions[i+3], edgePositions[i+4], edgePositions[i+5]);
      const dir = p2.clone().sub(p1);
      const len = dir.length();
      dir.normalize();
      const mid = p1.clone().add(dir.clone().multiplyScalar(len * 0.5));
      const quat = new THREE.Quaternion().setFromUnitVectors(up, dir);
      const scale = new THREE.Vector3(thickness, len, thickness);
      const mat = new THREE.Matrix4().compose(mid, quat, scale);
      mats.push(mat);
    }
    return mats;
  }, [edgePositions, spec.min_dist]);

  // Apply the matrices to the InstancedMesh
  useEffect(() => {
    if (meshRef.current) {
      edgeMatrices.forEach((mat, idx) => {
        meshRef.current!.setMatrixAt(idx, mat);
      });
      meshRef.current.instanceMatrix.needsUpdate = true;
    }
  }, [edgeMatrices]);

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
      <instancedMesh ref={meshRef} args={[undefined, undefined, edgeMatrices.length]}>
        <cylinderGeometry args={[1, 1, 1, 8]} />
        <meshStandardMaterial color="white" />
      </instancedMesh>
      <points>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[pointsPositions, 3]} />
        </bufferGeometry>
        <pointsMaterial color="red" size={SEED_SPHERE_RADIUS_MM * MM_TO_UNIT * 2} sizeAttenuation={false} />
      </points>
    </>
  );
};

export default VoronoiLatticePreview;