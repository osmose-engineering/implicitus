import React, { useMemo, useEffect, useRef } from 'react';
import * as THREE from 'three';


export interface VoronoiStrutsProps {
  vertices: [number, number, number][];
  edges: [number, number][];      // list of index pairs
  strutRadius: number;
  color?: string | number;
}

export const VoronoiStruts: React.FC<VoronoiStrutsProps> = ({
  vertices,
  edges = [],
  strutRadius,
  color = 'white'
}) => {
  // Convert index‐pairs to coordinate pairs from Voronoi vertices
  const edgePairs = useMemo(() => {
    return (edges || []).map(([i, j]) => [
      vertices[i] as [number, number, number],
      vertices[j] as [number, number, number]
    ]);
  }, [edges, vertices]);

  const meshRef = useRef<THREE.InstancedMesh>(null!);

  useEffect(() => {
    const mesh = meshRef.current;
    // ensure instance count matches dynamic edgePairs length
    mesh.count = edgePairs.length;
    // Position and scale instances
    edgePairs.forEach(([a, b], i) => {
      const pa = new THREE.Vector3(...a);
      const pb = new THREE.Vector3(...b);
      const dir = pb.clone().sub(pa);
      const len = dir.length();
      // mid-point
      const mid = pa.add(dir.multiplyScalar(0.5));
      // orientation
      const quat = new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        dir.normalize()
      );
      // build matrix
      const mat = new THREE.Matrix4();
      mat.compose(
        mid,
        quat,
        new THREE.Vector3(strutRadius, len, strutRadius)
      );
      mesh.setMatrixAt(i, mat);
    });
    mesh.instanceMatrix.needsUpdate = true;
    // Apply uniform color prop
    (mesh.material as THREE.MeshBasicMaterial).color.set(color as any);
  }, [edgePairs, color, strutRadius]);

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, edgePairs.length]}>
      <cylinderGeometry args={[1, 1, 1, 8, 1]} />
      <meshBasicMaterial
        vertexColors={false}
        color={color as any}
      />
    </instancedMesh>
  );
};
