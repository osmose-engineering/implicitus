import React, { useMemo, useRef, useEffect } from 'react';
import * as THREE from 'three';
import { InstancedMesh, Object3D, BufferGeometry, Float32BufferAttribute, MeshStandardMaterial } from 'three';
import triangulate from 'delaunay-triangulate';


export interface VoronoiStrutsProps {
  seedPoints: [number, number, number][];
  strutRadius: number;
  color?: string | number;
}

export const VoronoiStruts: React.FC<VoronoiStrutsProps> = ({ seedPoints, strutRadius, color = 'white' }) => {
  // 1️⃣ Compute Delaunay edges once
  const edges = useMemo(() => {
    // Compute 3D Delaunay tetrahedra
    const cells = triangulate(seedPoints);
    const seen = new Set<string>();
    const list: [ [number,number,number], [number,number,number] ][] = [];

    for (const cell of cells) {
      for (let i = 0; i < 4; i++) {
        for (let j = i + 1; j < 4; j++) {
          const a = cell[i], b = cell[j];
          const key = a < b ? `${a},${b}` : `${b},${a}`;
          if (!seen.has(key)) {
            seen.add(key);
            list.push([ seedPoints[a], seedPoints[b] ]);
          }
        }
      }
    }
    return list;
  }, [seedPoints]);

  // 2️⃣ Prepare instanced mesh
  const count = edges.length;
  const meshRef = useRef<InstancedMesh>(null!);
  const dummy = useMemo(() => new Object3D(), []);

  // 3️⃣ Geometry & material
  const geometry = useMemo(() => {
    // cylinder aligned along Y of unit length
    return new THREE.CylinderGeometry(1, 1, 1, 3, 1, true);
  }, []);
  const material = useMemo(() => new MeshStandardMaterial({ color }), [color]);

  // 4️⃣ Instance placement
  useEffect(() => {
    edges.forEach(([a, b], i) => {
      const pa = new THREE.Vector3(...a);
      const pb = new THREE.Vector3(...b);
      const dir = pb.clone().sub(pa);
      const len = dir.length();

      dummy.position.copy(pa).addScaledVector(dir, 0.5);
      dummy.scale.set(strutRadius, len, strutRadius);
      dummy.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        dir.normalize()
      );
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
  }, [edges, strutRadius]);

  return (
    <instancedMesh ref={meshRef} args={[geometry, material, count]} />
  );
};