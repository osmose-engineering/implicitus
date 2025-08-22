import React, { useMemo, useEffect } from 'react';
import * as THREE from 'three';
import { useFrame, useThree } from '@react-three/fiber';
import { VoronoiMaterial } from './VoronoiMaterial';

const DEBUG = false;

const MAX_SEEDS = 512;

interface VoronoiMeshProps {
  seedPoints: [number, number, number][];
  /** Offset for the Voronoi SDF */
  thickness?: number;
  /** Upper bound on ray-march iterations. Larger values improve accuracy at the cost of speed. */
  maxSteps?: number;
  /** Minimum increment in the ray marcher */
  epsilon?: number;
  showSolid?: boolean;
  showInfill?: boolean;
  sphereCenter?: [number, number, number];
  sphereRadius?: number;
}

const VoronoiMesh: React.FC<VoronoiMeshProps> = ({
  seedPoints,
  thickness = 0.1,
  maxSteps = 1024,
  epsilon = 0.001,
  showSolid = true,
  showInfill = true,
  sphereCenter,
  sphereRadius
}) => {
  // If there are no seed points, there is nothing to render.
  // Returning early avoids creating geometries with infinite or NaN
  // dimensions which would stall the renderer and hide the grid.
  if (!seedPoints || seedPoints.length === 0) {
    return null;
  }
  // Derive world-space bounding box from seedPoints
  const xs = seedPoints.map(p => p[0]);
  const ys = seedPoints.map(p => p[1]);
  const zs = seedPoints.map(p => p[2]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const minZ = Math.min(...zs);
  const maxZ = Math.max(...zs);

  // Default sphere center & radius if not provided
  const defaultCenter = new THREE.Vector3(
    (minX + maxX) / 2,
    (minY + maxY) / 2,
    (minZ + maxZ) / 2
  );
  const centerVec = sphereCenter
    ? new THREE.Vector3(...sphereCenter)
    : defaultCenter;
  const radiusVal = sphereRadius !== undefined
    ? sphereRadius
    : Math.min(maxX - minX, maxY - minY, maxZ - minZ) / 2;

  const numSeeds = seedPoints.length;
  const count = Math.min(MAX_SEEDS, numSeeds);
  const texSize = Math.max(1, Math.ceil(Math.sqrt(count)));
  const seedsArray = useMemo(() => new Float32Array(seedPoints.flat()), [seedPoints]);
  const texData = useMemo(() => new Float32Array(texSize * texSize * 4), [texSize]);

  const seedTexture = useMemo(() => {
    const tex = new THREE.DataTexture(texData, texSize, texSize, THREE.RGBAFormat, THREE.FloatType);
    tex.magFilter = THREE.NearestFilter;
    tex.minFilter = THREE.NearestFilter;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.flipY = false;
    tex.needsUpdate = true;
    return tex;
  }, [texData, texSize]);

  // Nearest‐neighbor seed spacing diagnostics
  useEffect(() => {
    if (!DEBUG) return;
    const dists = seedPoints.map((p, i) => {
      let m = Infinity;
      for (let j = 0; j < seedPoints.length; j++) {
        if (i === j) continue;
        const dx = p[0] - seedPoints[j][0];
        const dy = p[1] - seedPoints[j][1];
        const dz = p[2] - seedPoints[j][2];
        m = Math.min(m, Math.hypot(dx, dy, dz));
      }
      return m;
    });
    const minD = Math.min(...dists);
    const maxD = Math.max(...dists);
    const avgD = dists.reduce((a, b) => a + b, 0) / dists.length;
    console.log('NN spacing (mm):', { minD, avgD, maxD });
  }, [seedPoints]);

  const material = useMemo(() => {
    const m = new VoronoiMaterial();
    // Bind the dynamic DataTexture of seeds
    m.uniforms.uSeedsTex.value = seedTexture;
    m.uniforms.uSeedsTex.needsUpdate = true;
    // Initialize SDF offset thickness (from prop) and a thin edge thickness
    m.uniforms.uThickness.value = thickness;
    m.uniforms.uThickness.needsUpdate = true;
    m.uniforms.uEdgeThickness.value = thickness;
    m.uniforms.uEdgeThickness.needsUpdate = true;
    // Wire up visibility toggles
    m.uniforms.uShowSolid.value = showSolid;
    m.uniforms.uShowSolid.needsUpdate = true;
    m.uniforms.uShowInfill.value = showInfill;
    m.uniforms.uShowInfill.needsUpdate = true;
    // Bind sphere shape parameters
    m.uniforms.uSphereCenter.value.copy(centerVec);
    m.uniforms.uSphereCenter.needsUpdate = true;
    m.uniforms.uSphereRadius.value = radiusVal;
    m.uniforms.uSphereRadius.needsUpdate = true;
    return m;
  }, [seedTexture, thickness, showSolid, showInfill, centerVec, radiusVal]);

  const { camera } = useThree();

  useFrame(() => {
    if (DEBUG) {
      // 🛠 Voronoi Debug
      console.log('🛠 Voronoi Debug', {
        cameraPosition: camera.position.toArray(),
        bbox: [minX, minY, minZ, maxX, maxY, maxZ],
        seedCount: count,
        firstSeed: seedsArray.slice(0, 3),
        thickness,
        edgeThickness: material.uniforms.uEdgeThickness.value,
        maxSteps,
        epsilon,
      });

      // Nearest-neighbor spacing debug inside frame
      {
        const dists = seedPoints.map((p, i) => {
          let m = Infinity;
          for (let j = 0; j < seedPoints.length; j++) {
            if (i === j) continue;
            const dx = p[0] - seedPoints[j][0];
            const dy = p[1] - seedPoints[j][1];
            const dz = p[2] - seedPoints[j][2];
            m = Math.min(m, Math.hypot(dx, dy, dz));
          }
          return m;
        });
        const minD = Math.min(...dists);
        const maxD = Math.max(...dists);
        const avgD = dists.reduce((a, b) => a + b, 0) / dists.length;
        console.log('NN spacing (mm):', { minD, avgD, maxD });
      }

      // Quick SDF and ray‐box diagnostics
      // Sample SDF at the box center
      const center = new THREE.Vector3(
        (minX + maxX) / 2,
        (minY + maxY) / 2,
        (minZ + maxZ) / 2
      );
      // Compute minimal distance from center to seeds
      const minDist = Math.min(
        ...seedPoints.map(([x, y, z]) =>
          Math.hypot(x - center.x, y - center.y, z - center.z)
        )
      );
      const sdfAtCenter = minDist - thickness;
      console.log('SDF @ center:', sdfAtCenter);

      // Compute ray‐box intersection for a ray from camera through center
      const ro = camera.position;
      const rd = center.clone().sub(ro).normalize();
      // Compute t0/t1 slabs
      const inv = new THREE.Vector3(1 / rd.x, 1 / rd.y, 1 / rd.z);
      const t0s = new THREE.Vector3(
        (minX - ro.x) * inv.x,
        (minY - ro.y) * inv.y,
        (minZ - ro.z) * inv.z
      );
      const t1s = new THREE.Vector3(
        (maxX - ro.x) * inv.x,
        (maxY - ro.y) * inv.y,
        (maxZ - ro.z) * inv.z
      );
      const tmin = Math.max(
        Math.min(t0s.x, t1s.x),
        Math.min(t0s.y, t1s.y),
        Math.min(t0s.z, t1s.z)
      );
      const tmax = Math.min(
        Math.max(t0s.x, t1s.x),
        Math.max(t0s.y, t1s.y),
        Math.max(t0s.z, t1s.z)
      );
      console.log('Ray‐box t0/t1:', tmin, tmax);

      // SDF at ray entry point diagnostics
      const entryPoint = ro.clone().add(rd.clone().multiplyScalar(tmin));
      const entryDist = Math.min(
        ...seedPoints.map(([x, y, z]) =>
          Math.hypot(x - entryPoint.x, y - entryPoint.y, z - entryPoint.z)
        )
      ) - thickness;
      console.log('SDF @ entry:', entryDist);
    }

    const mat = material;
    if (!mat) return;

    // Estimate ray-march step count based on current camera position
    const ro = camera.position;
    const boxCenter = new THREE.Vector3(
      (minX + maxX) / 2,
      (minY + maxY) / 2,
      (minZ + maxZ) / 2
    );
    const rd = boxCenter.clone().sub(ro).normalize();

    // Box intersection
    const inv = new THREE.Vector3(1 / rd.x, 1 / rd.y, 1 / rd.z);
    const t0s = new THREE.Vector3(
      (minX - ro.x) * inv.x,
      (minY - ro.y) * inv.y,
      (minZ - ro.z) * inv.z
    );
    const t1s = new THREE.Vector3(
      (maxX - ro.x) * inv.x,
      (maxY - ro.y) * inv.y,
      (maxZ - ro.z) * inv.z
    );
    const tNear = Math.max(
      Math.min(t0s.x, t1s.x),
      Math.min(t0s.y, t1s.y),
      Math.min(t0s.z, t1s.z)
    );
    const tFar = Math.min(
      Math.max(t0s.x, t1s.x),
      Math.max(t0s.y, t1s.y),
      Math.max(t0s.z, t1s.z)
    );

    // Sphere intersection
    let tSphereNear = Infinity;
    let tSphereFar = -Infinity;
    {
      const oc = ro.clone().sub(centerVec);
      const b = oc.dot(rd);
      const c = oc.lengthSq() - radiusVal * radiusVal;
      const disc = b * b - c;
      if (disc >= 0) {
        const sqrtD = Math.sqrt(disc);
        tSphereNear = -b - sqrtD;
        tSphereFar = -b + sqrtD;
      }
    }

    const t0 = Math.max(Math.max(tNear, tSphereNear), 0);
    const t1 = Math.min(tFar, tSphereFar);
    const travelDist = Math.max(0, t1 - t0);

    let computedSteps = Math.ceil(travelDist / epsilon);
    if (!isFinite(computedSteps) || computedSteps < 1) computedSteps = 1;
    const clampedSteps = Math.min(computedSteps, maxSteps);

    // Update texture
    texData.fill(0);
    for (let i = 0; i < count; i++) {

      texData[i * 4 + 0] = seedsArray[i * 3 + 0];
      texData[i * 4 + 1] = seedsArray[i * 3 + 1];
      texData[i * 4 + 2] = seedsArray[i * 3 + 2];
    }
    seedTexture.needsUpdate = true;
    
    // Debug: are we bound to the right DataTexture?
    //{
    //  const bound = mat.uniforms.uSeedsTex?.value;
    //  if (bound instanceof THREE.DataTexture && bound.image && bound.image.data) {
    //    const texDataArray = bound.image.data as Float32Array;
    //    console.log(
    //      'BIND CHECK:',
    //      bound === seedTexture,
    //      texDataArray.slice(0, 6)
    //    );
    //  } else {
    //    console.log('BIND CHECK: No valid seed texture bound:', bound);
    //  }
    //}

    // Update uniforms
    mat.uniforms.uNumSeeds.value = count;
    mat.uniforms.uNumSeeds.needsUpdate = true;
    mat.uniforms.uBoxMin.value = new THREE.Vector3(minX, minY, minZ);
    mat.uniforms.uBoxMin.needsUpdate = true;
    mat.uniforms.uBoxMax.value = new THREE.Vector3(maxX, maxY, maxZ);
    mat.uniforms.uBoxMax.needsUpdate = true;
    mat.uniforms.uThickness.value = thickness;
    mat.uniforms.uThickness.needsUpdate = true;
    mat.uniforms.uMaxSteps.value = clampedSteps;
    mat.uniforms.uMaxSteps.needsUpdate = true;
    mat.uniforms.uEpsilon.value = epsilon;
    mat.uniforms.uEpsilon.needsUpdate = true;
    mat.uniforms.uShowSolid.value = showSolid;
    mat.uniforms.uShowSolid.needsUpdate = true;
    mat.uniforms.uShowInfill.value = showInfill;
    mat.uniforms.uShowInfill.needsUpdate = true;

    // Update sphere shape uniforms
    mat.uniforms.uSphereCenter.value.copy(centerVec);
    mat.uniforms.uSphereCenter.needsUpdate = true;
    mat.uniforms.uSphereRadius.value = radiusVal;
    mat.uniforms.uSphereRadius.needsUpdate = true;

    //console.log('Uniforms:', {
      //uNumSeeds: mat.uniforms.uNumSeeds.value,
      //uBoxMin:   mat.uniforms.uBoxMin.value.toArray ? mat.uniforms.uBoxMin.value.toArray() : mat.uniforms.uBoxMin.value,
      //uBoxMax:   mat.uniforms.uBoxMax.value.toArray ? mat.uniforms.uBoxMax.value.toArray() : mat.uniforms.uBoxMax.value
    //});
  });

  return (
    <mesh
      material={material}
      position={[
        (minX + maxX) / 2,
        (minY + maxY) / 2,
        (minZ + maxZ) / 2
      ]}
    >
      <boxGeometry args={[maxX - minX, maxY - minY, maxZ - minZ]} />
    </mesh>
  );
};

export default VoronoiMesh;
