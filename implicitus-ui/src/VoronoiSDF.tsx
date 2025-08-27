import React, { useMemo } from 'react'
import { useFrame, extend } from '@react-three/fiber'
import * as THREE from 'three'
import { DataTexture, RGBAFormat, FloatType } from 'three'
import { VoronoiLatticeMaterial } from './VoronoiLatticeMaterial'
extend({ VoronoiLatticeMaterial })

const MAX_SEEDS = 1024;
const MM_TO_UNIT = 0.1;

type Props = {
  spec: {
    seed_points?: Array<[number, number, number]>;
    wall_thickness?: number;
    min_dist?: number;
  };
  bounds: [number, number, number][];
  maxSteps?: number;
  epsilon?: number;
};

export default function VoronoiSDF({
  spec,
  bounds,
  maxSteps = 64,
  epsilon = 0.001,
}: Props) {
  const seedPoints = spec.seed_points ?? [];
  const thickness = ((spec.wall_thickness ?? spec.min_dist ?? 1) as number) * MM_TO_UNIT;
  const [bboxMin, bboxMax] = bounds;
  const seedsArray = useMemo(() => {
    const pts = seedPoints || [];
    const arr = new Float32Array(pts.length * 3);
    pts.forEach((p, i) => {
      arr[i * 3 + 0] = p[0];
      arr[i * 3 + 1] = p[1];
      arr[i * 3 + 2] = p[2];
    });
    return arr;
  }, [seedPoints]);

  const numSeeds = (seedsArray.length / 3) | 0;

  // DataTexture for seeds
  const texSize = Math.max(1, Math.ceil(Math.sqrt(numSeeds)));
  const texData = useMemo(() => {
    const dt = new Float32Array(texSize * texSize * 4);
    for (let i = 0; i < seedsArray.length / 3; i++) {
      dt[i * 4 + 0] = seedsArray[i * 3 + 0];
      dt[i * 4 + 1] = seedsArray[i * 3 + 1];
      dt[i * 4 + 2] = seedsArray[i * 3 + 2];
      dt[i * 4 + 3] = 0;
    }
    return dt;
  }, [seedsArray, texSize]);
  const seedsTexture = useMemo(() => {
    const tex = new DataTexture(texData, texSize, texSize, RGBAFormat, FloatType);
    tex.magFilter = THREE.NearestFilter;
    tex.minFilter = THREE.NearestFilter;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.flipY = false;
    tex.needsUpdate = true;
    return tex;
  }, [texData, texSize]);

  // Debug: log essential parameters when they change
  React.useEffect(() => {
    console.log('VoronoiSDF params', {
      seedPointsCount: seedPoints.length,
      numSeeds,
      texSize,
      bboxMin,
      bboxMax,
      thickness,
      maxSteps,
      epsilon,
    });
  }, [seedPoints.length, numSeeds, texSize, bboxMin, bboxMax, thickness, maxSteps, epsilon]);

  const [minX, minY, minZ] = bboxMin;
  const [maxX, maxY, maxZ] = bboxMax;

  const materialRef = React.useRef<THREE.ShaderMaterial>(null)
  // Debug: log uniform values once on mount
  React.useEffect(() => {
    const mat = materialRef.current;
    if (mat?.uniforms) {
      const u = mat.uniforms;
      console.log('VoronoiSDF uniforms', {
        uNumSeeds:  u.uNumSeeds?.value,
        uTexSize:   u.uTexSize?.value,
        uBoxMin:    u.uBoxMin?.value,
        uBoxMax:    u.uBoxMax?.value,
        uThickness: u.uThickness?.value,
        uMaxSteps:  u.uMaxSteps?.value,
        uEpsilon:   u.uEpsilon?.value,
      });
    }
  }, []);
  useFrame(() => {
    // console.log('VoronoiSDF useFrame start', {
    //   mat: materialRef.current,
    //   numSeeds,
    //   seedsArrayLength: seedsArray.length,
    //   texSize,
    //   uniforms: materialRef.current?.uniforms
    // });
    const mat = materialRef.current
    if (!mat || !mat.uniforms) return

    // Update seed positions via DataTexture instead of array uniforms
    if (mat.uniforms.uSeedsTex) {
      const tex = mat.uniforms.uSeedsTex.value as DataTexture;
      const texData = tex.image.data as Float32Array;
      texData.fill(0);
      const count = Math.min(numSeeds, MAX_SEEDS);
      for (let i = 0; i < count; i++) {
        texData[i * 4 + 0] = seedsArray[i * 3 + 0];
        texData[i * 4 + 1] = seedsArray[i * 3 + 1];
        texData[i * 4 + 2] = seedsArray[i * 3 + 2];
      }
      tex.needsUpdate = true;
      mat.uniforms.uNumSeeds.value = count;
      // console.log('Updated seed texture and count', {
      //   uNumSeeds: mat.uniforms.uNumSeeds.value,
      //   sampleFirstSeed: [texData[0], texData[1], texData[2]]
      // });
    }

    // Box min
    if (mat.uniforms.uBoxMin) {
      const val = mat.uniforms.uBoxMin.value
      if (typeof val.set === 'function') {
        val.set(minX, minY, minZ)
      } else {
        mat.uniforms.uBoxMin.value = new THREE.Vector3(minX, minY, minZ)
      }
    }
    // Box max
    if (mat.uniforms.uBoxMax) {
      const val = mat.uniforms.uBoxMax.value
      if (typeof val.set === 'function') {
        val.set(maxX, maxY, maxZ)
      } else {
        mat.uniforms.uBoxMax.value = new THREE.Vector3(maxX, maxY, maxZ)
      }
    }
    // console.log('Box min/max', {
    //   uBoxMin: mat.uniforms.uBoxMin.value,
    //   uBoxMax: mat.uniforms.uBoxMax.value
    // });
    if (mat.uniforms.uThickness) mat.uniforms.uThickness.value = thickness
    if (mat.uniforms.uMaxSteps) mat.uniforms.uMaxSteps.value = maxSteps
    if (mat.uniforms.uEpsilon) mat.uniforms.uEpsilon.value = epsilon
    // console.log('Other uniforms', {
    //   uThickness: mat.uniforms.uThickness?.value,
    //   uMaxSteps: mat.uniforms.uMaxSteps?.value,
    //   uEpsilon: mat.uniforms.uEpsilon?.value
    // });
  })

  // The box that bounds our lattice volume
  const size: [number,number,number] = [
    maxX - minX,
    maxY - minY,
    maxZ - minZ,
  ];

  return (
    <mesh>
      <boxGeometry args={size} />
      <voronoiLatticeMaterial
        attach="material"
        ref={materialRef}
        uSeedsTex={seedsTexture}
        uNumSeeds={numSeeds}
        uTexSize={texSize}
      />
    </mesh>
  )
}