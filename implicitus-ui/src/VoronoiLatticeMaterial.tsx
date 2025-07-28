// components/VoronoiLatticeMaterial.tsx
import { shaderMaterial } from '@react-three/drei'
import * as THREE from 'three'
import glsl from 'babel-plugin-glsl/macro'
import { extend } from '@react-three/fiber'

const MAX_SEEDS = 512
const texSize = Math.ceil(Math.sqrt(MAX_SEEDS));

// Create and configure seed DataTexture with nearest filtering and clamp wrapping
const seedTexture = new THREE.DataTexture(
  new Float32Array(texSize * texSize * 4),
  texSize, texSize,
  THREE.RGBAFormat,
  THREE.FloatType
);
seedTexture.magFilter = THREE.NearestFilter;
seedTexture.minFilter = THREE.NearestFilter;
seedTexture.wrapS    = THREE.ClampToEdgeWrapping;
seedTexture.wrapT    = THREE.ClampToEdgeWrapping;
seedTexture.flipY    = false;
seedTexture.needsUpdate = true;

export const VoronoiLatticeMaterial = shaderMaterial(
  // --- Uniforms
  {
    uSeedsTex:     { value: seedTexture },
    uTexSize:      { value: texSize },
    uNumSeeds:     { value: 0 },
    uBoxMin:       { value: new THREE.Vector3(-1, -1, -1) },
    uBoxMax:       { value: new THREE.Vector3( 1,  1,  1) },
    uThickness:    { value: 0.1 },
    uMaxSteps:     { value: 64 },
    uEpsilon:      { value: 0.001 },
  },
  // --- Vertex Shader (pass‐through)
  glsl`
    varying vec3 vWorldPos;
    void main() {
      vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
      gl_Position = projectionMatrix * viewMatrix * vec4(vWorldPos, 1.0);
    }
  `,
  // --- Fragment Shader (raymarching Voronoi SDF)
  glsl`
    #define MAX_SEEDS ${MAX_SEEDS}   // adjust to your cap
    precision highp float;
    precision highp int;

    uniform int    uNumSeeds;
    uniform vec3   uBoxMin;
    uniform vec3   uBoxMax;
    uniform float  uThickness;
    uniform int    uMaxSteps;
    uniform float  uEpsilon;

    uniform sampler2D uSeedsTex;
    uniform int       uTexSize;

    varying vec3 vWorldPos;

    vec3 getSeed(int i) {
      float fi = float(i);
      float x = mod(fi, float(uTexSize)) + 0.5;
      float y = floor(fi / float(uTexSize)) + 0.5;
      vec2 uv = vec2(x, y) / float(uTexSize);
      return texture(uSeedsTex, uv).rgb;
    }

    // compute signed‐distance to the nearest Voronoi wall
    float voronoiSDF(vec3 p) {
      float d1 = 1e20;
      float d2 = 1e20;
      for (int i = 0; i < MAX_SEEDS; ++i) {
        if (i >= uNumSeeds) break;
        float d = distance(p, getSeed(i));
        if (d < d1) {
          d2 = d1;
          d1 = d;
        } else if (d < d2) {
          d2 = d;
        }
      }
      return (d2 - d1) * 0.5;
    }

    // Ray‐slice through the box
    void main() {
      // reconstruct ray origin & dir in world‐space
      vec3 ro = cameraPosition;
      vec3 rd = normalize(vWorldPos - cameraPosition);

      float t = 0.0;
      for (int i = 0; i < uMaxSteps; ++i) {
        vec3 p = ro + rd * t;
        // exit if outside the box bounding
        if (any(lessThan(p, uBoxMin)) || any(greaterThan(p, uBoxMax))) {
          discard;
        }
        float d = voronoiSDF(p) - uThickness * 0.5;
        if (d < uEpsilon) {
          // shade the wall
          gl_FragColor = vec4(1.0, 0.75, 0.2, 1.0);
          return;
        }
        t += d;
      }
      discard;
    }
  `
)

// Tell drei to inject this material into R3F’s JSX registry
extend({ VoronoiLatticeMaterial })