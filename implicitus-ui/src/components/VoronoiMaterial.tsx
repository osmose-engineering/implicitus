import * as THREE from 'three';
import { shaderMaterial } from '@react-three/drei';
import { extend } from '@react-three/fiber';

const MAX_SEEDS = 512;
const texSize = Math.ceil(Math.sqrt(MAX_SEEDS));

// Create and configure seed DataTexture
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

export const VoronoiMaterial = shaderMaterial(
  {
    uSeedsTex:   { value: seedTexture },
    uTexSize:    { value: texSize },
    uNumSeeds:   { value: 0 },
    uBoxMin:     { value: new THREE.Vector3(0,0,0) },
    uBoxMax:     { value: new THREE.Vector3(0,0,0) },
    uThickness:  { value: 0.1 },
    uEdgeThickness: { value: 0.1 },
    uMaxSteps:   { value: 64 },
    uEpsilon:    { value: 0.001 }
  },
  // vertex shader
  `
  varying vec3 vWorldPos;
  void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPos = worldPos.xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
  `,
  // fragment shader
  `
#define MAX_SEEDS ${MAX_SEEDS}
precision highp float;
precision highp int;
varying vec3 vWorldPos;

uniform sampler2D uSeedsTex;
uniform int       uTexSize;
uniform int       uNumSeeds;
uniform vec3      uBoxMin;
uniform vec3      uBoxMax;
uniform float     uThickness;
uniform float     uEdgeThickness;
uniform int       uMaxSteps;
uniform float     uEpsilon;

// Box‐ray intersection
bool intersectBox(vec3 ro, vec3 rd, vec3 boxMin, vec3 boxMax, out float tNear, out float tFar) {
  vec3 inv = 1.0 / rd;
  vec3 t0s = (boxMin - ro) * inv;
  vec3 t1s = (boxMax - ro) * inv;
  vec3 tsmaller = min(t0s, t1s);
  vec3 tbigger  = max(t0s, t1s);
  tNear = max(max(tsmaller.x, tsmaller.y), tsmaller.z);
  tFar  = min(min(tbigger.x,  tbigger.y),  tbigger.z);
  return tNear < tFar && tFar > 0.0;
}

// SDF lookup functions remain unchanged
vec3 getSeed(int i) {
  float fi = float(i);
  float x = mod(fi, float(uTexSize)) + 0.5;
  float y = floor(fi / float(uTexSize)) + 0.5;
  vec2 uv = vec2(x, y) / float(uTexSize);
  return texture(uSeedsTex, uv).rgb;
}

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
  // Signed distance to the planar Voronoi boundary, offset by thickness
  float sd = (d2 - d1) * 0.5 - uThickness;
  return sd;
}

void main() {
  // Ray origin and direction in world-space
  vec3 ro = cameraPosition;
  vec3 rd = normalize(vWorldPos - ro);

  // Intersect ray with box volume
  float t0, t1;
  if (!intersectBox(ro, rd, uBoxMin, uBoxMax, t0, t1)) {
    discard;
  }
  float t = max(t0, 0.0);
  // Ray-march loop
  for (int i = 0; i < uMaxSteps; i++) {
    vec3 p = ro + rd * t;
    float d = voronoiSDF(p);
    // Highlight shell: where distance is near zero
    if (abs(d) < uEdgeThickness) {
      gl_FragColor = vec4(vec3(1.0), 1.0);
      return;
    }
    t += max(abs(d), uEpsilon);
    if (t > t1) break;
  }
  discard;
}
`
);

extend({ VoronoiMaterial });