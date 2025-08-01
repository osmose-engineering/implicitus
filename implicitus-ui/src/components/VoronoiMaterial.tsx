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
    uSeedsTex: seedTexture,
    uTexSize: texSize,
    uNumSeeds: 0,
    uBoxMin: new THREE.Vector3(0, 0, 0),
    uBoxMax: new THREE.Vector3(0, 0, 0),
    uThickness: 0.1,
    uEdgeThickness: 0.02,
    uMaxSteps: 64,
    uEpsilon: 0.001,
    uShowSolid: true,
    uShowInfill: true,
    uShowAccumulate: true,
    uSphereCenter: new THREE.Vector3(0, 0, 0),
    uSphereRadius: 1.0,
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
uniform bool      uShowSolid;
uniform bool      uShowInfill;
uniform bool      uShowAccumulate;
uniform vec3      uSphereCenter;
uniform float     uSphereRadius;

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

// Sphere‐ray intersection
bool intersectSphere(vec3 ro, vec3 rd, vec3 center, float radius, out float tNear, out float tFar) {
  vec3 oc = ro - center;
  float b = dot(oc, rd);
  float c = dot(oc, oc) - radius * radius;
  float disc = b * b - c;
  if (disc < 0.0) return false;
  float sqrtD = sqrt(disc);
  tNear = -b - sqrtD;
  tFar  = -b + sqrtD;
  return tFar > 0.0;
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

// Shape SDF for a sphere
float sphereSDF(vec3 p) {
  return length(p - uSphereCenter) - uSphereRadius;
}

void main() {
  vec3 ro = cameraPosition;
  vec3 rd = normalize(vWorldPos - ro);

  // Intersect with box and sphere
  float t0b, t1b, t0s, t1s;
  intersectBox(ro, rd, uBoxMin, uBoxMax, t0b, t1b);
  if (!intersectSphere(ro, rd, uSphereCenter, uSphereRadius, t0s, t1s)) {
    discard;
  }
  float t0 = max(max(t0b, t0s), 0.0);
  float t1 = min(t1b, t1s);
  if (t1 < t0) discard;
  float tEntry = t0;
  float tExit  = t1;
  float t       = tEntry;

  float acc = 0.0;
  for (int i = 0; i < uMaxSteps; i++) {
    vec3 p = ro + rd * t;
    float vorD   = voronoiSDF(p);
    float shapeD = sphereSDF(p);
    bool insideShape = (shapeD < 0.0);
    bool wall       = insideShape && abs(vorD) < uEdgeThickness;

    // Solid branch
    if (uShowSolid && insideShape) {
      float depthNorm = clamp((t - tEntry) / (tExit - tEntry), 0.0, 1.0);
      float shade = pow(1.0 - depthNorm, 2.2);
      gl_FragColor = vec4(vec3(shade), 1.0);
      return;
    }
    // Infill branch
    if (uShowInfill) {
      if (uShowAccumulate) {
        if (wall) acc += 1.0;
      } else {
        if (wall) {
          float depthNorm = clamp((t - tEntry) / (tExit - tEntry), 0.0, 1.0);
          float shade = pow(1.0 - depthNorm, 2.2);
          gl_FragColor = vec4(vec3(shade), 1.0);
          return;
        }
      }
    }

    t += max(abs(vorD), uEpsilon);
    if (t > t1) break;
  }
  // After marching, if accumulating, draw intensity
  if (uShowAccumulate && acc > 0.0) {
    float intensity = acc / float(uMaxSteps);
    gl_FragColor = vec4(vec3(intensity), 1.0);
    return;
  }
  discard;
}
`
);

extend({ VoronoiMaterial });