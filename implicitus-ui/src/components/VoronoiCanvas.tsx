import React, { useMemo, useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import VoronoiMesh from './VoronoiMesh';
import { VoronoiStruts } from './VoronoiStruts';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

// Debug: override with a small hex‐lattice test instead of server data
const DEBUG_HEX_TEST = false;
function generateHexTest(bboxMin, bboxMax, spacing) {
// Debug: draw outlines of a 3×3 hexagonal patch of cells
  const pts = [];
  const edges = [];
  const sqrt3 = Math.sqrt(3);
  const cols = 3;
  const rows = 3;
  // center offsets to position patch in the view
  const cx0 = (bboxMin[0] + bboxMax[0]) / 2 - spacing * 3 / 2;
  const cy0 = (bboxMin[1] + bboxMax[1]) / 2 - (spacing * sqrt3 * (rows + 0.5)) / 2;
  const cz = (bboxMin[2] + bboxMax[2]) / 2;

  // generate each cell's outline
  for (let q = 0; q < cols; q++) {
    for (let r = 0; r < rows; r++) {
      // compute cell center
      const cx = cx0 + spacing * (3/2 * q);
      const cy = cy0 + spacing * (sqrt3 * (r + q/2));
      // record base index
      const base = pts.length;
      // 6 corners of one cell
      for (let i = 0; i < 6; i++) {
        const angle = Math.PI/3 * i; // flat-top orientation, no rotation
        const x = cx + spacing * Math.cos(angle);
        const y = cy + spacing * Math.sin(angle);
        pts.push([x, y, cz]);
      }
      // edges around the cell
      for (let i = 0; i < 6; i++) {
        edges.push([base + i, base + ((i + 1) % 6)]);
      }
    }
  }
  return [pts, edges];
}

// Debug: generate a small 3D hexagonal block for testing
export function generateHexTest3D(bboxMin, bboxMax, spacing) {
  const pts = [];
  const edges = [];
  const coords3D = [];
  const sqrt3 = Math.sqrt(3);
  const dz = spacing * Math.sqrt(6) / 3;

  // center the block
  const cx0 = (bboxMin[0] + bboxMax[0]) / 2;
  const cy0 = (bboxMin[1] + bboxMax[1]) / 2;
  const cz0 = (bboxMin[2] + bboxMax[2]) / 2 - dz / 2;

  const R = 1;       // hex radius per slice
  const layers = 2;  // number of vertical layers

  // generate points
  for (let k = 0; k < layers; k++) {
    const zk = cz0 + k * dz;
    for (let q = -R; q <= R; q++) {
      const r1 = Math.max(-R, -q - R);
      const r2 = Math.min(R, -q + R);
      for (let r = r1; r <= r2; r++) {
        const x = cx0 + spacing * (3/2 * q);
        const y = cy0 + spacing * (sqrt3 * (r + q/2));
        pts.push([x, y, zk]);
        coords3D.push([q, r, k]);
      }
    }
  }

  // 1) in-plane neighbors: idx<jdx & same k-layer & distance ≈ spacing
  const thr2Plane = spacing * 1.01;
  for (let i = 0; i < pts.length; i++) {
    for (let j = i + 1; j < pts.length; j++) {
      const [qi, ri, ki] = coords3D[i];
      const [qj, rj, kj] = coords3D[j];
      if (ki === kj) {
        // same layer: threshold using flat neighbor distance (√3*spacing)
        const dx = pts[i][0] - pts[j][0];
        const dy = pts[i][1] - pts[j][1];
        if (dx*dx + dy*dy <= (spacing * Math.sqrt(3) * 1.01)**2) {
          edges.push([i, j]);
        }
      }
    }
  }
  // 2) vertical neighbors: same (q,r), adjacent layers
  for (let i = 0; i < coords3D.length; i++) {
    const [qi, ri, ki] = coords3D[i];
    for (let j = i + 1; j < coords3D.length; j++) {
      const [qj, rj, kj] = coords3D[j];
      if (qi === qj && ri === rj && Math.abs(ki - kj) === 1) {
        edges.push([i, j]);
      }
    }
  }

  return [pts, edges];
}

// Robust statistic helper -------------------------------------------------
// Exported so tests or other modules can reason about the thresholds.
export function robustEdgeThreshold(lengths: number[], factor: number) {
  if (lengths.length === 0) return Infinity;
  const sorted = [...lengths].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median =
    sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];
  const q1 = sorted[Math.floor((sorted.length - 1) * 0.25)];
  const q3 = sorted[Math.floor((sorted.length - 1) * 0.75)];
  const iqr = q3 - q1;
  return median + factor * iqr;
}

// Exposed for testing: remove unusually long edges using a robust threshold.
// We also separate vertical edges so that legitimately tall struts aren’t
// discarded simply because the z-range is large compared to x/y.
export function computeFilteredEdges(
  points: [number, number, number][],
  edges: [number, number][],
  thresholdFactor: number = 1.5,
  zTolerance: number = EDGE_Z_VARIATION_TOLERANCE
) {
  if (!Array.isArray(edges) || edges.length === 0) return [];
  type EdgeInfo = { edge: [number, number]; length: number; vertical: boolean };
  const data: EdgeInfo[] = [];
  const verticalLengths: number[] = [];
  const otherLengths: number[] = [];

  for (const [i, j] of edges) {
    const pi = points[i];
    const pj = points[j];
    if (!Array.isArray(pi) || !Array.isArray(pj)) continue;
    const [xi, yi, zi] = pi;
    const [xj, yj, zj] = pj;
    const dx = xi - xj,
      dy = yi - yj,
      dz = zi - zj;
    if (Math.abs(dz) < zTolerance) {
      // Skip edges that fail to span meaningfully in the z-axis.
      continue;
    }
    const length = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const vertical = Math.abs(dz) > Math.max(Math.abs(dx), Math.abs(dy));
    data.push({ edge: [i, j], length, vertical });
    (vertical ? verticalLengths : otherLengths).push(length);
  }

  if (data.length === 0) return [];

  // Compute robust thresholds for vertical vs. non-vertical edges. Using the
  // median and IQR keeps the estimate stable even when a few edges are wildly
  // longer than the rest.
  const vThreshold = robustEdgeThreshold(verticalLengths, thresholdFactor);
  const oThreshold = robustEdgeThreshold(otherLengths, thresholdFactor);

  return data
    .filter(({ length, vertical }) => length <= (vertical ? vThreshold : oThreshold))
    .map(({ edge }) => edge);
}

// Threshold for detecting edges with effectively zero variation along the z-axis.
// Edges flatter than this tolerance are usually an indication that upstream
// geometry generation collapsed the edge onto a plane.  The value is exported so
// tests can dial it in or reference it directly.
export const EDGE_Z_VARIATION_TOLERANCE = 1e-5;

interface VoronoiCanvasProps {
  seedPoints: [number, number, number][];
  vertices?: [number, number, number][];
  edges?: [number, number][];
  bbox: [number, number, number, number, number, number];
  thickness?: number;
  maxSteps?: number;
  epsilon?: number;
  showStruts?: boolean;
  showSolid?: boolean;
  showInfill?: boolean;
  strutRadius?: number;
  strutColor?: string | number;
  infillPoints?: [number, number, number][];
  infillEdges?: [number, number][];
  /** Optional full cell geometry */
  cells?: Array<{ verts: number[][]; faces: number[][] }>;
  /** Optional multiplier for edge-length filtering */
  edgeLengthThreshold?: number;
  /** Z-variation tolerance for filtering nearly flat edges */
  edgeZVariationTolerance?: number;
}

// Toggle verbose logging. In the browser `process` may be undefined, so guard
// the access to avoid a ReferenceError at runtime.
const DEBUG_CANVAS =
  typeof process !== 'undefined'
    ? process.env.NODE_ENV === 'development'
    : false;

const VoronoiCanvas: React.FC<VoronoiCanvasProps> = ({
  seedPoints = [],
  vertices = [],
  edges = [],
  bbox,
  thickness,
  maxSteps,
  epsilon,
  showStruts = DEBUG_HEX_TEST || false,
  showSolid = true,
  showInfill = false,
  strutRadius = 0.5,
  strutColor = 'white',
  infillPoints = [],
  infillEdges = [],
  cells = [],
  edgeLengthThreshold = 1.5,
  edgeZVariationTolerance = EDGE_Z_VARIATION_TOLERANCE,
}) => {
  // Strut geometry can come from either the primary vertex/edge arrays or, when
  // those are absent, from the infill geometry.  Previously we logged a warning
  // as soon as the component mounted with no `edges` prop which produced noisy
  // console output while data was still loading.  Instead we defer any warnings
  // until after sanitising the inputs and establishing whether *any* usable
  // edge list was supplied.
  // In debug mode, hide the ray-march solid box
  if (DEBUG_HEX_TEST) showSolid = false;
  DEBUG_CANVAS && console.log('VoronoiCanvas debug props:', { seedPoints, vertices, edges, bbox });
  const safeSeedPoints = Array.isArray(seedPoints)
    ? seedPoints
    : (typeof seedPoints === 'string' ? JSON.parse(seedPoints) : []);
  const safeVertices = Array.isArray(vertices)
    ? vertices
    : (typeof vertices === 'string' ? JSON.parse(vertices) : []);
  const safeEdges = Array.isArray(edges)
    ? edges
    : (typeof edges === 'string' ? JSON.parse(edges) : []);
  const safeInfillPoints = Array.isArray(infillPoints)
    ? infillPoints
    : (typeof infillPoints === 'string' ? JSON.parse(infillPoints) : []);
  const safeInfillEdges = Array.isArray(infillEdges)
    ? infillEdges
    : (typeof infillEdges === 'string' ? JSON.parse(infillEdges) : []);
  const safeCells = Array.isArray(cells)
    ? cells
    : (typeof cells === 'string' ? JSON.parse(cells) : []);
  // Ensure only well-formed point and edge arrays
  const validSeedPoints = Array.isArray(safeSeedPoints)
    ? safeSeedPoints.filter(p => Array.isArray(p) && p.length === 3 && p.every(coord => typeof coord === 'number'))
    : [];
  const validVertices = Array.isArray(safeVertices)
    ? safeVertices.filter(p => Array.isArray(p) && p.length === 3 && p.every(coord => typeof coord === 'number'))
    : [];
  const validEdges = Array.isArray(safeEdges)
    ? safeEdges.filter(e => Array.isArray(e) && e.length === 2 && e.every(idx => Number.isInteger(idx)))
    : [];
  const validInfillPoints = Array.isArray(safeInfillPoints)
    ? safeInfillPoints.filter(p => Array.isArray(p) && p.length === 3 && p.every(coord => typeof coord === 'number'))
    : [];
  const validInfillEdges = Array.isArray(safeInfillEdges)
    ? safeInfillEdges.filter(e => Array.isArray(e) && e.length === 2 && e.every(idx => Number.isInteger(idx)))
    : [];
  const validCells = Array.isArray(safeCells)
    ? safeCells.filter(c => Array.isArray(c?.verts) && Array.isArray(c?.faces))
    : [];

  // Prefer the explicit vertex/edge arrays for strut rendering; fall back to
  // infill geometry when the main arrays are missing.  This allows the canvas to
  // render struts even when only `seed_points`/`infill` data is available from
  // the backend.
  const pointsForStruts =
    validVertices.length > 0
      ? validVertices
      : validInfillPoints.length > 0
        ? validInfillPoints
        : validSeedPoints;
  const edgesForStruts = validEdges.length > 0 ? validEdges : validInfillEdges;
  const noEdges = edgesForStruts.length === 0;
  const usingFallbackSeeds =
    validVertices.length === 0 && validInfillPoints.length === 0;

  const voronoiSpec = useMemo(() => ({
    pattern: 'voronoi',
    seed_points: validSeedPoints,
    bbox_min: [bbox[0], bbox[1], bbox[2]],
    bbox_max: [bbox[3], bbox[4], bbox[5]],
    use_voronoi_edges: true,
  }), [validSeedPoints, bbox]);

  useEffect(() => {
    DEBUG_CANVAS && console.log('VoronoiCanvas infill spec:', voronoiSpec);
  }, [voronoiSpec]);
  DEBUG_CANVAS && console.log('VoronoiCanvas validInfillPoints count:', validInfillPoints.length);
  DEBUG_CANVAS && console.log('VoronoiCanvas validInfillEdges count:', validInfillEdges.length);
  DEBUG_CANVAS && console.log('VoronoiCanvas validCells count:', validCells.length);
  DEBUG_CANVAS && console.log('VoronoiCanvas sample validInfillPoints (first 5):', validInfillPoints.slice(0,5));
  DEBUG_CANVAS && console.log('VoronoiCanvas sample validInfillEdges (first 5):', validInfillEdges.slice(0,5));
  DEBUG_CANVAS && console.log('VoronoiCanvas safe data:', { safeSeedPoints, safeVertices, safeEdges });
  // Debug: inspect z-slices and sample points
  const zValues = validSeedPoints.map(p => p[2]);
  DEBUG_CANVAS && console.log('VoronoiCanvas validSeedPoints z-range:', Math.min(...zValues), Math.max(...zValues));
  DEBUG_CANVAS && console.log('VoronoiCanvas sample validSeedPoints (first 5):', validSeedPoints.slice(0,5));
  // Filter out spurious edges. We use a robust median/IQR-based threshold and
  // treat vertical edges separately so tall z-struts survive the filter.  When
  // no explicit vertex/edge arrays are supplied we fall back to the infill
  // geometry; only after this fallback do we warn about truly missing data.
  const filteredEdges = useMemo(() => {
    if (edgesForStruts.length === 0) {
      (edges.length > 0 || infillEdges.length > 0) &&
        console.warn('VoronoiCanvas: no valid edges provided');
      return [];
    }
    if (pointsForStruts.length === 0) {
      console.warn(
        'VoronoiCanvas: no vertices provided; skipping edge filtering and strut geometry.'
      );
      return [];
    }
    DEBUG_CANVAS &&
      console.log('VoronoiCanvas useMemo input:', {
        validVertices: pointsForStruts,
        validEdges: edgesForStruts,
        edgeLengthThreshold,
      });
    return computeFilteredEdges(
      pointsForStruts,
      edgesForStruts,
      edgeLengthThreshold,
      0
    );
  }, [
    pointsForStruts,
    edgesForStruts,
    edgeLengthThreshold,
    edges,
    infillEdges,
  ]);
  DEBUG_CANVAS &&
    console.log('VoronoiCanvas filteredEdges count:', filteredEdges.length);
  DEBUG_CANVAS && console.log('VoronoiCanvas sample filteredEdges (first 5):', filteredEdges.slice(0,5));

  // After filtering, ensure that the remaining edges collectively span
  // meaningful distance along the z-axis.  A globally flat edge set often
  // indicates upstream geometry that collapsed into a single plane.  Warn (or
  // optionally throw) only once per mount.
  const warnedFlatRef = useRef(false);
  const hasFlatEdges = useMemo(() => {
    if (
      filteredEdges.length === 0 ||
      pointsForStruts.length === 0 ||
      usingFallbackSeeds
    ) {
      return false;
    }
    let zMin = Infinity;
    let zMax = -Infinity;
    for (const [i, j] of filteredEdges) {
      const zi = pointsForStruts[i]?.[2];
      const zj = pointsForStruts[j]?.[2];
      if (zi === undefined || zj === undefined) continue;
      zMin = Math.min(zMin, zi, zj);
      zMax = Math.max(zMax, zi, zj);
    }
    const span = zMax - zMin;
    const flat = span < edgeZVariationTolerance;
    if (flat && !warnedFlatRef.current) {
      const msg = `VoronoiCanvas: edge z-range below tolerance (${edgeZVariationTolerance})`;
      if (
        typeof process !== 'undefined' &&
        process.env.VORONOI_ASSERT_Z === 'true'
      ) {
        throw new Error(msg);
      }
      console.warn(msg);
      warnedFlatRef.current = true;
    }
    return flat;
  }, [
    filteredEdges,
    pointsForStruts,
    usingFallbackSeeds,
    edgeZVariationTolerance,
  ]);
  DEBUG_CANVAS && console.log('VoronoiCanvas hasFlatEdges:', hasFlatEdges);
  // For debug: only render seed points inside the sphere when no edges
  const debugSeedPoints = useMemo(() => {
    if (filteredEdges.length > 0) return [];
    // Derive sphere center and radius from seedPoints bounds
    const xs = validSeedPoints.map(p => p[0]);
    const ys = validSeedPoints.map(p => p[1]);
    const zs = validSeedPoints.map(p => p[2]);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const minZ = Math.min(...zs), maxZ = Math.max(...zs);
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const cz = (minZ + maxZ) / 2;
    const radius = Math.max(maxX - minX, maxY - minY, maxZ - minZ) / 2;
    DEBUG_CANVAS && console.log('VoronoiCanvas debug sphere center and radius (from seed bounds):', { cx, cy, cz, radius });
    return validSeedPoints.filter(([x, y, z]) =>
      (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius * radius
    );
  }, [filteredEdges, validSeedPoints, bbox]);
  DEBUG_CANVAS && console.log('VoronoiCanvas debugSeedPoints count:', debugSeedPoints.length);
  // If no edges, choose points to render: use sphere-filtered debug points, or all seeds if none filtered
  const fallbackPoints = filteredEdges.length === 0
    ? (debugSeedPoints.length > 0 ? debugSeedPoints : validSeedPoints)
    : [];
  DEBUG_CANVAS && console.log('VoronoiCanvas fallbackPoints count:', fallbackPoints.length, fallbackPoints);
  // Debug: inspect fallback points further
  const fallbackZ = fallbackPoints.map(p => p[2]);
  DEBUG_CANVAS && console.log('VoronoiCanvas fallbackPoints z-range:', Math.min(...fallbackZ), Math.max(...fallbackZ));
  // Log x/y range for fallback points
  const xValues = fallbackPoints.map(p => p[0]);
  const yValues = fallbackPoints.map(p => p[1]);
  DEBUG_CANVAS && console.log('VoronoiCanvas fallbackPoints x-range:', Math.min(...xValues), Math.max(...xValues));
  DEBUG_CANVAS && console.log('VoronoiCanvas fallbackPoints y-range:', Math.min(...yValues), Math.max(...yValues));
  DEBUG_CANVAS && console.log('Rendering JSX: fallbackPoints count', fallbackPoints.length, 'flat length', fallbackPoints.flat().length);
  DEBUG_CANVAS && console.log('VoronoiCanvas sample fallbackPoints (first 5):', fallbackPoints.slice(0,5));

  // Ref and effect for logging and computing bounding sphere of the points buffer
  const bufferRef = useRef<THREE.BufferGeometry>(null);
  useEffect(() => {
    if (bufferRef.current) {
      const geom = bufferRef.current;
      const positionAttr = geom.getAttribute('position');
      if (positionAttr) {
        DEBUG_CANVAS && console.log(
          'bufferGeometry useEffect: position attribute count',
          positionAttr.count,
          'boundingSphere before compute',
          geom.boundingSphere
        );
        geom.computeBoundingSphere();
        DEBUG_CANVAS && console.log(
          'bufferGeometry useEffect: boundingSphere after compute',
          geom.boundingSphere
        );
      }
    }
  }, [fallbackPoints]);

  // Merge all cell geometries into a single mesh
  const mergedCellGeometry = useMemo(() => {
    if (validCells.length === 0) return null;
    // filter cells to only those inside the seed-points bounding sphere
    const xs = validSeedPoints.map(p => p[0]);
    const ys = validSeedPoints.map(p => p[1]);
    const zs = validSeedPoints.map(p => p[2]);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const minZ = Math.min(...zs), maxZ = Math.max(...zs);
    const sphereCx = (minX + maxX) / 2;
    const sphereCy = (minY + maxY) / 2;
    const sphereCz = (minZ + maxZ) / 2;
    const sphereR  = Math.max(maxX - minX, maxY - minY, maxZ - minZ) / 2;

    const filteredCells = validCells.filter(cell => {


      if (!Array.isArray(cell.verts) || cell.verts.length === 0) return false;


      // compute centroid of this cell
      const centroid = cell.verts.reduce(
        (acc, [x, y, z]) => [acc[0] + x, acc[1] + y, acc[2] + z],
        [0, 0, 0]
      ).map(sum => sum / cell.verts.length);
      const [cx, cy, cz] = centroid;
      return (
        (cx - sphereCx) ** 2 +
        (cy - sphereCy) ** 2 +
        (cz - sphereCz) ** 2
      ) <= sphereR * sphereR;
    });
    const positions: number[] = [];
    const indices: number[] = [];
    let vertexOffset = 0;
    for (const cell of filteredCells) {
      // collect all vertex positions
      for (const [x, y, z] of cell.verts) {
        positions.push(x, y, z);
      }
      // collect face indices, offset by current vertex count
      for (const face of cell.faces) {
        indices.push(
          face[0] + vertexOffset,
          face[1] + vertexOffset,
          face[2] + vertexOffset
        );
      }
      vertexOffset += cell.verts.length;
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(positions), 3)
    );
    geom.setIndex(indices);
    geom.computeVertexNormals();
    return geom;
  }, [validCells, validSeedPoints]);

  return (
    <div
      style={{
        width: '100%',
        height: '400px',
        maxHeight: '400px',
        minHeight: 0,
        overflow: 'hidden',
        position: 'relative',
        flexShrink: 0
      }}
      data-testid="voronoi-canvas-root"
      data-has-flat-edges={hasFlatEdges}
    >
      {noEdges ? (
        <div
          className="warning-banner"
          role="alert"
          data-testid="no-edges-warning"
        >
          No edges were returned; unable to render Voronoi mesh.
        </div>
      ) : (
        <Canvas
          style={{ width: '100%', height: '100%', display: 'block' }}
          resize={{ scroll: false }}
          gl={{ version: 2 }}
          camera={{ position: [15, 15, 15], fov: 60 }}
        >
          {showStruts ? (
            showSolid && (
              // Simple box for Strut view
              <mesh
              position={[
                (bbox[0] + bbox[3]) / 2,
                (bbox[1] + bbox[4]) / 2,
                (bbox[2] + bbox[5]) / 2
              ]}
            >
              <boxGeometry args={[
                bbox[3] - bbox[0],
                bbox[4] - bbox[1],
                bbox[5] - bbox[2]
              ]} />
              <meshBasicMaterial
                color="white"
                transparent
                opacity={0.2}
              />
            </mesh>
          )
        ) : (
          (showSolid || showInfill) && (
            // Ray-marched solid for Ray-March view
            <VoronoiMesh
              seedPoints={validSeedPoints}
              bbox={bbox}
              thickness={thickness}
              maxSteps={maxSteps}
              epsilon={epsilon}
              showSolid={showSolid}
              showInfill={showInfill}
              infillPoints={validInfillPoints}
              infillEdges={validInfillEdges}
            />
          )
        )}
        {showStruts && (
          <VoronoiStruts
            vertices={pointsForStruts}
            edges={filteredEdges}
            strutRadius={strutRadius}
            color={strutColor}
          />
        )}
        {showSolid && showInfill && !showStruts && validInfillEdges.length > 0 && (
          <lineSegments>
            <bufferGeometry attach="geometry">
              <bufferAttribute
                attach="attributes-position"
                array={new Float32Array(
                  validInfillEdges.flatMap(([i, j]) => [
                    ...validInfillPoints[i],
                    ...validInfillPoints[j],
                  ])
                )}
                count={validInfillEdges.length * 2}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial color="cyan" />
          </lineSegments>
        )}
        <ambientLight intensity={0.5} />
        <directionalLight
          intensity={0.8}
          position={[10, 10, 10]}
          castShadow={false}
        />
        <pointLight position={[5, 5, 5]} />
        {/* Grid and axes helpers */}
        <gridHelper args={[200, 20]} />
        <axesHelper args={[100]} />
        {mergedCellGeometry && (
          <mesh geometry={mergedCellGeometry}>
            <meshStandardMaterial color="lightblue" wireframe />
          </mesh>
        )}
        {/*
        {fallbackPoints.length > 0 && (
          fallbackPoints.slice(0, 3).map((p, i) => (
            <mesh key={`debug-sphere-${i}`} position={[p[0], p[1], p[2]]}>
              <sphereGeometry args={[0.5, 16, 16]} />
              <meshBasicMaterial color="yellow" />
            </mesh>
          ))
        )}
        */}
        {/* 
        {fallbackPoints.length > 0 && (
          <points frustumCulled={false}>
            <bufferGeometry
              attach="geometry"
              ref={bufferRef}
            >
              <bufferAttribute
                attach="attributes-position"
                array={new Float32Array(fallbackPoints.flat())}
                count={fallbackPoints.length}
                itemSize={3}
              />
            </bufferGeometry>
            <pointsMaterial size={0.5} color="red" />
          </points>
        )}
        */}
        <OrbitControls />

        </Canvas>
      )}
    </div>
  );
};

export default VoronoiCanvas;
