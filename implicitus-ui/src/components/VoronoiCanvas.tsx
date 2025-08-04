import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import VoronoiMesh from './VoronoiMesh';
import { VoronoiStruts } from './VoronoiStruts';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
// Debug: override with a small hex‐lattice test instead of server data
const DEBUG_HEX_TEST = false;
function generateHexTest(bboxMin, bboxMax, spacing) {
  // Debug: draw outlines of a 3×3 honeycomb patch of cells
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

// Debug: generate a small 3D honeycomb block for testing
function generateHexTest3D(bboxMin, bboxMax, spacing) {
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

interface VoronoiCanvasProps {
  seedPoints: [number, number, number][];
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
}

const VoronoiCanvas: React.FC<VoronoiCanvasProps> = ({
  seedPoints,
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
}) => {
  // In debug mode, hide the ray-march solid box
  if (DEBUG_HEX_TEST) showSolid = false;
  // Debug override or use server-provided data
  const [useSeedPoints, useEdges] = DEBUG_HEX_TEST
    ? generateHexTest3D(
        [bbox[0], bbox[1], bbox[2]],
        [bbox[3], bbox[4], bbox[5]],
        // choose spacing relative to bbox extent
        (bbox[3] - bbox[0]) / 8
      )
    : [seedPoints, edges];
  // Filter out long edges to avoid hairball: only keep edges shorter than ~1.5× the average edge length
  const filteredEdges = useMemo(() => {
    if (useEdges.length === 0) return [];
    // compute lengths
    const lengths = useEdges.map(([i, j]) => {
      const [xi, yi, zi] = useSeedPoints[i];
      const [xj, yj, zj] = useSeedPoints[j];
      const dx = xi - xj, dy = yi - yj, dz = zi - zj;
      return Math.sqrt(dx*dx + dy*dy + dz*dz);
    });
    const avg = lengths.reduce((sum, d) => sum + d, 0) / lengths.length;
    const threshold = avg * 1.5;
    return useEdges.filter((_, idx) => lengths[idx] <= threshold);
  }, [useEdges, useSeedPoints]);
  return (
    <div style={{
      width: '100%',
      height: '400px',
      maxHeight: '400px',
      minHeight: 0,
      overflow: 'hidden',
      position: 'relative',
      flexShrink: 0
    }}>
      <Canvas
        style={{ width: '100%', height: '100%', display: 'block' }}
        resize={{ scroll: false }}
        gl={{ version: 2 }}
        camera={{ position: [3, 3, 3], fov: 60 }}
      >
        {showSolid && (
          showStruts ? (
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
          ) : (
            // Ray-marched solid for Ray-March view
            <VoronoiMesh
              seedPoints={useSeedPoints}
              bbox={bbox}
              thickness={thickness}
              maxSteps={maxSteps}
              epsilon={epsilon}
              showSolid={showSolid}
              showInfill={showInfill}
            />
          )
        )}
        {showStruts && (
          <VoronoiStruts
            seedPoints={useSeedPoints}
            edges={useEdges}
            strutRadius={strutRadius}
            color={strutColor}
          />
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
        {/* Debug: visualize seed points */}
        {/*seedPoints.map(([x, y, z], idx) => (
          <mesh key={idx} position={[x, y, z]}>
            <sphereGeometry args={[0.2, 16, 16]} />
            <meshBasicMaterial color="red" />
          </mesh>
        ))*/}
        <OrbitControls />
        
      </Canvas>
    </div>
  );
};

export default VoronoiCanvas;
