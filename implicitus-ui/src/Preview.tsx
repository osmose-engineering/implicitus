// src/Preview.tsx
import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import VoronoiLatticePreview from './VoronoiLatticePreview';
// Compute axis-aligned bounds for a primitive
function computeBounds(primitive: { type: string; params: any }): { min: [number, number, number]; max: [number, number, number] } {
  switch (primitive.type) {
    case 'sphere': {
      const r = (primitive.params.size_mm ?? 0) / 2 * MM_TO_UNIT;
      return { min: [-r, -r, -r], max: [r, r, r] };
    }
    case 'box':
    case 'cube': {
      const size = primitive.params.size || { x: primitive.params.size_mm, y: primitive.params.size_mm, z: primitive.params.size_mm };
      return {
        min: [-(size.x*MM_TO_UNIT)/2, -(size.y*MM_TO_UNIT)/2, -(size.z*MM_TO_UNIT)/2],
        max: [(size.x*MM_TO_UNIT)/2, (size.y*MM_TO_UNIT)/2, (size.z*MM_TO_UNIT)/2]
      };
    }
    case 'cylinder': {
      const r = (primitive.params.radius_mm ?? 0) * MM_TO_UNIT;
      const h = (primitive.params.height_mm ?? 0) * MM_TO_UNIT;
      return { min: [-r, -r, -h/2], max: [r, r, h/2] };
    }
    default:
      return { min: [-1, -1, -1], max: [1, 1, 1] };
  }
}

const MM_TO_UNIT = 0.1;

interface PreviewProps {
  spec: any | any[];
  visibility: { [key: string]: boolean };
}

const Geometry: React.FC<{ primitive: { type: string; params: Record<string, number> }; infill?: { pattern: string; density: number } }> = ({ primitive, infill }) => {
  const { type, params } = primitive;
  const meshProps = useMemo(() => {
    console.log('[Geometry] type:', type, 'params:', params);
    switch (type) {
      case 'sphere': {
        // Prefer size_mm, else use radius_mm * 2, fallback to 1
        const size = params.size_mm ?? (params.radius_mm !== undefined ? params.radius_mm * 2 : 1);
        const props = { args: [ (size / 2) * MM_TO_UNIT ] as [number], geometry: 'sphere' };
        console.log('[Geometry] meshProps for', type, props);
        return props;
      }
      case 'cube':
      case 'box': {
        let args: [number, number, number];
        if (params.size && typeof params.size === 'object') {
          const { x, y, z } = params.size as { x: number; y: number; z: number };
          args = [x * MM_TO_UNIT, y * MM_TO_UNIT, z * MM_TO_UNIT];
        } else {
          const s = (params.size_mm ?? 1) * MM_TO_UNIT;
          args = [s, s, s];
        }
        const props = { args, geometry: 'box' as const };
        console.log('[Geometry] meshProps for', type, props);
        return props;
      }
      case 'cylinder': {
        const radius = (params.radius_mm ?? 1) * MM_TO_UNIT;
        const height = (params.height_mm ?? 1) * MM_TO_UNIT;
        const props = {
          args: [radius, radius, height, 32] as [number, number, number, number],
          geometry: 'cylinder'
        };
        console.log('[Geometry] meshProps for', type, props);
        return props;
      }
      default:
        return null;
    }
  }, [type, params]);

  if (!meshProps) return null;

  const { geometry, args } = meshProps;
  return (
    <mesh>
      {geometry === 'sphere' && <sphereGeometry args={args} />}
      {geometry === 'box'    && <boxGeometry    args={args} />}
      {geometry === 'cylinder' && <cylinderGeometry args={args} />}
      <meshStandardMaterial
        color="orange"
        wireframe={!!infill}
      />
    </mesh>
  );
};

const Preview: React.FC<PreviewProps> = ({ spec, visibility }) => {
  //console.log('[Preview] component rendered, spec prop:', spec);
  // Accept either a single node or an array of nodes
  const nodes = Array.isArray(spec) ? spec : [spec];

  const primitives = useMemo(() => {
    return nodes.map((node: any) => {
      const raw = node.primitive || node.root?.primitive ||
                  (Array.isArray(node.children) && node.children[0].primitive) || {};
      const key = Object.keys(raw)[0] || '';
      const details = (raw as any)[key] || {};
      let type = key;
      let params: Record<string, any> = {};
      switch (type) {
        case 'box':
        case 'cube':
          if (details.size != null) {
            if (typeof details.size === 'object') {
              params = { size: details.size };
            } else {
              params = { size_mm: details.size };
            }
          } else {
            params = { size_mm: details.size_mm ?? 1 };
          }
          break;
        case 'sphere':
          params = { size_mm: details.radius * 2 };
          break;
        case 'cylinder':
          params = { radius_mm: details.radius, height_mm: details.height };
          break;
        default:
          params = {};
      }
      const infill = node.modifiers?.infill;
      return { type, params, infill };
    });
  }, [spec]);

  const VIEW_SIZE = 10;
  const normalization = useMemo(() => {
    const boundsUnits = primitives.map(p => computeBounds({ type: p.type, params: p.params }));
    const mins = [0, 1, 2].map(i => Math.min(...boundsUnits.map(b => b.min[i])));
    const maxs = [0, 1, 2].map(i => Math.max(...boundsUnits.map(b => b.max[i])));
    const modelSize = Math.max(maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2]) || 1;
    const center: [number, number, number] = [
      (mins[0] + maxs[0]) / 2,
      (mins[1] + maxs[1]) / 2,
      (mins[2] + maxs[2]) / 2
    ];
    const scaleFactor = VIEW_SIZE / modelSize;
    return { boundsUnits, mins, maxs, center, modelSize, scaleFactor };
  }, [primitives]);

  // add a percentage margin around the model for helpers
  const MARGIN_RATIO = 0.5; // 50% margin
  const helperSize = normalization.modelSize * (1 + 2 * MARGIN_RATIO);

  // if any node does not yet have a confirmed primitive, prompt for confirmation
  const allConfirmed = nodes.every(node => {
    if (node.primitive || node.root?.primitive) return true;
    if (
      Array.isArray(node.children) && 
      node.children.length > 0 && 
      (node.children[0].primitive || node.children[0].root?.primitive)
    ) return true;
    return false;
  });
  if (!allConfirmed) {
    return (
      <div style={{ padding: '1rem', color: '#666' }}>
        Please confirm your spec above before rendering the 3D preview.
      </div>
    );
  }

  // if no primitives have been resolved yet, prompt for confirmation
  if (primitives.length === 0) {
    return (
      <div style={{ padding: '1rem', color: '#666' }}>
        Length = 0. Please confirm your spec above before rendering the 3D preview.
      </div>
    );
  }

  //console.log('[Preview] about to render Canvas, primitives:', primitives);

  // position camera dynamically so the model fits nicely
  const cameraDistance = VIEW_SIZE * 1.5;
  const cameraPosition: [number, number, number] = [cameraDistance, cameraDistance, cameraDistance];

  return (
    <div style={{ width: '100%', height: '400px' }}>
      <Canvas 
        camera={{ position: cameraPosition, fov: 50 }}
        gl={{ preserveDrawingBuffer: true, antialias: true }}
      >
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 5, 5]} intensity={1} />
        <group
          scale={[normalization.scaleFactor, normalization.scaleFactor, normalization.scaleFactor]}
          position={[-normalization.center[0], -normalization.center[1], -normalization.center[2]]}
        >
          <axesHelper args={[helperSize]} />
          <gridHelper args={[helperSize, 10]} />
          {primitives.map((primitive, idx) => {
            const cb = computeBounds({ type: primitive.type, params: primitive.params });
            return (
              <group key={idx}>
                {/* draw base solid */}
                {visibility.primitive && <Geometry primitive={primitive} />}
                {/* draw Voronoi lattice overlay if infill modifier present */}
                {visibility.infill && primitive.infill?.pattern === 'voronoi' && (
                  <VoronoiLatticePreview
                    spec={primitive.infill as any}
                    bounds={[cb.min, cb.max]}
                  />
                )}
              </group>
            );
          })}
        </group>
        <OrbitControls target={[0, 0, 0]} />
      </Canvas>
    </div>
  );
};

export default Preview;