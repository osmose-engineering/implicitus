// src/Preview.tsx
import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';

const MM_TO_UNIT = 0.1;

interface PreviewProps {
  spec: any | any[];
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

const Preview: React.FC<PreviewProps> = ({ spec }) => {
  //console.log('[Preview] component rendered, spec prop:', spec);
  // Accept either a single node or an array of nodes
  const nodes = Array.isArray(spec) ? spec : [spec];
  //console.log('[Preview] nodes:', nodes);
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

  const primitives: { type: string; params: Record<string, number>; infill?: { pattern: string; density: number } }[] = useMemo(() => {
    //console.log('[Preview] computing primitives from spec:', spec);
    return nodes.map((node: any) => {
      let raw = {};
      if (node.primitive) {
        raw = node.primitive;
      } else if (node.root?.primitive) {
        raw = node.root.primitive;
      } else if (Array.isArray(node.children) && node.children.length > 0) {
        const firstChild = node.children[0];
        raw = firstChild.primitive || firstChild.root?.primitive || {};
      }
      // normalize infill parameters
      const infillRaw = node.infill || {};
      const pattern = infillRaw.pattern ?? infillRaw.infill_pattern ?? infillRaw.infill_shape;
      const density = infillRaw.density ?? infillRaw.infill_thickness_mm ?? infillRaw.infill_mm;
      const infill = pattern && density != null ? { pattern, density } : undefined;
      //console.log('[Preview] raw primitive data:', raw);
      if (raw.type && raw.params) {
        return { ...raw, infill };
      }
      const key = Object.keys(raw)[0];
      const details = (raw as any)[key];
      console.log('[Preview] resolved primitive:', { key, details });
      switch (key) {
        case 'box':
        case 'cube': {
          // support both numeric size and object size { x, y, z }
          if (details.size && typeof details.size === 'object') {
            const { x, y, z } = details.size as { x: number; y: number; z: number };
            return { type: 'box', params: { size_mm: undefined, size: { x, y, z } }, infill };
          } else if (typeof details.size === 'number') {
            return { type: 'box', params: { size_mm: details.size }, infill };
          }
          return { type: 'box', params: {}, infill };
        }
        case 'sphere':
          return { type: 'sphere', params: { size_mm: details.radius * 2 }, infill };
        case 'cylinder':
          return { type: 'cylinder', params: { radius_mm: details.radius, height_mm: details.height }, infill };
        default:
          return { type: '', params: {}, infill };
      }
    });
  }, [spec]);

  //console.log('[Preview] computed primitives:', primitives);
  // if no primitives have been resolved yet, prompt for confirmation
  if (primitives.length === 0) {
    return (
      <div style={{ padding: '1rem', color: '#666' }}>
        Length = 0. Please confirm your spec above before rendering the 3D preview.
      </div>
    );
  }
  //console.log('[Preview] about to render Canvas, primitives:', primitives);

  return (
    <div style={{ width: '100%', height: '400px' }}>
      <Canvas 
        camera={{ position: [3, 3, 3], fov: 50 }}
        gl={{ preserveDrawingBuffer: true, antialias: true }}
      >
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 5, 5]} intensity={1} />
        <axesHelper args={[5]} />
        <gridHelper args={[10, 10]} />
        {primitives.map((primitive, idx) => (
          <group key={idx}>
            {/* outer solid shell */}
            <Geometry primitive={primitive} />
            {/* inner wireframe infill approximation */}
            {primitive.infill && (() => {
              // compute scaled-down params for infill shell
              const inset = primitive.infill.density;
              const infillParams: Record<string, number> = {};
              // copy and shrink each dimensional param
              if (primitive.params.size_mm !== undefined) {
                infillParams.size_mm = primitive.params.size_mm - inset * 2;
              }
              if (primitive.params.radius_mm !== undefined) {
                infillParams.radius_mm = primitive.params.radius_mm - inset;
              }
              if (primitive.params.height_mm !== undefined) {
                infillParams.height_mm = primitive.params.height_mm - inset * 2;
              }
              return (
                <Geometry
                  primitive={{ type: primitive.type, params: infillParams }}
                  infill={primitive.infill}
                />
              );
            })()}
          </group>
        ))}
        <OrbitControls />
      </Canvas>
    </div>
  );
};

export default Preview;