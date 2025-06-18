// src/Preview.tsx
import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';

const MM_TO_UNIT = 0.1;

interface PreviewProps {
  spec: any;
}

const Geometry: React.FC<{ primitive: { type: string; params: Record<string, number> } }> = ({ primitive }) => {
  const { type, params } = primitive;
  const meshProps = useMemo(() => {
    console.debug('[Geometry] type:', type, 'params:', params);
    switch (type) {
      case 'sphere': {
        // Prefer size_mm, else use radius_mm * 2, fallback to 1
        const size = params.size_mm ?? (params.radius_mm !== undefined ? params.radius_mm * 2 : 1);
        const props = { args: [ (size / 2) * MM_TO_UNIT ] as [number], geometry: 'sphere' };
        console.debug('[Geometry] meshProps for', type, props);
        return props;
      }
      case 'cube':
      case 'box': {
        const s = (params.size_mm ?? 1) * MM_TO_UNIT;
        const props = { args: [s, s, s] as [number, number, number], geometry: 'box' };
        console.debug('[Geometry] meshProps for', type, props);
        return props;
      }
      case 'cylinder': {
        const radius = (params.radius_mm ?? 1) * MM_TO_UNIT;
        const height = (params.height_mm ?? 1) * MM_TO_UNIT;
        const props = {
          args: [radius, radius, height, 32] as [number, number, number, number],
          geometry: 'cylinder'
        };
        console.debug('[Geometry] meshProps for', type, props);
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
      <meshStandardMaterial color="orange" wireframe />
    </mesh>
  );
};

const Preview: React.FC<PreviewProps> = ({ spec }) => {
  console.debug('[Preview] spec prop:', spec);
  // support nested proto spec or flattened spec
  const raw = (spec as any).primitive || (spec as any).root?.primitive || {};
  console.debug('[Preview] raw primitive data:', raw);
  const primitive: { type: string; params: Record<string, number> } = useMemo(() => {
    if (raw.type && raw.params) {
      return raw;
    }
    // oneof keys
    const key = Object.keys(raw)[0];
    const details = (raw as any)[key];
    console.debug('[Preview] resolved primitive:', { key, details });
    switch (key) {
      case 'box':
        return {
          type: 'box',
          params: { size_mm: details.size.x }
        };
      case 'sphere':
        return {
          type: 'sphere',
          params: { size_mm: details.radius * 2 }
        };
      case 'cylinder':
        return {
          type: 'cylinder',
          params: { radius_mm: details.radius, height_mm: details.height }
        };
      default:
        return { type: '', params: {} };
    }
  }, [raw]);

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
        <Geometry primitive={primitive} />
        <OrbitControls />
      </Canvas>
    </div>
  );
};

export default Preview;