// src/Preview.tsx
import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';

const MM_TO_UNIT = 0.1;

interface PreviewProps {
  spec: {
    shape: string;
    size_mm?: number;
    radius_mm?: number;
    height_mm?: number;
    // extend as you add more primitives…
  };
}

const Geometry: React.FC<{ spec: PreviewProps['spec'] }> = ({ spec }) => {
    console.debug('[Preview] incoming spec:', spec);
  const meshProps = useMemo(() => {
    switch (spec.shape) {
      case 'sphere':
        return { args: [ ((spec.size_mm ?? spec.radius_mm ?? 1) / 2) * MM_TO_UNIT ] as [number], geometry: 'sphere' };
      case 'cube':
      case 'box':
        const s = (spec.size_mm ?? 1) * MM_TO_UNIT;
        return { args: [s, s, s] as [number, number, number], geometry: 'box' };
      case 'cylinder':
        // Use explicit radius_mm and height_mm, no size_mm
        const radius = (spec.radius_mm ?? 1) * MM_TO_UNIT;
        const height = (spec.height_mm ?? 1) * MM_TO_UNIT;
        return {
          args: [radius, radius, height, 32] as [number, number, number, number],
          geometry: 'cylinder'
        };
      default:
        return null;
    }
  }, [spec]);

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

const Preview: React.FC<PreviewProps> = ({ spec }) => (
  <div style={{ width: '100%', height: '400px' }}>
    <Canvas 
      camera={{ position: [3, 3, 3], fov: 50 }}
      gl={{ preserveDrawingBuffer: true, antialias: true }}
    >
      <ambientLight intensity={0.5} />
      <directionalLight position={[5, 5, 5]} intensity={1} />
      <axesHelper args={[5]} />
      <gridHelper args={[10, 10]} />
      <Geometry spec={spec} />
      <OrbitControls />
    </Canvas>
  </div>
);

export default Preview;