import React from 'react';
import { Canvas } from '@react-three/fiber';
import VoronoiMesh from './VoronoiMesh';
import { VoronoiStruts } from './VoronoiStruts';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

interface VoronoiCanvasProps {
  seedPoints: [number, number, number][];
  bbox: [number, number, number, number, number, number];
  thickness?: number;
  maxSteps?: number;
  epsilon?: number;
  showRaymarch?: boolean;
  showStruts?: boolean;
  strutRadius?: number;
  strutColor?: string | number;
}

const VoronoiCanvas: React.FC<VoronoiCanvasProps> = ({
  seedPoints,
  bbox,
  thickness,
  maxSteps,
  epsilon,
  showRaymarch = true,
  showStruts = false,
  strutRadius = 0.5,
  strutColor = 'white',
}) => {
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
        {showRaymarch && <VoronoiMesh seedPoints={seedPoints} bbox={bbox} thickness={thickness} maxSteps={maxSteps} epsilon={epsilon} />}
        {showStruts  && <VoronoiStruts seedPoints={seedPoints} strutRadius={strutRadius} color={strutColor} />}
        <ambientLight />
        <pointLight position={[5, 5, 5]} />
        {/* Grid and axes helpers */}
        <primitive object={new THREE.GridHelper(200, 20)} />
        <primitive object={new THREE.AxesHelper(100)} />
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
