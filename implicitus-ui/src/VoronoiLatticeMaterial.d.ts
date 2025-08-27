import * as THREE from 'three';
export declare const VoronoiLatticeMaterial: import("@react-three/fiber").ConstructorRepresentation<THREE.ShaderMaterial & {
    uSeedsTex: {
        value: THREE.DataTexture;
    };
    uTexSize: {
        value: number;
    };
    uNumSeeds: {
        value: number;
    };
    uBoxMin: {
        value: THREE.Vector3;
    };
    uBoxMax: {
        value: THREE.Vector3;
    };
    uThickness: {
        value: number;
    };
    uMaxSteps: {
        value: number;
    };
    uEpsilon: {
        value: number;
    };
}> & {
    key: string;
};

declare global {
  namespace JSX {
    interface IntrinsicElements {
      voronoiLatticeMaterial: any;
    }
  }
}
