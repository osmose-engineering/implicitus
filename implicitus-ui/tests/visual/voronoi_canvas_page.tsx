import React from 'react';
import ReactDOM from 'react-dom/client';
import VoronoiCanvas from '../../src/components/VoronoiCanvas';

const root = ReactDOM.createRoot(document.getElementById('root')!);

root.render(
  <div style={{ width: 300, height: 300 }}>
    <VoronoiCanvas
      seedPoints={[
        [0, 0, 0],
        [1, 0, 0.5],
        [0, 1, 1.0],
        [1, 1, 1.5],
        [0.5, 0.5, 2.0],
      ]}
      vertices={[
        [0, 0, 0],
        [1, 0, 0.5],
        [0, 1, 1.0],
        [1, 1, 1.5],
        [0.5, 0.5, 2.0],
      ]}
      edges={[
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 0],
      ]}
      bbox={[0, 0, 0, 1, 1, 2]}
    />
  </div>,
);
