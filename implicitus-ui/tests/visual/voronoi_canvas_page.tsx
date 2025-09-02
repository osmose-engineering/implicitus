import React from 'react';
import ReactDOM from 'react-dom/client';
import VoronoiCanvas from '../../src/components/VoronoiCanvas';

const root = ReactDOM.createRoot(document.getElementById('root')!);

root.render(
  <div style={{ width: 300, height: 300 }}>
    <VoronoiCanvas
      seedPoints={[
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]}
      vertices={[
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]}
      edges={[
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
      ]}
      bbox={[0, 0, 0, 1, 1, 1]}
    />
  </div>
);
