const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');

const app = express();
const port = process.env.PORT || 3000;

// Middleware to parse JSON bodies
app.use(bodyParser.json());

// In-memory model storage
const models = new Map();

/**
 * Store a new model.
 * Expects a JSON body representing an implicitus.Model with at least an `id` field.
 * Responds with { id: string }.
 */
app.post('/models', (req, res) => {
  const model = req.body;
  if (!model || !model.id) {
    return res.status(400).json({ error: 'Missing model.id' });
  }
  models.set(model.id, model);
  res.json({ id: model.id });
});

/**
 * Retrieve a stored model by ID.
 * Responds with the full model JSON or 404 if not found.
 */
app.get('/models/:id', (req, res) => {
  const model = models.get(req.params.id);
  if (!model) {
    return res.status(404).json({ error: 'Model not found' });
  }
  res.json(model);
});


/**
 * Slice a stored model at a given layer.
 * Query parameters:
 *   - layer (required): z-height for the slice
 *   - x_min, x_max, y_min, y_max, nx, ny (optional): slice bounding box and resolution
 * Returns JSON: { contours: [ [ [x,y], ... ], ... ] }
 */
app.get('/models/:id/slices', (req, res) => {
  const model = models.get(req.params.id);
  if (!model) {
    return res.status(404).json({ error: 'Model not found' });
  }

  // Parse slice parameters
  const z = parseFloat(req.query.layer);
  if (Number.isNaN(z)) {
    return res.status(400).json({ error: 'Invalid or missing "layer" parameter' });
  }
  // Optional parameters with defaults
  const x_min = parseFloat(req.query.x_min) || -1.0;
  const x_max = parseFloat(req.query.x_max) || 1.0;
  const y_min = parseFloat(req.query.y_min) || -1.0;
  const y_max = parseFloat(req.query.y_max) || 1.0;
  const nx = parseInt(req.query.nx) || 50;
  const ny = parseInt(req.query.ny) || 50;

  // Proxy slice request to Rust slicer service
  axios.post('http://127.0.0.1:4000/slice', {
    model,
    layer: z,
    x_min,
    x_max,
    y_min,
    y_max,
    nx,
    ny
  })
  .then(response => {
    // Forward contours from slicer service
    res.json(response.data);
  })
  .catch(err => {
    console.error('Slicer service error:', err.message);
    res.status(500).json({ error: 'Slicing service failure', details: err.message });
  });
});

app.listen(port, () => {
  console.log(`Implicitus Canvas API listening on port ${port}`);
});