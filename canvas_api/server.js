const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');

const app = express();
const port = process.env.PORT || 3000;
const DESIGN_API = process.env.DESIGN_API || 'http://127.0.0.1:5000';

app.use(bodyParser.json());

console.warn('Canvas API is deprecated. Please use the Design API directly.');

app.post('/design', async (req, res) => {
  try {
    const { data } = await axios.post(`${DESIGN_API}/design`, req.body);
    res.json(data);
  } catch (err) {
    console.error('Design API error:', err.message);
    res.status(500).json({ error: 'Design service failure', details: err.message });
  }
});

// All other routes are deprecated
app.all('*', (_req, res) => {
  res.status(410).json({ error: 'Canvas API is deprecated. Use the Design API.' });
});

app.listen(port, () => {
  console.log(`Deprecated Canvas API listening on port ${port}`);
});
