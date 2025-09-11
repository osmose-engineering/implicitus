
import http from 'http';

const server = http.createServer((req, res) => {
  if (req.method === 'POST' && req.url === '/slice') {
    let data = '';
    req.on('data', chunk => { data += chunk; });
    req.on('end', () => {
      // Return a simple circular contour for tests
      const pts = Array.from({ length: 40 }, (_, i) => {
        const t = (2 * Math.PI * i) / 40;
        return [Math.cos(t), Math.sin(t)];
      });
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify({ contours: [pts] }));
    });
    return;
  }
  res.statusCode = 404;
  res.end();
});

server.listen(4000);

process.on('SIGTERM', () => server.close());
process.on('SIGINT', () => server.close());

