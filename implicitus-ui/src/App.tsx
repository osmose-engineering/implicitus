import React, { useState } from 'react';
import './App.css'
import Preview from './Preview';

function App() {
  const [prompt, setPrompt] = useState('');
  const [spec, setSpec] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError(null);
    setSpec(null);
    setLoading(true);
    console.log('[UI] Sending prompt →', prompt);
    try {
      const response = await fetch('http://localhost:8000/design', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });
      // read raw text and debug
      const rawText = await response.text();
      console.log('[UI] raw response text:', rawText);
      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText} - ${rawText}`);
      }
      let data: any;
      try {
        data = JSON.parse(rawText);
      } catch (parseError) {
        console.error('[UI] JSON parse error:', parseError);
        throw new Error(`Failed to parse JSON spec: ${parseError}`);
      }
      console.log('[UI] Parsed spec object:', data);
      setSpec(data);
    } catch (err: any) {
      console.error('[UI] fetch error:', err);
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Implicitus</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          placeholder="Describe your design prompt..."
          rows={6}
          cols={60}
          disabled={loading}
        />
        <br />
        <button type="submit" disabled={loading || !prompt.trim()}>
          {loading ? 'Designing...' : 'Generate Design'}
        </button>
      </form>
      {spec && (
        <div style={{ textAlign: 'left', margin: '1em', padding: '1em', border: '1px dashed #888' }}>
          <strong>Full Object Spec:</strong>
          <pre>{JSON.stringify(spec, null, 2)}</pre>
        </div>
      )}
      {error && <div className="error">{error}</div>}
      {spec && (
        <div>
          
          <Preview spec={spec} />
        </div>
      )}
    </div>
  );
}

export default App
