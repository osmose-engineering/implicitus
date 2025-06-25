import React, { useState } from 'react';
import './App.css'
import Preview from './Preview';

function App() {
  const [prompt, setPrompt] = useState('');
  const [spec, setSpec] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState<string>('');
  const [specText, setSpecText] = useState<string>('');
  const [modelProto, setModelProto] = useState<any>(null);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError(null);
    setSpec(null);
    setSummary('');
    setSpecText('');
    setModelProto(null);
    setLoading(true);
    console.log('[UI] Sending prompt →', prompt);
    let bodyData: any = {};
    if (specText.trim() !== '') {
      try {
        bodyData.spec = JSON.parse(specText);
      } catch (parseError: any) {
        setError(`Failed to parse edited spec JSON: ${parseError.message || parseError}`);
        setLoading(false);
        return;
      }
    } else {
      bodyData.raw = prompt;
    }
    try {
      const response = await fetch('http://localhost:8000/design/review', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(bodyData),
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
      setSummary(data.summary);
      setSpec(data.spec);
      setSpecText(JSON.stringify(data.spec, null, 2));
    } catch (err: any) {
      console.error('[UI] fetch error:', err);
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleConfirm = async () => {
    setError(null);
    setLoading(true);
    try {
      const parsed = JSON.parse(specText);
      const response = await fetch('http://localhost:8000/design/submit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ spec: parsed }),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Server error: ${response.status} ${response.statusText} - ${text}`);
      }
      const data = await response.json();
      console.log('[UI] Confirm response data:', data);
      setModelProto(data);
    } catch (err: any) {
      setError(err.message || 'An error occurred during confirm');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App" style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <h1>Implicitus</h1>
      <div style={{ display: 'flex', flex: 1 }}>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div style={{ flex: 1, overflow: 'auto' }}>
            {/* Chat input window */}
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
            {error && <div className="error">{error}</div>}
          </div>
          <div style={{ flex: 1, overflow: 'auto' }}>
            {/* Editable JSON window */}
            {summary && (
              <div style={{ textAlign: 'left', margin: '1em', padding: '1em', border: '1px dashed #888' }}>
                <strong>Summary:</strong>
                <p>{summary}</p>
                <strong>Edit Spec:</strong>
                <textarea
                  value={specText}
                  onChange={e => setSpecText(e.target.value)}
                  rows={10}
                  cols={60}
                  disabled={modelProto !== null}
                />
                <br />
                <button onClick={handleConfirm} disabled={loading}>
                  Confirm &amp; Apply
                </button>
              </div>
            )}
          </div>
          <div style={{ flexBasis: 'auto', margin: '1em', padding: '1em', borderTop: '1px solid #ccc' }}>
            <h2>Final Model Proto</h2>
            {modelProto ? (
              <pre>{JSON.stringify(modelProto, null, 2)}</pre>
            ) : (
              <p><em>No final model proto yet. After you confirm, it will appear here.</em></p>
            )}
          </div>
        </div>
        <div style={{ flex: 1, overflow: 'auto' }}>
          {/* Preview panel */}
          {spec && (
            <div>
              <Preview spec={spec} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App
