import React, { useState } from 'react';
import Editor from '@monaco-editor/react';
import * as monaco from 'monaco-editor';
import './App.css'
import Preview from './Preview';

const handleEditorDidMount = (editor: monaco.editor.IStandaloneCodeEditor, monacoInstance: typeof monaco) => {
  monacoInstance.languages.json.jsonDefaults.setDiagnosticsOptions({
    validate: true,
    schemas: [
      {
        uri: 'http://implicitus.io/schema/spec.json',
        fileMatch: ['*'],
        schema: {
          type: 'array',
          items: {
            type: 'object'
          }
        }
      }
    ],
  });
};

function App() {
  const [prompt, setPrompt] = useState('');
  const [spec, setSpec] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState<string>('');
  const [specText, setSpecText] = useState<string>('');
  const [modelProto, setModelProto] = useState<any>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<{speaker: 'user'|'assistant'; text: string}[]>([]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    // keep existing sessionId for follow-up updates
    setError(null);
    if (!sessionId) {
      setSpec(null);
      setSummary('');
      setSpecText('');
      setModelProto(null);
    }
    setLoading(true);
    setMessages(prev => [...prev, { speaker: 'user', text: prompt }]);
    console.log('[UI] Sending prompt →', prompt);
    let bodyData: any = {};
    // For updates (existing session), always send raw prompt and edited spec
    if (sessionId) {
      bodyData.raw = prompt;
      try {
        bodyData.spec = JSON.parse(specText);
      } catch (parseError: any) {
        setError(`Failed to parse edited spec JSON: ${parseError.message || parseError}`);
        setLoading(false);
        return;
      }
      bodyData.sid = sessionId;
    } else if (prompt.trim()) {
      // New design request: send only the raw prompt
      bodyData.raw = prompt;
    } else if (specText.trim()) {
      // New design request using manual spec editing
      try {
        bodyData.spec = JSON.parse(specText);
      } catch (parseError: any) {
        setError(`Failed to parse edited spec JSON: ${parseError.message || parseError}`);
        setLoading(false);
        return;
      }
    } else {
      setError("Please provide a prompt or a valid spec.");
      setLoading(false);
      return;
    }
    console.log('[UI] Prepared request bodyData →', bodyData);
    console.log('[UI] Current specText →', specText);
    console.log('[UI] Current prompt →', prompt);
    try {
      const endpoint = sessionId ? '/design/update' : '/design/review';
      console.log('[UI] using endpoint →', endpoint);
      const response = await fetch(`http://localhost:8000${endpoint}`, {
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
      console.log('[UI] Received parsed response data →', data);
      setSummary(data.summary);
      setSpec(data.spec);
      setSpecText(JSON.stringify(data.spec, null, 2));
      setMessages(prev => [...prev, { speaker: 'assistant', text: data.summary }]);
      // on first review, capture whichever key the server returned (sessionId or sid)
      const newSid = data.sessionId ?? data.sid;
      if (!sessionId && newSid) {
        setSessionId(newSid);
      }
      setPrompt('');
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
        body: JSON.stringify({ spec: parsed, sid: sessionId }),
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
            <div className="chat-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
              <div className="chat-history" style={{ flex: 1, overflowY: 'auto', padding: '1em' }}>
                {messages.map((msg, idx) => (
                  <div
                    key={idx}
                    style={{
                      display: 'flex',
                      justifyContent: msg.speaker === 'user' ? 'flex-start' : 'flex-end',
                      margin: '0.5em 0'
                    }}
                  >
                    <div className={`chat-message ${msg.speaker}`} style={{ maxWidth: '70%' }}>
                      <strong>{msg.speaker === 'user' ? 'You' : 'Implicitus'}:</strong> {msg.text}
                    </div>
                  </div>
                ))}
              </div>
              <form onSubmit={handleSubmit} style={{ borderTop: '1px solid #ccc', padding: '0.5em' }}>
                <input
                  type="text"
                  value={prompt}
                  onChange={e => setPrompt(e.target.value)}
                  placeholder="What do you want to design?"
                  disabled={loading}
                  style={{ width: '80%', marginRight: '0.5em' }}
                />
                <button type="submit" disabled={loading || !prompt.trim()}>
                  Generate
                </button>
              </form>
            </div>
            {error && <div className="error">{error}</div>}
          </div>
          <div style={{ flex: 1, overflow: 'auto' }}>
            {/* Editable JSON window */}
            {summary && (
              <div style={{ textAlign: 'left', margin: '1em', padding: '1em', border: '1px dashed #888' }}>
                <strong>JSON Spec:</strong>
                <Editor
                  height="300px"
                  defaultLanguage="json"
                  value={specText}
                  onChange={value => setSpecText(value!)}
                  onMount={handleEditorDidMount}
                  options={{
                    readOnly: modelProto !== null,
                    automaticLayout: true,
                    minimap: { enabled: false },
                    formatOnPaste: true,
                    formatOnType: true,
                  }}
                />
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
