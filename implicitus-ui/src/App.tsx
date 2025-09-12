// Reorders infill keys: configurable params first, then bbox_min, bbox_max, seed_points last
function reorderInfillKeys(node: any) {
  const infillOriginal = node.modifiers?.infill;
  if (!infillOriginal) return node;
  // clone infill and normalize spacing
  const infill: any = { ...infillOriginal };
  // rename snake_case and camelCase keys to spacing
  if (infill.hasOwnProperty('min_dist')) {
    infill.spacing = infill.min_dist;
    delete infill.min_dist;
  }
  if (infill.hasOwnProperty('minDist')) {
    infill.spacing = infill.minDist;
    delete infill.minDist;
  }
  const ordered: any = {};
  const primaryKeys = [
    'pattern','spacing','wall_thickness',
    'uniform','num_points','adaptive','max_depth',
    'threshold','resolution','shell_offset','auto_cap'
  ];
  // add primary keys in order
  primaryKeys.forEach(k => {
    if (infill.hasOwnProperty(k)) ordered[k] = infill[k];
  });
  // then bbox_min, bbox_max, seed_points last
  ['bbox_min','bbox_max','seed_points'].forEach(k => {
    if (infill.hasOwnProperty(k)) ordered[k] = infill[k];
  });
  // clone other keys (if any) in original order
  Object.keys(infill).forEach(k => {
    if (!ordered.hasOwnProperty(k)) ordered[k] = infill[k];
  });
  // return new node with reordered infill
  return {
    ...node,
    modifiers: {
      ...node.modifiers,
      infill: ordered
    }
  };
}

function reorderSpec(specArray: any[]) {
  return specArray.map(node => reorderInfillKeys(node));
}
import React, { useState, useRef, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import * as monaco from 'monaco-editor';
import './App.css'
import VoronoiCanvas, { EDGE_Z_VARIATION_TOLERANCE } from './components/VoronoiCanvas';
import { Checkbox } from './components/UI';
import { Tabs, TabList, Tab, TabPanel } from 'react-tabs';
import 'react-tabs/style/react-tabs.css';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
// Simple debounce helper
function debounce<T extends (...args: any[]) => void>(fn: T, delay: number) {
  let timeout: ReturnType<typeof setTimeout> | null;
  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => {
      fn(...args);
    }, delay);
  };
}

function App() {
  const [isDirty, setIsDirty] = useState(false);
  const [prompt, setPrompt] = useState('');
  const [spec, setSpec] = useState<any[]>([]);
  const [meshVertices, setMeshVertices] = useState<[number, number, number][]>([]);
  const [meshEdges, setMeshEdges] = useState<number[][]>([]);
  const [sliceSeedPoints, setSliceSeedPoints] = useState<[number, number, number][]>([]);
  // Seed points from slicer server take precedence if available
  const seedPoints: [number, number, number][] =
    sliceSeedPoints.length > 0
      ? sliceSeedPoints
      : spec[0]?.modifiers?.infill?.seed_points ?? [];
  const edges = meshEdges;
  // Reset mesh whenever the spec changes; geometry will be fetched separately.
  useEffect(() => {
    const infill = spec[0]?.modifiers?.infill;
    const verts = infill?.cell_vertices || infill?.vertices;
    const edgeList = infill?.edge_list || infill?.edges;
    setMeshVertices(Array.isArray(verts) ? verts : []);
    setMeshEdges(Array.isArray(edgeList) ? edgeList : []);
  }, [spec]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState<string>('');
  const [specText, setSpecText] = useState<string>('');
  const [modelProto, setModelProto] = useState<any | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<{speaker: 'user'|'assistant'; text: string}[]>([]);
  // Layer visibility toggles for preview
  const [visibility, setVisibility] = useState<{[key: string]: boolean}>({
    primitive: true,
    infill: true,
  });
  const [showRaymarch, setShowRaymarch] = useState(true);
  const [showStruts, setShowStruts]     = useState(false);
  const [strutRadius, setStrutRadius]   = useState(0.02);
  const [edgeZTolerance, setEdgeZTolerance] = useState(EDGE_Z_VARIATION_TOLERANCE);

  const [tabIndex, setTabIndex] = useState(1);

  const fetchVoronoiMesh = async (pts: [number, number, number][]) => {
    if (!pts || pts.length === 0) return;
    try {
      const resp = await fetch(`${API_BASE}/design/mesh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seed_points: pts }),
      });
      if (!resp.ok) return;
      const data = await resp.json();
      setMeshVertices(Array.isArray(data.vertices) ? data.vertices : []);
      setMeshEdges(Array.isArray(data.edges) ? data.edges : []);
    } catch (err) {
      console.error('[UI] mesh fetch error:', err);
    }
  };

  const fetchSlice = async (model: any) => {
    if (!model) return;
    try {
      const postResp = await fetch(`${API_BASE}/models`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(model),
      });
      if (!postResp.ok) {
        setError(`Model upload failed: ${postResp.status} ${postResp.statusText}`);
        return;
      }
      const postData = await postResp.json().catch(() => null);
      const modelId = postData?.id;
      if (!modelId) {
        setError('Model upload response missing id');
        return;
      }
      const resp = await fetch(`${API_BASE}/models/${modelId}/slices?layer=0`);
      if (!resp.ok) {
        const text = await resp.text();
        const msg = `Slice request failed: ${resp.status} ${resp.statusText} - ${text}`;
        console.error('[UI] slice request failed:', msg);
        setError(msg);
        throw new Error(msg);
      }
      const data = await resp.json();
      if (data.debug) {
        console.log('[design_api] debug info:', data.debug);
      }
      const seeds =
        Array.isArray(data.seed_points)
          ? data.seed_points
          : Array.isArray(data.debug?.seed_points)
            ? data.debug.seed_points
            : null;
      if (seeds) {
        setSliceSeedPoints(seeds);
        console.log('[design_api] seed points:', seeds);
      } else {
        setSliceSeedPoints([]);
        console.warn('[design_api] no seed points found in slice response');
        setError('No seed points found in slice response');
      }
    } catch (err: any) {
      console.error('[UI] slice fetch error:', err);
      setError(err.message || 'Slice fetch error');
      throw err;
    }
  };

  useEffect(() => {
    if (sliceSeedPoints.length > 0) {
      fetchVoronoiMesh(sliceSeedPoints);
    } else {
      setMeshVertices([]);
      setMeshEdges([]);
    }
  }, [sliceSeedPoints]);

  const handleValidate = async () => {
    setError(null);
    setLoading(true);
    try {
      const parsed = JSON.parse(specText);
      // include sessionId if available
      const validateBody: any = { spec: parsed };
      if (sessionId) validateBody.sid = sessionId;
      const response = await fetch(`${API_BASE}/design/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(validateBody),
      });
      const rawText = await response.text();
      if (!response.ok) {
        throw new Error(`Validation error: ${rawText}`);
      }
      const data = JSON.parse(rawText);
      setSummary(data.summary || 'Validation successful.');
      setMessages(prev => [...prev, { speaker: 'assistant', text: `Validation: ${data.summary}` }]);
    } catch (err: any) {
      setError(err.message || 'Validation failed.');
    } finally {
      setLoading(false);
      setIsDirty(false);
    }
  };

  const handleEditorDidMount = (editor: monaco.editor.IStandaloneCodeEditor, monacoInstance: typeof monaco) => {
    monacoInstance.languages.json.jsonDefaults.setDiagnosticsOptions({
      // Draft mode: only basic syntax and comments validation, no schema enforcement
      validate: true,
      allowComments: true,
      schemas: [],  // disable JSON schema checks to allow free-form editing (e.g., voronoi fields)
    });
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    // keep existing sessionId for follow-up updates
    setError(null);
    setLoading(true);
    setMessages(prev => [...prev, { speaker: 'user', text: prompt }]);
    console.log('[UI] Sending prompt →', prompt);
    let bodyData: any = {};
    // For updates (existing session), always send raw prompt and current spec
    if (sessionId) {
      bodyData.raw = prompt;
      bodyData.spec = spec;
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
      // use /design/update for any follow-up on an existing spec,
      // or when there is already a spec and the user enters a new prompt
      const isUpdate = Boolean(sessionId) || (spec.length > 0 && prompt.trim().length > 0);
      const endpoint = isUpdate ? '/design/update' : '/design/review';
      console.log('[UI] using endpoint →', endpoint);
      const response = await fetch(`${API_BASE}${endpoint}`, {
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
        console.log('[UI] API spec:', data.spec);
      } catch (parseError) {
        console.error('[UI] JSON parse error:', parseError);
        throw new Error(`Failed to parse JSON spec: ${parseError}`);
      }
      // Handle updated spec with nested modifiers
      if (data.spec && Array.isArray(data.spec)) {
        const infill = data.spec[0]?.modifiers?.infill;
        console.log('[UI] backend infill counts:', {
          points: infill?.seed_points?.length,
          edges: infill?.edges?.length,
          sampleEdges: infill?.edges?.slice(0, 5),
        });
        if (infill?.debug) {

          console.log('[design_api] debug info:', infill.debug);

        }
        setSpec(data.spec);
        setSpecText(JSON.stringify(reorderSpec(data.spec), null, 2));
        if (infill?.seed_points) {
          fetchVoronoiMesh(infill.seed_points);
        }
        await fetchSlice({ id: 'preview', root: { children: data.spec } });
        if (data.summary) {
          setSummary(data.summary);
          setMessages(prev => [...prev, { speaker: 'assistant', text: data.summary }]);
        }
        // Capture session ID for subsequent updates
        if (!sessionId && data.sid) {
          setSessionId(data.sid);
        }
        setIsDirty(false);
        setLoading(false);
        setPrompt('');
        return;
      }
      if (data.question) {
        // Extract plain question text from the JSON-wrapped response
        let questionText: string;
        if (typeof data.question === 'string') {
          try {
            const parsedQ = JSON.parse(data.question);
            questionText = parsedQ.question ?? parsedQ.query ?? data.question;
          } catch {
            questionText = data.question;
          }
        } else {
          const qObj = data.question as any;
          questionText = qObj.question ?? qObj.query ?? String(data.question);
        }
        // Display the LLM's follow-up question to the user
        setMessages(prev => [...prev, { speaker: 'assistant', text: questionText }]);
        // Pre-fill the prompt input with the LLM question
        setPrompt(questionText);
        const newSid = data.sessionId ?? data.sid;
        if (!sessionId && newSid) {
          setSessionId(newSid);
        }
        setLoading(false);
        return;
      }
      // Updated block per instructions
      const isInitial = !sessionId;
      let assistantText: string;
      if (isInitial && data.summary) {
        assistantText = `Created a ${data.summary}`;
      } else {
        assistantText = data.confirmation ?? data.summary;
      }
      setMessages(prev => [...prev, { speaker: 'assistant', text: assistantText }]);
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
      const response = await fetch(`${API_BASE}/design/submit`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ spec: parsed, sid: sessionId, raw: "" }),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Server error: ${response.status} ${response.statusText} - ${text}`);
      }
      const data = await response.json();
      console.log('API spec:', data.spec);
      console.log('[UI] Confirm response data:', data);
      if (data.debug) {

        console.log('[design_api] debug info:', data.debug);

      }
      setModelProto(data);
      if (data.locked_model) {
        await fetchSlice(data.locked_model);
      }
    } catch (err: any) {
      setError(err.message || 'An error occurred during confirm');
    } finally {
      setLoading(false);
    }
  };

  const handleApplyChanges = async () => {
    setError(null);
    setLoading(true);
    try {
      const parsed = JSON.parse(specText);
      // strip any existing seed_points to avoid sending large arrays back,
      // but cache them locally so we can restore after update
      const removedSeeds: [number, number, number][][] = [];
      const specToSend = parsed.map((node: any, idx: number) => {
        if (node.modifiers?.infill?.seed_points) {
          removedSeeds[idx] = node.modifiers.infill.seed_points;
          const { seed_points, ...restInfill } = node.modifiers.infill;
          return {
            ...node,
            modifiers: {
              ...node.modifiers,
              infill: restInfill,
            },
          };
        }
        return node;
      });
      const response = await fetch(`${API_BASE}/design/update`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ spec: specToSend, sid: sessionId, raw: '' }),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Server error: ${response.status} ${response.statusText} - ${text}`);
      }
      const data = await response.json();
      if (data.spec && Array.isArray(data.spec)) {
        // Reattach seed_points from response or cached copy
        const updatedSpec = data.spec.map((node: any, idx: number) => {
          const infill = node.modifiers?.infill;
          const cached = removedSeeds[idx];
          if (!infill?.seed_points && cached) {
            if (infill) {
              infill.seed_points = cached;
            } else {
              node.modifiers = { ...(node.modifiers || {}), infill: { seed_points: cached } };
            }
          }
          return node;
        });
        setSpec(updatedSpec);
        setSpecText(JSON.stringify(reorderSpec(updatedSpec), null, 2));
        const infill = updatedSpec[0]?.modifiers?.infill;
        if (infill?.debug) {

          console.log('[design_api] debug info:', infill.debug);

        }
        const seeds = infill?.seed_points;
        if (seeds) {
          setSliceSeedPoints(seeds);
          fetchVoronoiMesh(seeds);
        }
        setIsDirty(false);
        if (data.summary) {
          setSummary(data.summary);
          setMessages(prev => [...prev, { speaker: 'assistant', text: data.summary }]);
        }
      }
    } catch (err: any) {
      setError(err.message || 'An error occurred while applying changes');
    } finally {
      setLoading(false);
    }
  };

  // Helper: Regenerate infill seed points if needed
  const regenerateSeeds = async (currentSpec: any[]) => {
    // Strip out any existing seed_points so backend regenerates fresh
    const specToSend = currentSpec.map(node => {
      if (node.modifiers?.infill) {
        const { seed_points, ...restInfill } = node.modifiers.infill;
        return {
          ...node,
          modifiers: {
            ...node.modifiers,
            infill: restInfill,
          },
        };
      }
      return node;
    });
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/design/update`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ spec: specToSend, sid: sessionId, raw: "" }),
      });
      const rawText = await response.text();
      if (!response.ok) throw new Error(rawText);
      const data = JSON.parse(rawText);
      if (data.spec && Array.isArray(data.spec)) {
        setSpec(data.spec);
        setSpecText(JSON.stringify(data.spec, null, 2));
        const infill = data.spec[0]?.modifiers?.infill;
        if (infill?.debug) {

          console.log('[design_api] debug info:', infill.debug);

        }
        if (infill?.seed_points) {
          fetchVoronoiMesh(infill.seed_points);
        }
        if (data.summary) {
          setSummary(data.summary);
          setMessages(prev => [...prev, { speaker: 'assistant', text: data.summary }]);
        }
      }
    } catch (err: any) {
      console.error('[UI] regenerateSeeds error:', err);
      setError(err.message || 'Failed to regenerate seeds');
    } finally {
      setLoading(false);
    }
  };
  // Debounced regenerateSeeds ref
  const debouncedRegenerateSeeds = useRef(
    debounce((parsed: any[]) => {
      regenerateSeeds(parsed);
    }, 400)
  ).current;

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
              <form
                onSubmit={handleSubmit}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5em',
                  borderTop: '1px solid #ccc',
                  padding: '0.5em'
                }}
              >
                <input
                  type="text"
                  value={prompt ?? ''}
                  onChange={e => setPrompt(e.target.value || '')}
                  placeholder="What do you want to design?"
                  disabled={loading}
                  style={{ width: '80%', marginRight: '0.5em' }}
                />
                <button type="submit" disabled={loading || !(prompt?.trim())}>
                  Generate
                </button>
              </form>
            </div>
            {error && <div className="error">{error}</div>}
          </div>
          <div style={{ flex: 1, overflow: 'auto' }}>
            {/* Editable JSON window */}
            {specText && (
              <div style={{ textAlign: 'left', margin: '1em', padding: '1em', border: '1px dashed #888' }}>
                <strong>JSON Spec:</strong>
                <Editor
                  height="300px"
                  language="json"
                  theme="vs-light"
                  value={specText}
                  onChange={value => {
                    const text = value ?? '';
                    setSpecText(text);
                    setIsDirty(true);
                  }}
                  onMount={handleEditorDidMount}
                  options={{
                    readOnly: modelProto !== null,
                    automaticLayout: true,
                    minimap: { enabled: false },
                    formatOnPaste: true,
                    formatOnType: true,
                  }}
                />
                <button onClick={handleApplyChanges} disabled={loading || !isDirty}>
                  Apply Changes
                </button>
                <button onClick={handleConfirm} disabled={loading}>
                  Finalize
                </button>
                <button onClick={handleValidate} disabled={loading || !isDirty}>
                  Validate
                </button>
                {/* Removed Max Seed Count and Desired Seed Count UI */}
                {spec.length > 0 && spec[0].modifiers?.infill?.seed_points && (
                  <div style={{ marginTop: '1em' }}>
                    <strong>Seed Points:</strong>
                    <textarea
                      rows={6}
                      style={{ width: '100%', height: '120px', fontFamily: 'monospace', fontSize: '0.9em' }}
                      value={JSON.stringify(spec[0].modifiers.infill.seed_points, null, 2)}
                      onChange={e => {
                        const text = (e.target as HTMLTextAreaElement).value;
                        try {
                          const pts = JSON.parse(text);
                          const newSpec = [...spec];
                          newSpec[0] = {
                            ...newSpec[0],
                            modifiers: {
                              ...newSpec[0].modifiers,
                              infill: {
                                ...newSpec[0].modifiers.infill,
                                seed_points: pts
                              }
                            }
                          };
                          setSpec(newSpec);
                          setSpecText(JSON.stringify(newSpec, null, 2));
                        } catch {
                          // ignore parse errors
                        }
                      }}
                    />
                  </div>
                )}
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
          {/* Layer visibility toggles */}
          <div style={{ margin: '1em', textAlign: 'left' }}>
            <label>
              <input
                type="checkbox"
                checked={visibility.primitive}
                onChange={() => setVisibility(v => ({ ...v, primitive: !v.primitive }))}
              />
              Show Solid
            </label>
            <label style={{ marginLeft: '1em' }}>
              <input
                type="checkbox"
                checked={visibility.infill}
                onChange={() => setVisibility(v => ({ ...v, infill: !v.infill }))}
              />
              Show Infill
            </label>
            <label style={{ marginLeft: '1em' }}>
              Z Tol:
              <input
                type="number"
                value={edgeZTolerance}
                step={1e-6}
                min={0}
                onChange={e => setEdgeZTolerance(e.target.valueAsNumber)}
                style={{ width: '6em', marginLeft: '0.5em' }}
              />
            </label>
          </div>

          {/* Tabbed preview: Ray-March and Strut views */}
          <Tabs selectedIndex={tabIndex} onSelect={index => setTabIndex(index)}>
            <TabList>
              <Tab>Ray-March View</Tab>
              <Tab>Strut View</Tab>
            </TabList>
            <TabPanel>
              {seedPoints.length > 0 && (
                <VoronoiCanvas
                  key="ray"
                  seedPoints={seedPoints}
                  vertices={meshVertices}
                  edges={edges}
                  infillPoints={meshVertices}
                  infillEdges={edges}
                  bbox={[0, 0, 0, 1, 1, 1]}
                  thickness={0.35}
                  maxSteps={256}
                  epsilon={0.001}
                  showSolid={visibility.primitive}
                  showInfill={visibility.infill}
                  showRaymarch={true}
                  showStruts={false}
                  edgeZVariationTolerance={edgeZTolerance}
                />
              )}
            </TabPanel>
            <TabPanel>
              {seedPoints.length > 0 && (
                <VoronoiCanvas
                  key="strut"
                  seedPoints={seedPoints}
                  vertices={meshVertices}
                  edges={edges}
                  infillPoints={meshVertices}
                  infillEdges={edges}
                  bbox={[0, 0, 0, 1, 1, 1]}
                  thickness={0.35}
                  maxSteps={256}
                  epsilon={0.001}
                  showSolid={visibility.primitive}
                  showInfill={visibility.infill}
                  showRaymarch={false}
                  showStruts={true}
                  strutRadius={strutRadius}
                  strutColor="white"
                  edgeZVariationTolerance={edgeZTolerance}
                />
              )}
            </TabPanel>
          </Tabs>
        </div>
      </div>
    </div>
  );
}

export default App
