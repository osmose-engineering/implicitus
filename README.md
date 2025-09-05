# Implicitus

**From Prompt to Print: A Continuous Implicit Modeling Platform**

Implicitus provides a fully implicit pipeline for manufacturing design, enabling seamless integration between text prompts, AI agents, and slicing engines—without intermediate mesh conversions.

## Current Capabilities

- **Protobuf Schema & Bindings**  
  Versioned `.proto` definitions for primitives (Box, Sphere, Cylinder) and transforms (Translate, Scale), with auto-generated Rust and Python bindings and JSON interoperability.

- **SDF Evaluation Engine**  
  High-performance Rust core library to evaluate signed-distance fields for loaded implicit models.

- **AI Adapter Stub**  
  Python module to transform text prompts into placeholder implicit models via protobuf messages.

- **Rust Slicer Service**  
  Warp-based microservice exposing `POST /slice` to compute layer contours using a marching-squares algorithm, then sorts and closes the loop into ordered contours.

- **Canvas API**  
  Express.js server providing:
  - `POST /models` to store implicit models in-memory.  
  - `GET /models/:id` to retrieve stored models.  
  - `GET /models/:id/slices?layer=<z>&...` to proxy slice requests to the Rust slicer and return JSON contours.

- **End-to-End Workflow**  
  Complete prompt → model storage → slicing pipeline validated via curl and Python scripts.

  ## Current Progress

- **LLM-driven specification parsing**: Users can enter natural language prompts which are sent to a local Mistral model. The model returns JSON specs that are parsed into intermediate "draft" specs.
- **Editable JSON branch workflow**: Draft specs are presented in an editable JSON pane. Users can manually adjust or correct the specification before committing.
- **Commit to locked model**: A "Confirm & Lock" action merges the draft JSON into the canonical Proto-based model. The locked model is rendered in the 3D preview.
- **3-panel UI layout**: Left side split into chat input (top) and editable JSON (bottom); right side shows the 3D preview.
- **Design API endpoints**:
  - `POST /design/review`: Accepts a natural language or JSON spec and returns the parsed spec + a human-readable summary.
  - `POST /design/submit`: Accepts a finalized spec JSON and maps it to the internal Proto, validating it for downstream processing.
- **Next steps**: Integrate ongoing conversational edits, branch history, and diff-based review workflows.

### Voronoi infill payload

Responses that include Voronoi edges now also return a matching `vertices` array.
Each entry in `edges` references two indices into this `vertices` list:

```json
{
  "modifiers": {
    "infill": {
      "pattern": "voronoi",
      "seed_points": [[0,0,0], [1,0,0]],
      "vertices": [[0,0,0], [1,0,0]],
      "edges": [[0,1]]
    }
  }
}
```


## Getting Started

**Prerequisites**  
- Rust ≥ 1.60  
- Python 3.8+  
- Node.js 14+

**Setup & Run**

```bash
# 1a) Start the Rust slicer service
cd core_engine
cargo build
cargo run --bin slicer_server   # listens on port 4000

# 1b) Start the API Server) Start the Rust slicer service
uvicorn design_api.main:app --host 127.0.0.1 --port 8000 --reload   # listens on port 8000

# 1c) Start the UI Server
cd implicitus_ui
npm run dev # listens on port 3000

# 2) Start the Canvas API
cd ../canvas_api
npm install
node server.js                  # listens on port 3000

# 3) Run the AI adapter stub (Python)
cd ../ai_adapter
pip install -r requirements.txt
python3 - << 'EOF'
from ai_adapter.adapter import generate_model
print(generate_model("unit sphere"))
EOF

# 4) Test slicing end-to-end via curl
curl -X POST http://localhost:3000/models \
     -H 'Content-Type: application/json' \
     -d '{"id":"test_sphere","root":{"primitive":{"sphere":{"radius":1.0}}}}'
curl "http://localhost:3000/models/test_sphere/slices?layer=0.0&nx=5&ny=5"

# Or run the Python test script
cd ../
python3 test_slice.py

## UI Unit Tests

Execute the frontend test suite with [Vitest](https://vitest.dev):

```bash
cd implicitus-ui
npm test
# or
npx vitest
```

## Logging

The Design API configures Python logging before any Voronoi utilities are
imported. The log level can be adjusted using the `IMPLICITUS_LOG_LEVEL`
environment variable, which defaults to `DEBUG` for development. Debug-level
logging is required to capture the `REPO ROOT` and `UNIFORM_CELL_DUMP.json`
entries emitted by `design_api/services/voronoi_gen/uniform/construct.py`.

```bash
# override the default log level
export IMPLICITUS_LOG_LEVEL=INFO
```

Run with `IMPLICITUS_LOG_LEVEL=DEBUG` to diagnose Voronoi cell generation and
record the uniform cell dump for inspection.

To trace how seed data flows through the system, enable seed debug logging:

```bash
export IMPLICITUS_DEBUG_SEEDS=1
```

With the flag set, the Design API will write unsanitized turn entries to
`design_api/seed_debug.log` alongside `conversation_log.jsonl`. These JSON lines
preserve `seed_points`, `vertices`, and related fields that are otherwise
scrubbed from the main conversation log.
