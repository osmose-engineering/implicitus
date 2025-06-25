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


## Getting Started

**Prerequisites**  
- Rust ≥ 1.60  
- Python 3.8+  
- Node.js 14+

**Setup & Run**

```bash
# 1) Start the Rust slicer service
cd core_engine
cargo build
cargo run --bin slicer_server   # listens on port 4000

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