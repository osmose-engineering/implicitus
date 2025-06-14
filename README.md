

# Implicitus

**From Prompt to Print: A Continuous Implicit Modeling Platform**

---

## Sprint 1: MVP Snapshot

**Sprint Goal**  
Deliver a minimal end-to-end text→implicit→slice pipeline showcasing:

- A basic `.proto` schema for primitives and transforms.
- Core engine stub capable of loading schema and evaluating SDF at sample points.
- AI adapter stub reading prompts and generating placeholder implicit models.
- Canvas API stub with `POST /models` and `GET /models/{id}/slices?layer=` end-points.
- Example prompt and automated example script demonstrating the flow.

**Timeline**  
2-week iteration: define schema, implement stubs, write example, add tests.

**Deliverables**  
- `schema/implicitus.proto` with minimal primitives.
- `core_engine` stub with SDF evaluator API.
- `ai_adapter` stub with mock LLM call.
- `canvas_api` stub server.
- `examples/hello_world.json` and placeholder slice output.
- README documenting setup and example usage.

---

## Getting Started

**Prerequisites**  
- Rust >=1.60 (for `core_engine`)  
- Python 3.8+ (for `ai_adapter` & examples)  
- Node.js 14+ (for `canvas_api`)  

**Setup**  
```bash
# Clone the repo
git clone https://github.com/your-org/implicitus.git
cd implicitus

# Build core engine
cd core_engine
cargo build

# Install Python dependencies
cd ../ai_adapter
pip install -r requirements.txt

# Install Node dependencies for API
cd ../canvas_api
npm install
```

**Running the Example**  
```bash
# From project root
make example
```

---

## Next Steps

- Flesh out schema with transforms and Booleans  
- Implement real LLM integration  
- Expand validation pipeline  
- Implement slice exporter  

---

## Contributing

Please read **CONTRIBUTING.md** for guidelines on how to help out.