

# Backlog

## Done
- Parsed user requests into primitives (sphere, box, cylinder) with size_mm.
- Detected and applied infill modifiers (pattern, density).
- Supported boolean operations (union, difference, intersection).
- Implemented `parse_raw_spec`, `generate_summary`, `review_request`, `update_request` backend functions.
- Exposed REST endpoints: POST `/design/review`, `/design/update`, `/design/submit`.
- Managed session state with `draftSpec` and `lockedModel` in memory.
- Frontend chat panel for user prompts.
- Frontend editable JSON panel (Monaco editor) for spec revisions.
- Frontend 3‑panel layout: chat input, JSON editor, 3D preview.
- 3D preview rendering of basic primitives via Three.js (or equivalent).
- Session-based update flow: follow-up prompts mutate existing spec.

## To-Do
- **Basic solver** to capture early user activity for training
- **Optimiser loop** to capture diff between model built and desired model.
- **Primitive expansion** to fully encompass range of possible shapes/actions.
- **SDF‑based GPU ray‑marching** for interactive infill preview (gyroid, Voronoi lattices).
- **Adaptive sampling & LOD** strategies for complex lattices and scanned parts.
- **JSON diff/highlight** view to show spec changes.
- **Visual before/after diff** panel for spec edits.
- **Enhanced chat UI**: message alignment, persistent conversational threads.
- **Complex primitives & boolean chains** support in UI and backend.
- **Scanned mesh import & remap** (resample SDF) functionality.
- **Model export**: bake final mesh or generate printable output formats.
- **Full conversational context** with history-driven model updates.
- **UI polish**: color‑coding, error highlighting, inline validation.