# COPILOT_INSTRUCTIONS.md — GUIRA (Fire Detection Drone System) — Agent Master Instructions

## 0 — Purpose & scope

This file defines **non-negotiable rules** and workflows any automated agent or human engineer must follow when modifying or extending GUIRA. GUIRA implements drone-based wildfire detection, live ingestion, mapping, simulation, and RAG-enabled analysis pipelines. All code generation, edits, training, CI, docs, and infra work in this repository MUST conform to the rules below.

---

## 1 — Top-level rules (non-negotiable)

1. **Repo-first approach**: Always scan existing files before adding or copying code. Reuse existing modules and patterns. Run:

   ```bash
   git ls-files | xargs grep -nE "yolo|timesformer|csrnet|vari|spread|dji|ingest|indexer|frontend|webrtc|live" || true
   ```
2. **Four mandatory metadata blocks** for any new model code or model directory (`models/<name>/README.md` or header comment in scripts):

   * `MODEL:` exact model + version + weight path
   * `DATA:` dataset names & links + local path conventions
   * `TRAINING/BUILD RECIPE:` hyperparams, transforms, training command, compute needs
   * `EVAL & ACCEPTANCE:` key metrics, thresholds, test scripts
3. **Place code adjacent to related modules.** New module directories must include a `README.md` describing purpose and usage.
4. **Tests with code:** Every new module must have unit tests in `tests/<module>/` and example minimal data in `tests/data/<module>/`.
5. **All route orchestration must be inside the route function** (user preference). If a background worker is required, the route must still contain the orchestration & enqueue logic.
6. **Every change** must include: docstrings (Google style), inline comments for non-obvious logic, at least one unit test, and update to `docs/CHANGELOG.md`.

---

## 2 — When assigned a task (agent checklist)

When starting any change, automatically perform:

1. **Repo-scan**

   * List candidate files:
     `git ls-files | grep -E "(models|backend|frontend|ingest|controller|simulation|docs)" > /tmp/guira_files.txt`
   * Open top 5 matches relevant to the feature and scan for TODOs/FIXMEs.

2. **Context summary**

   * Create `docs/dev_context_<YYYYMMDD>.md` with a 5–8 line summary: existing models, endpoints, pipelines, test coverage.

3. **Design proposal**

   * Create `src/gui ra/<module>/PROPOSAL.md` (or `docs/proposals/`) with 5–10 bullets: files to create/modify, tests, acceptance criteria.

4. **Implement**

   * Follow coding standards (Section 3). Keep changes minimal per PR.

5. **Write tests**

   * Add unit & integration tests under `tests/` and minimal sample data under `tests/data/`.

6. **Self-check**

   * Run linting & test suite locally; write `dev/logs/<task>-report.txt` with results.

7. **Branch & PR**

   * Branch name `feature/<module>-<short>` or `fix/<module>-<short>`.
   * Commit message: `module(scope): short — #<ticket>`.
   * PR body must include the `PROPOSAL.md`, test outputs, and run instructions.

---

## 3 — Coding standards & style

* **Python 3.10+** with type hints in all public APIs.
* Use **PEP8**: run `black`, `ruff`, `mypy` as pre-commit hooks.
* **Docstrings**: Google style for modules, classes, functions.
* **Logging**: structured logging (include `session_id`, `user_id`, `trace_id`).
* **Secrets**: never hardcode. Use environment variables or KeyVault/KMS. Provide `.env.example`.
* **Dependencies**: add to `requirements.txt` and `dev-requirements.txt`. Update `Dockerfile` & CI.
* **Idempotency**: scripts must be re-runnable and safe.

---

## 4 — File / folder conventions

* Code root: `src/gui ra/`

  * `backend/` (FastAPI/Quart routes, ingest, controller)
  * `models/` (training scripts & model-specific code)
  * `ingest/` (session_indexer, pipelines)
  * `controller/` (DJI & device connectors)
  * `simulation/` (scenario runner)
  * `frontend/` (React + Vite)
* Data: `data/raw/`, `data/processed/`, `data/manifests/`
* Experiments & models: `experiments/`, `models/`
* Docs: `docs/` and `docs/models/`, `docs/datasets/`, `docs/integrations/`
* Tests: `tests/` with `tests/data/` minimal fixtures

---

## 5 — API & route rules (FastAPI/Quart)

* All routes placed under `src/gui ra/backend/api/`.
* **Route responsibilities**:

  * Parse & validate inputs (use Pydantic).
  * Short-circuit invalid requests.
  * Orchestrate pipeline inside the route or call a small local helper (defined in same file).
  * For long-running tasks, enqueue background job but return `{status, job_id, diagnostics}`.
* Return JSON:
  `{ "status": "ok"|"error", "job_id": <uuid>?, "artifacts": {...}, "diagnostics": {...} }`
* Logs: audit who called what, payload sizes, duration.

---

## 6 — AI model integration rules (applies to all models)

For any model-related directory add `README.md` with MODEL/DATA/TRAINING/EVAL blocks.

**A. MODEL block**

* Model name + precise version, pre-trained base, weights path.

**B. DATA block**

* Exact datasets used & local paths. Include sample dataset in `tests/data/`.

**C. TRAINING/BUILD RECIPE**

* Hyperparameters, transforms, augmentations, training command, expected GPU memory.

**D. EVALUATION/ACCEPTANCE**

* List metrics + thresholds. Provide `scripts/evaluate_<model>.py`.

**E. ARTIFACTS**

* Checkpoint paths: `models/<name>/runs/<timestamp>/best.pt`
* Exported formats (ONNX/TorchScript/TensorRT) and inference wrapper.

**Implementation rule**: all model training scripts accept `--config config.yaml` and `--override key=val`.

---

## 7 — Dataset & registry rules

* Maintain `docs/datasets/REGISTRY.md` enumerating sources with licenses and local download manifest keys.
* All download scripts placed in `scripts/download_<dataset>.py` and **must**:

  * Validate checksums,
  * Save LICENSE/README in `docs/datasets/<name>.md`,
  * Write `data/manifests/<dataset>_{train,val,test}.json` with `{image, label, metadata}`.
* Never commit raw datasets.

---

## 8 — Ingest / Index / RAG rules

* Reuse `ingest/session_indexer.py` — add streaming entrypoint `index_live_chunk(session_id, chunk_metadata, is_final=False)`.
* Document vector schema in `infra/search_index_schema.json`.
* Vector doc schema must include:

  * `id, session_id, user_id, doc_type, start_ts, end_ts, text, vector, metadata, source_blob_url`.
* Retrieval: default top-K=6 + metadata filters. Include metadata JSON in prompt.

---

## 9 — Live ingestion & screen-share (webRTC / WebSocket)

**High-level rules** (full spec must be present in `docs/live/README.md`):

### Endpoints

* `POST /api/live/session/start`:

  * Body: `{user_id, label, source, source_platform, options}`
  * Generates `session_id`, WebRTC offer/answer credentials or `ws_url`, token.
  * Must store session record and consent flags.

* `POST /api/live/session/stop`:

  * Finalizes session; calls `ingest/session_indexer.index_final(session_id)`.

* `POST /api/live/{session_id}/feedback/request`:

  * Accept coaching query. Return `feedback_job_id` if heavy; stream partial results if quick.

### Ingest modes

* **Preferred**: WebRTC → media server (Janus/mediasoup/Pion) → processing workers.
* **Fallback**: chunked WebSocket frames (JPEG/PNG) to `ws://server/live/{session_id}/upload`.

### Processing pipeline (per frame or batch)

1. Decode frame → detection (YOLOv8n for live) → tracker (ByteTrack) → pose (BlazePose for speed).
2. Emit overlays to `ws://server/ws/session/{session_id}` for frontend rendering.
3. Aggregate sliding windows → event spotter → index chunk via `index_live_chunk`.
4. If `store_raw` → roll short clips to blob storage and update `source_blob_url`.

### Latency targets

* Overlay emit: ≤ 1.2s median per frame under target GPU.
* Chunk embedding + upsert: background job confirmed within 30s.

### Privacy & broadcast rules

* Must require `user_acknowledges_broadcast_rights` checkbox for TV/DRM captures.
* Provide UI toggles: `mask_faces`, `mask_audio`, `blur_crowd`.
* Default to conservative privacy settings (mask=True, retention 90 days).

---

## 10 — DJI Cloud API & controller

* Place controller code in `src/gui ra/controller/`.
* Store connector info in `data/connectors/` or secure store (KeyVault).
* `controller/` responsibilities:

  * Authenticate with DJI Cloud using env vars: `DJI_APP_ID`, `DJI_APP_KEY`, `DJI_APP_LICENSE`.
  * Subscribe to telemetry (MQTT) topics; parse messages.
  * Fetch cloud stream HLS/RTMP endpoints or orchestrate device streaming.
  * Relay telemetry & video to backend via internal API or Pub/Sub (Redis).
* Add `docs/integrations/DJI_CLOUD_API.md` describing registration, scopes, MQTT topics, sample code.

---

## 11 — Models & training reference (short recipes)

Place these in `docs/models/<model>.md` and also embed in each model folder.

### Fire Detection — YOLOv8 (recommended)

* Data: `flame_rgb`, `flame2_rgb_ir`, `sfgdn_fire`, `wit_uas_thermal`.
* Classes: `fire` (0), `smoke` (1).
* Command example:

  ```bash
  python models/fire_yolov8/train_fire.py --data data/fire/data.yaml --model yolov8s.pt --img 640 --epochs 150 --batch 16 --device 0
  ```
* Augment: mosaic, brightness, synthetic smoke overlay.
* Eval: mAP@0.5 >= 0.6 (site-dependent). Provide `scripts/evaluate_fire.py`.

### Smoke (temporal) — TimeSFormer

* Clips: 16 frames, 8 fps window.
* Command example:

  ```bash
  python models/smoke_timesformer/train_smoke.py --manifest data/smoke/manifest.jsonl --frames 16 --epochs 30
  ```
* Eval: AUC, F1.

### Fauna — YOLOv8 + CSRNet

* YOLO for bounding boxes; CSRNet for density/count.
* Ensure taxonomy mapping in `configs/labelmaps/fauna.yaml`.

### Vegetation — ResNet50 + VARI

* VARI computed and passed as auxiliary feature or extra channel.

### Fire spread — Hybrid (physics+NN)

* Input: raster stacks (DEM, fuel, wind grids).
* Output: future burn masks. Loss includes physics regularizer.

---

## 12 — Simulation & scenario runner

* Put simulation code in `simulation/runner.py` and scenario YAMLs in `simulation/scenarios/`.
* Runner must emit telemetry + video frames to the controller or to backend ingest endpoints so full-stack can run offline.
* Create at least one canonical scenario `scenarios/north-forest-fire-test.yaml` for CI smoke test.

---

## 13 — Frontend UI conventions & components

* Frontend under `src/gui ra/frontend/` (Vite + React + TypeScript).
* New UI components must follow this pattern:

  * `components/ScreenShareWizard.tsx` — start screen/camera share wizard.
  * `pages/LiveAnalysis.tsx` — mainstream live analysis page.
  * `components/MapView.tsx`, `CameraFeed.tsx`, `ScenarioPlayer.tsx`, `ExampleList.tsx`.
* Styling: clean, minimal; grey and blue palette; white background; avoid heavy gradients.
* Use WebSocket or SSE for overlays. Streaming LLM feedback via SSE or WebSocket.

---

## 14 — Tests, CI & performance

* Add unit tests for each module. Coverage target: ≥ 80% for new modules.
* CI (`.github/workflows/ci.yml`) must:

  * Install deps, run `pytest -q`, run `ruff`, `black --check`, `mypy`.
  * Run smoke training jobs (1 epoch or small-batch tests) for each model to catch shapes/import errors.
  * Run simulation smoke test: run `python simulation/runner.py scenarios/sample.yaml` and confirm backend receives at least one detection.
* Performance scripts under `tests/perf/` must measure detection throughput on sample frames.

---

## 15 — Security, privacy, and compliance (must follow)

* Encryption: use KeyVault/KMS for tokens and refresh tokens.
* Retention: default 90 days for video; per-user override with audit trail.
* Consent: for live capture, show explicit consent modal and store `consent_timestamp` and flags.
* Deletion: implement `DELETE /api/user/{user_id}/erase` to soft-delete metadata, delete blobs per policy, and remove vector docs.
* PII: do not log raw emails or IDs in public logs.
* Legal: block or warn on attempts to capture DRM-protected streams (best-effort detection); require user acknowledgment.

---

## 16 — CI/CD, infra & deployment

* Provide two infra options:

  * **Use existing Azure resources** (preferred): verify with `az` CLI (storage, search, openai).
  * **Provision infra**: provide `infra/bicep/` or `infra/terraform/` templates and `infra/README_CREATE.md`.
* Key env vars (example `.env.example`):

  ```
  DJ I_APP_ID=
  DJI_APP_KEY=
  DJI_APP_LICENSE=
  AZURE_STORAGE_ACCOUNT=
  AZURE_SEARCH_ENDPOINT=
  AZURE_SEARCH_KEY=
  AZURE_OPENAI_ENDPOINT=
  AZURE_OPENAI_KEY=
  KEYVAULT_NAME=
  USE_KEYVAULT=false
  ```
* `start.sh` must: build frontend → copy to backend static → start backend on `$PORT`.

---

## 17 — Governance: retraining & model updates

* Retrain detection when mAP degrades by ≥ 5% vs baseline on operational validation sets.
* Retrain smoke temporal model when AUC or F1 drops > 5%.
* Log model performance metrics per deployment. Use these signals for scheduled retraining.

---

## 18 — Prompt templates for automated agents

Include standard templates under `docs/agent_templates/` to instruct Copilot/agent tasks. Each template must require MODEL/DATA/TRAIN/EVAL blocks in created code.

---

## 19 — Required artifacts for any new feature PR

* `PROPOSAL.md` in feature folder.
* Implemented code + tests.
* Updated docs (module README, `docs/ARCHITECTURE.md`).
* Demo instructions and smoke-run outputs.
* `dev/logs/<task>-report.txt`.

---

## 20 — Live analysis & screen-share special instructions (summary)

* Live flows must generate same shape of artifacts as file/session ingestion so RAG & LLM workflows can consume them.
* For each live chunk index doc include metadata:

  * `doc_type=screen_session_chunk`, `source`, `source_platform`, `stream_offset_ms`, `session_type`, `max_speed`, `avg_accel`, `dominant_player_ids`, `privacy_flags`, `source_blob_url`, `user_acknowledges_broadcast_rights`.
* UI must have safety toggles & broadcast rights acknowledgment.

---

## 21 — Final behavior expectations for the agent (how to act)

* Always add MODEL/DATA/TRAIN/EVAL blocks with any AI-related changes.
* Prefer incremental PRs (one feature per PR).
* If GPU runs are impossible in this environment, produce fully runnable scripts with a `--dry-run` mode and clear instructions for executing on GPU infra.
* When in doubt, create `PROPOSAL.md` and proceed.

---

## Appendix — Quick starter checklists (copy-paste)

### New model folder checklist

```
mkdir -p models/my_model
touch models/my_model/README.md  # include MODEL/DATA/TRAIN/EVAL blocks
touch models/my_model/train_my_model.py
touch models/my_model/test_my_model.py
add requirements to requirements.txt
add minimal sample to tests/data/my_model/
update CI with smoke job for this model
```

### New live ingestion route minimal example (FastAPI pseudo)

```py
@bp.post("/api/live/session/start")
async def start_session(payload: StartSessionPayload):
    # 1) validate & short-circuit
    # 2) create session record (db)
    # 3) allocate webrtc/ws token (KeyVault/TURN)
    # 4) return session_id, ws_url, token
```

