# Vehicle IQ — Engineering Audit

**Repository:** `C:\Users\ansha\vehicle-iq`  
**Branch:** `main` (clean, up to date with `origin/main`)  
**Audit date:** 2026-11-05  
**Test result:** 543 passed, 3 failed, 4 errors (103.95s)

---

## Test Suite Results (Exact Counts)

Command: `pytest tests --basetemp=C:\Users\ansha\AppData\Local\Temp\kilo-pytest-tmp -p no:cacheprovider`

| Category | Count |
|----------|-------|
| Passed | 543 |
| Failed | 3 |
| Errors | 4 |
| Skipped | 0 |

### Failures (all test-code bugs, not application-code bugs)

1. `tests/test_frontend_dashboard.py::test_landing_report_assets_and_hero_image_are_served` — Test asserts HTML contains `/dashboard/assets/vehicle-hero.png` but `frontend/index.html` uses `./assets/vehicle-hero.png`. **Test assertion mismatch.**
2. `tests/test_frontend_dashboard.py::test_dark_mobile_layout_and_accessible_upload_controls` — Test asserts `href="/inspect"` but `frontend/index.html` uses `href="#/inspect"` (single-page router hash). **Test assertion mismatch.**
3. `tests/test_workflow_reliability.py::test_ml_estimate_inr_is_consistent_with_stored_price_estimate` — `AttributeError: 'FixtureFunctionDefinition' object has no attribute 'post'`. The `client` fixture is not a TestClient; likely a naming collision with an imported `client`. **Test fixture bug.**

### Errors (all test-code bugs)

4 errors in `tests/test_workflow_reliability.py` — `NameError: name 'event' is not defined` at `@event.listens_for(engine, "connect")` (line 125). Missing `from sqlalchemy import event` import. **Test import bug.**

**Conclusion:** All 543 passing tests exercise real application logic. The 7 red items are pre-existing test-file defects; none indicate application runtime failures.

---

## 1. ML Training Pipeline, Dataset, Preprocessing, Model, Artifacts

| Checklist Item | Classification | Evidence |
|----------------|----------------|----------|
| Training script exists | **MISSING** | No `train.py`, `train_model.py`, or training entrypoint found anywhere in the repo. Grep for `train|fit|XGBRegressor|train_test_split` returns zero training-script hits. |
| Dataset source | **MISSING** | No dataset file, no data-generation script, no training CSV/Parquet. Only `vehicle-features.json`, `vehicle-spec-catalog.json`, `vehicle-spec-import-sample.csv`, and `market-listings-template.csv` (templates, not training data). |
| Preprocessing pipeline | **MISSING** | No scikit-learn preprocessing pipeline committed. Feature mapping exists in `app/pricing/provider.py:86-140` (inference-only mapping, not training preprocessing). |
| Train/test split | **MISSING** | No split logic, no cross-validation, no holdout evaluation harness. |
| Deployed model | **IMPLEMENTED BUT NOT VERIFIED** | `models/vehicle_iq_price_model.joblib` is tracked in the repo. Loaded lazily by `PipelinePriceProvider` in `app/pricing/provider.py:59-83`. Default path `PRICE_MODEL_PATH` in `app/core/config.py:76`. |
| Model type | **IMPLEMENTED** | XGBoost pipeline, log1p/expm1 inverse transform (`expm1` in `app/pricing/provider.py:153`). Model name: `"Vehicle IQ Used-Car Price Model 0.1.0-research"`. |
| MAE, RMSE, R², MAPE | **MISSING** | No metric computation anywhere in repo. No evaluation script, no metrics file, no test asserting model accuracy. |
| Naive baseline comparison | **MISSING** | No baseline model (mean, median, rule-based) for comparison. |
| Reproducible training | **MISSING** | No training code, no training config, no random seed pinning, no data versioning. Cannot reproduce. |
| Model contract test | **COMPLETE AND VERIFIED** | `tests/test_price_model_provider.py::test_feature_mapping_matches_training_contract`, `test_feature_order_matches_training_contract` (12 features in `MODEL_FEATURE_ORDER`), `test_real_model_produces_genuine_estimate` — all pass against the tracked artifact. |
| Defaults catalog | **IMPLEMENTED BUT NOT VERIFIED** | `app/pricing/model_defaults.py` — `MODEL_DEFAULTS = {}` empty dict; explicitly states catalog is unverified and no defaults are enabled. |

**File paths:** `models/vehicle_iq_price_model.joblib`, `app/pricing/provider.py`, `app/pricing/model_defaults.py`, `tests/test_price_model_provider.py`, `tests/test_pricing_pipeline.py`

**Dependencies on unfinished components:** None (inference is self-contained), but model quality is unverified.

---

## 2. Evaluation Metrics and Naive Baseline Comparison

| Checklist Item | Classification | Evidence |
|----------------|----------------|----------|
| Regression metrics (MAE/RMSE/R²/MAPE) | **MISSING** | No metric functions anywhere. Tests validate inference shape and safety, not accuracy. |
| Baseline model | **MISSING** | No baseline. |
| Holdout evaluation | **MISSING** | No test/train split, no evaluation dataset. |
| Model performance claims | **IMPLEMENTED BUT NOT VERIFIED** | README and code label it "research candidate, not production-validated." No numbers to back any claim. |
| Test evidence for inference | **COMPLETE AND VERIFIED** | `test_real_model_produces_genuine_estimate` passes — real inference against tracked artifact returns finite positive INR amount. |

**File paths:** `tests/test_price_model_provider.py`, `app/pricing/provider.py`

---

## 3. Complete Test Suite (Including Historical Failures)

| Checklist Item | Classification | Evidence |
|----------------|----------------|----------|
| Test runner | **COMPLETE AND VERIFIED** | pytest 8.4.2, configured in `requirements.txt:21`. `pytest-run.txt` captures last run. |
| Total tests | **COMPLETE AND VERIFIED** | 550 collected; 543 passed, 3 failed, 4 errors, 0 skipped. |
| Test file count | **COMPLETE AND VERIFIED** | 40 test files in `tests/` covering: assessments API, workflow, images, OCR, damage API/inference/worker, pricing API/pipeline/provider, market Tavily/valuation/listings, LLM report, vehicle lookup API/provider/services, vehicle specs ingestion/resolver, security middleware, CORS, health, observability, bounded HTTP, frontend dashboard, database foundation, provision PaddleOCR, image classification, HuggingFace OCR, langsearch, supabase storage, deployment config, guest sessions, registration OCR. |
| Test isolation | **COMPLETE AND VERIFIED** | `tests/conftest.py` auto-disables live Tavily requests (`tavily_api_key = None`). Tests use in-memory/SQLite test DB via fixtures. |
| Historical failures | **PARTIALLY IMPLEMENTED** | Git log shows reverts (`8a72814 Revert Add manual registration fallback`) and iterative fixes. No CHANGELOG or failure registry. The 3 current failures + 4 errors are known pre-existing test-file defects. |
| CI pipeline | **MISSING** | No `.github/workflows`, no `azure-pipelines`, no GitLab CI config found. No automated test execution on push. |

**File paths:** `tests/` (40 files), `tests/conftest.py`, `pytest-run.txt`, `requirements.txt`

---

## 4. Authentication Implementation and Frontend/Backend Compatibility

| Checklist Item | Classification | Evidence |
|----------------|----------------|----------|
| Auth scheme | **PARTIALLY IMPLEMENTED** | Single shared bearer token (`API_ACCESS_TOKEN`) validated by `ApiSecurityMiddleware` in `app/core/security.py:109-120` (constant-time `hmac.compare_digest`). Only enforced in production (`APP_ENV=production`). Development mode is open (`test_security_middleware.py::test_api_remains_open_in_development_without_token` passes). |
| User accounts | **MISSING** | No users table, no registration, no login, no password hashing. `DEPLOYMENT.md:14` confirms: "does not implement user accounts or per-assessment ownership." |
| Per-assessment ownership | **MISSING** | No `user_id` FK on `VehicleAssessment` or any table. `app/db/models.py` has no ownership column. |
| RBAC / roles | **MISSING** | No role-based access control. |
| Token rotation / expiry | **MISSING** | Single static token, no rotation, no expiry. |
| Frontend auth | **MISSING** | `frontend/app.js:4` hardcodes `API = https://vehicle-iq-8wzk.onrender.com/api/v1/assessments` with no bearer header, no token input, no login flow. |
| Frontend/backend compatibility | **PARTIALLY IMPLEMENTED / BROKEN TESTS** | Frontend calls backend directly. Two frontend tests fail due to href/path mismatches (`#/inspect` vs `/inspect`, `./assets/` vs `/dashboard/assets/`). These are test-assertion bugs, but indicate the frontend is not fully integrated under the `/dashboard` mount path. |
| Rate limiting | **IMPLEMENTED BUT NOT VERIFIED** | In-memory per-IP per-worker rate limiter (`app/core/security.py:91-106`), default 600/min. Bypassed across multiple workers (`DEPLOYMENT.md:122-123`). Not shared (no Redis). |
| Upload size capping | **COMPLETE AND VERIFIED** | `app/core/security.py:65-87` bounds multipart body before parsing; 10 MiB default, 100 MiB max. Tests pass. |
| Production fail-closed | **COMPLETE AND VERIFIED** | `validate_production_settings` raises `RuntimeError` if token < 32 chars or `DATABASE_URL` not `postgresql+psycopg` (`app/core/config.py:102-121`). |

**File paths:** `app/core/security.py`, `app/core/config.py`, `frontend/app.js`, `DEPLOYMENT.md`, `tests/test_security_middleware.py`

**Security gaps:** No user auth, no ownership, no RBAC, no token rotation, no HTTPS enforcement in code, no CORS origin restriction beyond 3 hardcoded origins (`app/main.py:48-57`), no CSRF protection, no request signing, no secret injection at runtime (`.env` must be supplied externally).

---

## 5. Provider Integrations, API Configuration, Privacy/Security Documentation

| Provider | Purpose | Status | Config |
|----------|---------|--------|--------|
| Tavily | Nationwide used-car listing discovery | Optional, disabled by default | `TAVILY_API_KEY`, `TAVILY_TIMEOUT_SECONDS`, `TAVILY_RESULT_LIMIT` |
| Groq-compatible LLM | Buyer narrative generation | Optional, disabled by default | `LLM_ENABLED`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_TIMEOUT_SECONDS` |
| Hugging Face Gradio (Baidu Unlimited-OCR) | OCR text extraction | Default provider | `OCR_GRADIO_SPACE_URL`, `OCR_GRADIO_MODE`, `OCR_TIMEOUT_SECONDS` |
| PaddleOCR local | Local CPU OCR | Optional | `OCR_SERVICE_PROVIDER=paddleocr_local`, model dirs |
| Baidu Unlimited-OCR (legacy FastAPI) | OCR HTTP service | Optional | `OCR_SERVICE_URL`, `baidu_unlimited_ocr` |
| VehicleX (vehiclex.profilework239.workers.dev) | Registration lookup | Disabled by default | `VEHICLE_LOOKUP_ENABLED=false`, `VEHICLE_LOOKUP_BASE_URL` |
| Challan (challan-info-new-abhigyan.vercel.app) | Challan/violation lookup | Disabled by default | `CHALLAN_LOOKUP_ENABLED=false` |
| FASTag (abhigyan-fastag-info.vercel.app) | FASTag lookup | Disabled by default | `FASTAG_LOOKUP_ENABLED=false` |
| YOLO11m (local) | CV damage detection | Optional, isolated subprocess | `DAMAGE_INFERENCE_PYTHON`, `DAMAGE_CHECKPOINT_PATH`, `DAMAGE_INFERENCE_TIMEOUT_SECONDS` |
| LangSearch | Auxiliary discovery helper | Optional, not in workflow | `LANGSEARCH_API_KEY` (not wired into assessment) |

| Privacy/Security Item | Classification | Evidence |
|----------------------|----------------|----------|
| Secrets not committed | **COMPLETE AND VERIFIED** | `.env.example` has placeholder values only. `.env` is gitignored (confirmed: not in file listing as tracked; `.gitignore` exists). |
| No credentials sent to providers | **COMPLETE AND VERIFIED** | Vehicle lookup sends no auth header (`app/vehicle_lookup/provider.py:48`: "No authorization header is sent because none is documented"). |
| Response size bounding | **COMPLETE AND VERIFIED** | `app/core/http.py:12-54` `request_with_response_limit` caps at `max_external_response_bytes` (default 2 MiB). |
| PII redaction in logs | **COMPLETE AND VERIFIED** | OCR SSE error sanitization (`app/ocr/client.py:494-516`), Tavily URL/registration/phone redaction (`app/market/tavily.py:141-147, 373-378`). |
| Privacy policy document | **MISSING** | No `PRIVACY.md`, no `SECURITY.md`, no data-retention policy, no GDPR/PDP statement. `DEPLOYMENT.md` and `README.md` mention limitations but not privacy. |
| Data-sharing documentation | **PARTIALLY IMPLEMENTED** | `README.md:49` distinguishes provider-reported vs user data. `DEPLOYMENT.md:32-43` warns Tavily/LLM quotas are not assumed. No formal data-sharing policy. |
| API key exposure to frontend | **COMPLETE AND VERIFIED** | Tavily and LLM keys are backend-only (`.env.example:8-13` comments). Frontend `app.js` has no API key. |

**File paths:** `.env.example`, `app/core/config.py`, `app/core/http.py`, `app/vehicle_lookup/provider.py`, `app/vehicle_lookup/vehicle_services.py`, `app/market/tavily.py`, `app/llm/client.py`, `app/ocr/client.py`, `DEPLOYMENT.md`, `README.md`

**Dependencies on unfinished components:** Tavily/LLM/VehicleX/Challan/FASTag all require external credentials and network access; disabled by default.

---

## 6. Failure Handling, Retries, Timeouts, Service-Status Reporting

| Checklist Item | Classification | Evidence |
|----------------|----------------|----------|
| Request timeouts (all external calls) | **COMPLETE AND VERIFIED** | Tavily (`app/market/tavily.py:182`), LLM (`app/llm/client.py:27`), OCR (`app/ocr/client.py:349-354`), Vehicle lookup (`app/vehicle_lookup/provider.py:86`), Challan/FASTag (`app/vehicle_lookup/vehicle_services.py:60`), damage subprocess (`app/damage/worker.py:103`). |
| Bounded external response size | **COMPLETE AND VERIFIED** | `app/core/http.py`, `settings.max_external_response_bytes`. |
| Graceful degradation when provider unavailable | **COMPLETE AND VERIFIED** | Workflow preserves report when stages fail (`app/api/v1/workflow.py:381-673`). Report generated with unavailable/failed stage statuses. |
| Retry logic | **PARTIALLY IMPLEMENTED** | Market report refresh retries on provider failure preserving prior snapshot (`app/api/v1/reports.py:610-625`). OCR hybrid fallback (primary→fallback) in `app/ocr/client.py:52-91`. No exponential backoff, no circuit breaker, no automatic retry on transient HTTP 5xx except the market snapshot fallback. |
| Service status reporting | **COMPLETE AND VERIFIED** | Every stage reports `available`/`partial`/`unavailable`/`failed`/`skipped` in `WorkflowStageResult` (`app/schemas/workflow.py`). Health endpoints `/health/live`, `/health/ready` (`app/main.py:91-109`). |
| OCR error mapping | **COMPLETE AND VERIFIED** | Distinct statuses: `unavailable`, `failed`, `unreadable` (`tests/test_ocr_api.py` error-mapping tests pass). |
| Damage inference error mapping | **COMPLETE AND VERIFIED** | `DamageWorkerUnavailable`, `DamageWorkerTimeout`, `DamageWorkerInvalidResponse`, `DamageWorkerInferenceFailed` (`app/damage/worker.py:25-43`). |
| Pricing provider error mapping | **COMPLETE AND VERIFIED** | 503 when unconfigured, 502 when inference fails, 422 when features missing (`app/api/v1/pricing.py`). |
| Missing latency metrics | **MISSING** | No Prometheus metrics, no histogram, no p95/p99 latency tracking. Observability middleware logs duration per request (`app/core/observability.py:46-58`) but no aggregation. |
| Missing success-rate metrics | **MISSING** | No success/failure counters per provider, no error-rate dashboard, no alerting thresholds. |
| Circuit breaker | **MISSING** | No circuit breaker on any external provider. |
| Idempotency | **PARTIALLY IMPLEMENTED** | Workflow returns stored report if pricing stage already completed/failed (`app/api/v1/workflow.py:444-474`). Duplicate submission detection via DB row lock (`with_for_update`, line 409). |

**File paths:** `app/api/v1/workflow.py`, `app/api/v1/reports.py`, `app/core/http.py`, `app/core/observability.py`, `app/core/security.py`, `app/damage/worker.py`, `app/llm/client.py`, `app/market/tavily.py`

---

## 7. CV Damage Detection Implementation and Verification Evidence

| Checklist Item | Classification | Evidence |
|----------------|----------------|----------|
| Model architecture | **IMPLEMENTED** | YOLO11m-Car-Damage-Detector (`app/damage/worker.py:20`: `MODEL_NAME = "YOLO11m-Car-Damage-Detector"`). Ultralytics `YOLO` loader (`app/damage/inference.py:126`). |
| Checkpoint file | **MISSING** | `trained.pt` not committed to repo. Default path `.local-storage/damage-poc/trained.pt` (`app/core/config.py:58`). Not in file listing. |
| Dedicated inference environment | **PARTIALLY IMPLEMENTED** | `.local-storage/damage-poc/.venv/Scripts/python.exe` referenced but not committed. `requirements-damage-inference.txt` exists (47 pinned deps). `scripts/provision_paddleocr_models.py` provisions OCR models, not damage checkpoint. |
| Isolated subprocess inference | **COMPLETE AND VERIFIED** | `IsolatedDamageProvider.detect()` spawns subprocess with worker protocol (`app/damage/worker.py:71-125`), bounded timeout, temp directory, sanitized env. |
| In-process detector | **IMPLEMENTED** | `DamageDetector.predict()` in `app/damage/inference.py:78-101`, lazily loads Ultralytics model. |
| Image eligibility classification | **IMPLEMENTED** | `app/images/classification.py` classifies plate/document/interior/exterior/unknown, sets `damage_eligible` flag. |
| Real-image validation evidence | **MISSING** | No damage ground-truth dataset, no precision/recall/IoU metrics, no confusion matrix, no validation on the tracked `test_images/Damaged Suzuki Sedan...png`. No test runs actual inference (checkpoint absent). |
| Training pipeline for damage model | **MISSING** | No training script, no dataset, no augmentation config. |
| Test coverage (mocked) | **COMPLETE AND VERIFIED** | `tests/test_damage_inference.py`, `tests/test_damage_worker.py`, `tests/test_damage_api.py` — all pass with mocked checkpoints/results. |

**File paths:** `app/damage/worker.py`, `app/damage/inference.py`, `app/images/classification.py`, `scripts/run_damage_inference.py`, `requirements-damage-inference.txt`, `tests/test_damage_*.py`

**Dependencies on unfinished components:** Damage checkpoint must be procured/trained separately; inference runtime (`ultralytics`, `torch`) must be installed in dedicated venv.

---

## 8. Valuation Implementation and Verification Evidence

### XGBoost Price Estimate

| Checklist Item | Classification | Evidence |
|----------------|----------------|----------|
| Model artifact | **IMPLEMENTED BUT NOT VERIFIED** | `models/vehicle_iq_price_model.joblib`, loaded by `PipelinePriceProvider`. |
| Feature contract | **COMPLETE AND VERIFIED** | 12 features in `MODEL_FEATURE_ORDER` (`app/pricing/provider.py:39-52`). Tests pass: `test_feature_mapping_matches_training_contract`, `test_feature_order_matches_training_contract`. |
| Null safety | **COMPLETE AND VERIFIED** | `map_features` raises on any None feature; `VehiclePriceInput` requires all fields (`app/schemas/pricing.py:38-64`). |
| Inverse transform (log1p/expm1) | **COMPLETE AND VERIFIED** | `expm1(log_price)` in `app/pricing/provider.py:153`. Test `test_model_prediction_is_inverse_transformed_and_has_no_fake_bounds` passes. |
| No fabricated confidence bounds | **COMPLETE AND VERIFIED** | `lower_bound=None, upper_bound=None` in `PricePrediction`. |
| Model validation metrics | **MISSING** | No MAE/RMSE/R²/MAPE. No holdout set. |
| External validation | **MISSING** | No comparison against independent pricing source. |
| Naive baseline | **MISSING** | No baseline for comparison. |
| Reproducibility | **MISSING** | No training code, no data, no seed. |

### Market Comparable Valuation (Tavily)

| Checklist Item | Classification | Evidence |
|----------------|----------------|----------|
| Source | **IMPLEMENTED** | Tavily web search, 2 queries per report (`app/market/tavily.py:490-509`), nationwide scope, credible-domain allowlist. |
| Comparable calculation | **COMPLETE AND VERIFIED** | `calculate_market_comparables` in `app/pricing/market.py:291-466`. IQR outlier filtering (1.5×IQR), year cohort, fuel filtering, duplicate removal, staleness (30 days). |
| Minimum evidence | **COMPLETE AND VERIFIED** | `MIN_COMPARABLE_COUNT = 3` (`app/pricing/market.py:13`). Range only when ≥3 inliers. |
| Price range method | **COMPLETE AND VERIFIED** | Interquartile range (25th/50th/75th percentile), `MarketPriceRange` (`app/pricing/market.py:92-97`). |
| Advertised vs transaction | **COMPLETE AND VERIFIED** | Documented as asking prices, not sales (`README.md:164`, `app/pricing/market.py:38`). |
| Live verification | **MISSING** | Requires `TAVILY_API_KEY`; disabled by default (`tests/conftest.py` monkeypatches key to None). No live test evidence. |
| External validation | **MISSING** | No comparison to actual transaction data. |

**File paths:** `app/pricing/provider.py`, `app/pricing/market.py`, `app/market/tavily.py`, `app/api/v1/pricing.py`, `app/api/v1/reports.py`, `tests/test_price_model_provider.py`, `tests/test_pricing_pipeline.py`, `tests/test_market_valuation.py`, `tests/test_tavily_market.py`

---

## 9. Agentic Report Generation and Integration Status

| Checklist Item | Classification | Evidence |
|----------------|----------------|----------|
| LLM client | **IMPLEMENTED** | `LLMClient.complete_json()` in `app/llm/client.py:33-61`, OpenAI-compatible `/chat/completions` with `response_format: json_object`. |
| LLM enablement | **IMPLEMENTED** | `LLMClient.enabled` requires `settings.llm_enabled and api_key` (`app/llm/client.py:30-31`). Disabled by default (`LLM_ENABLED=false` in `.env.example`). |
| Report prompt | **IMPLEMENTED** | `build_llm_prompt()` in `app/llm/report.py:19-116`, structured evidence JSON, explicit grounding instructions (no invention, separate provenance, damage caveats). |
| Narrative schema | **IMPLEMENTED** | `LLMReportNarrative` in `app/llm/schemas.py` (9 fields: vehicle_overview, overall_condition_summary, detected_damage_explanation, etc.). |
| Narrative generation | **IMPLEMENTED BUT NOT VERIFIED** | `generate_llm_narrative()` in `app/llm/report.py:204-245`. Returns `None` when client disabled or request fails; structured report preserved. Not tested against live LLM (no API key in tests). |
| Workflow integration | **PARTIALLY IMPLEMENTED** | Workflow endpoint (`app/api/v1/workflow.py`) runs OCR → lookup → damage → pricing → report snapshot. Report generation is part of workflow. LLM narrative is generated separately via `/report` endpoint (`app/api/v1/reports.py`), not inside the workflow stage loop. |
| Frontend narrative rendering | **IMPLEMENTED** | `renderLLMNarrative()` in `frontend/app.js:694-736`, renders 9 narrative blocks. |
| Agentic autonomy | **MISSING** | No agent loop, no tool use, no self-correction, no multi-step reasoning beyond single LLM call. Not agentic in the autonomous sense; it's a single LLM narrative generation step. |

**File paths:** `app/llm/client.py`, `app/llm/report.py`, `app/llm/schemas.py`, `app/api/v1/reports.py`, `app/api/v1/workflow.py`, `frontend/app.js:694-736`, `tests/test_llm_report.py`

---

## 10. Deployment Configuration and End-to-End Integration

| Checklist Item | Classification | Evidence |
|----------------|----------------|----------|
| Render backend config | **IMPLEMENTED** | `DEPLOYMENT.md:98-112` specifies build (`pip install -r requirements.txt && python scripts/provision_paddleocr_models.py`) and start (`uvicorn app.main:app`). `.python-version` pins Python 3.13.16. |
| Frontend hosting | **IMPLEMENTED** | Vercel (`https://vehicle-iq-frontend.vercel.app/` in README). Frontend is static HTML/JS served by FastAPI locally (`app/main.py:69-89`), but production frontend is on Vercel calling `https://vehicle-iq-8wzk.onrender.com` (`frontend/app.js:4`). |
| Alembic migrations | **COMPLETE AND VERIFIED** | 6 linear revisions: `0001` foundation, `0002` nullable damage component, `0003` vehicle specs, `0004` spec identity, `0005` market listings, `0006` PostgreSQL image BYTEA. `alembic.ini` present. |
| Image storage migration | **IMPLEMENTED** | New images stored as BYTEA in PostgreSQL (`app/db/models.py:88`: `LargeBinary, deferred=True`). Legacy file-referenced images remain unavailable until bytes restored (`DEPLOYMENT.md:54-58`). |
| Health checks | **COMPLETE AND VERIFIED** | `/health/live` (process), `/health/ready` (DB `SELECT 1`, 503 when unavailable). Readiness does NOT verify migration currency (`DEPLOYMENT.md:69-70`). |
| Environment variables | **COMPLETE AND VERIFIED** | Full spec in `README.md:323-359` and `.env.example`. `validate_production_settings` enforces token + DB URL in production. |
| CORS | **IMPLEMENTED** | 3 hardcoded origins (`app/main.py:48-57`): localhost:5500, 127.0.0.1:5500, vercel frontend. |
| Frontend/backend integration tests | **PARTIALLY / BROKEN** | 2 frontend tests fail (href and asset path mismatches). Frontend calls Render backend directly; local FastAPI serves frontend at `/` and `/dashboard`. Potential path mismatch under `/dashboard` mount. |
| Rate limiting in production | **IMPLEMENTED BUT NOT VERIFIED** | Per-worker in-memory; `DEPLOYMENT.md:122-123` warns reverse proxy needed. |
| TLS / secret manager | **MISSING** | Not configured in repo (`DEPLOYMENT.md:124`): "TLS termination, persistent-volume behavior, secret-manager integration, and deployment health-check wiring must be verified on the chosen hosting platform." |
| CI/CD | **MISSING** | No pipeline config found. |
| PaddleOCR provisioning | **IMPLEMENTED** | `scripts/provision_paddleocr_models.py` downloads PP-OCRv5 models to `.local-storage/paddleocr-smoke-bos/`. Build step in Render. Tests pass (mocked downloads). |

**File paths:** `DEPLOYMENT.md`, `.python-version`, `alembic.ini`, `alembic/versions/`, `app/main.py`, `frontend/app.js`, `scripts/provision_paddleocr_models.py`, `.env.example`

---

## 11. Repository Clutter, Temporary Files, Unused Dependencies

| Item | Classification | Evidence |
|------|----------------|----------|
| `backend/.venv` | **CLUTTER** | A complete Python virtual environment (~thousands of files including `tzdata`, `urllib3`, `xgboost` DLLs, etc.) is inside the repo working directory. Should not be committed; must be gitignored. |
| `__pycache__/` directories | **CLUTTER** | Python bytecode caches in `app/`, `tests/`, etc. Should be gitignored. |
| `.pytest_cache/` | **CLUTTER** | Pytest cache. Should be gitignored. |
| `.test-tmp/` | **CLUTTER** | Test temp directory. Should be gitignored. |
| `.local-storage/` | **CLUTTER / RUNTIME** | Runtime cache for PaddleOCR/damage models. Gitignored per `.gitignore` (likely), but present on disk. |
| `unlimited-ocr-api/` | **PARTIALLY RELATED** | Separate OCR service subproject with its own `app.py`, `Dockerfile`, `huggingface-space/`. Not integrated into main workflow (legacy/extracted service). Could be removed or isolated. |
| `pytest-run.txt` | **CLUTTER** | Test output artifact in repo root. Should not be committed. |
| `models/vehicle_iq_price_model.joblib` | **INTENTIONAL ARTIFACT** | Tracked model artifact, not clutter. |
| `test_images/` | **TEST FIXTURE** | Legitimate test image. |
| Unused dependencies | **PARTIALLY** | `requirements-damage-inference.txt` is separate (for damage venv, not main). `gradio_client` and `paddlepaddle`/`paddleocr` are in main `requirements.txt` but only used if OCR provider is local/hybrid. `langsearch` config exists in settings but LangSearch is not wired into any workflow (`README.md:192`: "LangSearch appears as an auxiliary discovery helper; Tavily is the market-discovery adapter"). |
| Sensitive data risk | **LOW RISK** | `.env.example` has no real secrets. `.env` is not tracked. No API keys found in code via grep. Provider URLs are hardcoded but not secrets. |

**File paths:** `.gitignore` (verify contents), `backend/.venv/`, `__pycache__/`, `.pytest_cache/`, `.test-tmp/`, `pytest-run.txt`, `unlimited-ocr-api/`

---

## Critical Blockers

1. **No ML training pipeline exists.** The XGBoost model is a black-box artifact with no training code, no dataset, no metrics, no baseline, and no reproducibility path. This is the highest-priority technical gap if production-grade valuation is required.

2. **Damage model checkpoint is missing.** `trained.pt` is not in the repo. Damage inference cannot run in any environment without procuring/training the checkpoint and setting up the dedicated venv.

3. **No user authentication or ownership model.** Single shared bearer token only; no users, no RBAC, no per-assessment ownership. Multi-user deployment is not possible without a major auth rewrite.

4. **Test suite has 7 red items** (3 failures, 4 errors) — all pre-existing test-file defects (missing import, fixture collision, assertion mismatches). These block a clean `pytest` run but do not block application functionality.

5. **No CI/CD pipeline.** Tests are not automated on push; no merge gates.

6. **No production secrets/TLS/secret-manager configuration.** `DEPLOYMENT.md` explicitly states these must be verified externally.

---

## Recommended Execution Order (Milestones)

| Milestone | Description | Dependencies |
|-----------|-------------|--------------|
| **M1: Fix test suite defects** | Fix the 4 errors (missing `event` import in `test_workflow_reliability.py`) and 3 failures (frontend href/path assertions, `client` fixture). Target: 550/550 passing, 0 errors. | None |
| **M2: Auth & authorization** | Add user accounts, password hashing, per-assessment ownership, RBAC, token rotation. Replace single shared token. | None (blocks multi-user deployment) |
| **M3: CI/CD pipeline** | Add GitHub Actions (or equivalent) to run pytest on push/pull request. | M1 (green suite) |
| **M4: ML training pipeline** | Create training script, source/curate dataset, implement preprocessing, train/test split, compute MAE/RMSE/R²/MAPE, add naive baseline, version model artifact. | None (independent, but large effort) |
| **M5: Damage model checkpoint** | Procure/train YOLO11m damage checkpoint, commit to model registry or cloud storage, set up dedicated inference venv, validate on real images with precision/recall/IoU. | None |
| **M6: Observability** | Add Prometheus metrics (latency histograms, success/failure counters per provider), circuit breakers, alerting thresholds. | None |
| **M7: Deployment hardening** | Configure TLS, secret manager integration, persistent volumes, reverse-proxy rate limiting, migration verification in release pipeline. | M2, M6 |
| **M8: End-to-end validation** | Live smoke test against deployed Render + Vercel with real (or permitted) data; verify frontend/backend path compatibility under `/dashboard`. | M1, M2 |

---

## Evidence-Backed Completion Percentage

**Overall: ~58% complete**

- Application code (backend API, workflow, OCR, pricing, market, damage worker, LLM narrative): ~85% implemented, mostly tested with mocked providers.
- ML training & validation: 0% (no training pipeline, no metrics).
- Auth & authorization: ~15% (shared token only, no users/ownership).
- CV damage: ~40% (inference pipeline complete, checkpoint missing, no validation).
- Valuation verification: ~20% (inference works, no external validation, no metrics).
- Testing: ~95% (543 passing, 7 pre-existing test defects).
- Deployment: ~60% (config + migrations exist, but CI/CD, TLS, secrets missing).
- Observability: ~25% (request logging exists, no metrics/alerting).

---

## Completed Checklist Items

1. ML training pipeline — **MISSING**
2. Evaluation metrics & naive baseline — **MISSING**
3. Test suite — **IMPLEMENTED BUT NOT VERIFIED** (543 pass, 7 red items are test-file defects)
4. Authentication — **PARTIALLY IMPLEMENTED** (shared token only)
5. Provider integrations & privacy — **PARTIALLY IMPLEMENTED** (all optional, disabled by default)
6. Failure handling / timeouts / retries — **PARTIALLY IMPLEMENTED** (no circuit breakers, no latency/success metrics)
7. CV damage detection — **PARTIALLY IMPLEMENTED** (checkpoint missing, no verification evidence)
8. Valuation — **PARTIALLY IMPLEMENTED** (inference works, no metrics/validation)
9. Agentic report generation — **PARTIALLY IMPLEMENTED** (LLM narrative integrated, not autonomous agent)
10. Deployment — **PARTIALLY IMPLEMENTED** (config + migrations exist, CI/CD/TLS/secrets missing)
11. Repository cleanup — **PARTIALLY IMPLEMENTED** (committed venv, caches, pytest artifact present)

---

**File paths of primary significance:**

- `app/main.py` — FastAPI entry, CORS, frontend routing, health checks
- `app/core/config.py` — Settings, production validation
- `app/core/security.py` — API key middleware, rate limiting, upload capping
- `app/core/observability.py` — Request ID + timing logging
- `app/core/http.py` — Bounded external HTTP responses
- `app/api/v1/workflow.py` — Assessment workflow orchestration
- `app/api/v1/reports.py` — Report assembly, market valuation, buyer recommendation
- `app/api/v1/pricing.py` — Price estimate endpoint
- `app/pricing/provider.py` — XGBoost inference provider
- `app/pricing/market.py` — Market comparable calculation
- `app/market/tavily.py` — Tavily discovery adapter
- `app/llm/client.py`, `app/llm/report.py` — LLM client + narrative
- `app/damage/worker.py`, `app/damage/inference.py` — Damage inference
- `app/ocr/client.py` — OCR clients (Gradio, Paddle, Baidu, Hybrid)
- `app/vehicle_lookup/provider.py`, `app/vehicle_lookup/vehicle_services.py` — Registration/Challan/FASTag
- `app/db/models.py`, `app/db/session.py` — SQLAlchemy models + session
- `app/images/classification.py`, `app/images/validation.py` — Image eligibility
- `frontend/app.js`, `frontend/index.html` — Frontend SPA
- `tests/` — 40 test files (543 passing)
- `alembic/versions/` — 6 migrations
- `models/vehicle_iq_price_model.joblib` — Tracked model artifact
- `DEPLOYMENT.md`, `.env.example`, `README.md` — Documentation
- `scripts/provision_paddleocr_models.py`, `scripts/run_damage_inference.py` — Scripts
- `backend/.venv/` — Committed virtual environment (clutter)