# Vehicle IQ free-tier deployment preparation

Target layout: static `frontend/` on Vercel, FastAPI from the repository root on
Render, Supabase Free PostgreSQL, and a private Supabase Storage bucket. This is
deployment configuration documentation, not evidence that any account, service,
database, domain, model artifact, or provider has been provisioned or tested.

The Render backend runtime is pinned to Python `3.12.16` in `.python-version`.
Keep PaddleOCR/PaddlePaddle out of the root production install; those optional
local-only packages are listed in `requirements-local-ocr.txt`.
If the Render service has a `PYTHON_VERSION` environment variable set, make it
`3.12.16` or remove it; Render gives that setting precedence over this file.

## Before creating services

The browser uses revocable guest sessions for assessment endpoints. It stores
the opaque bearer credential in browser `localStorage` and sends it only in the
HTTPS `Authorization` header. This avoids exposing the internal
`API_ACCESS_TOKEN`; the trade-off is that an XSS vulnerability could steal the
guest credential. A guest who clears browser storage cannot recover that
session's assessments. Guest sessions expire after `GUEST_SESSION_TTL_DAYS` and
can be revoked. Existing records with no guest owner remain accessible only to
internal callers.

The model artifact is not included in the repository's normal deployment
inputs. Set `PRICE_MODEL_PATH` to an artifact that has been deliberately added
to the backend deployment, or leave pricing unavailable. The configured model
remains research-only and is not a verified market valuation. The YOLO worker
and checkpoint are local-only assets and are disabled on Render Free.

## 1. Supabase Free PostgreSQL and Storage

1. Create a Supabase project on the Free plan if the current signup and account
   eligibility permit it without payment or billing activation. Do not add a
   billing method for this target architecture.
2. In the dashboard's **Connect** dialog, choose a Postgres connection string.
   Supabase documents direct connections for persistent backends and the shared
   session pooler as the alternative for IPv4-only networks. Copy the connection
   details from the dashboard rather than composing a host or username. Convert
   the URL driver to `postgresql+psycopg://` and percent-encode reserved password
   characters. Use TLS (`sslmode=require`). The deployment's actual network
   route must be checked before choosing direct versus session-pooler mode.
3. In **Storage**, create a private bucket named `vehicle-images` (or set
   `SUPABASE_STORAGE_BUCKET` to the chosen name). Set its maximum object size to
   no more than `MAX_IMAGE_UPLOAD_BYTES` and restrict MIME types to JPEG, PNG,
   and WebP. The app does not create buckets automatically.
4. Copy the project URL and server-side Storage key into Render environment
   variables only. This adapter uses the Supabase Storage REST object endpoint
   with server-side `apikey` and bearer headers. Never put the service-role or
   secret key in Vercel or the browser. It bypasses Supabase row-level security;
   FastAPI remains responsible for object access checks. Supabase is changing
   legacy API key naming during 2026; check its current key guidance when
   provisioning and keep whichever privileged key is used server-side.

### Controlled database migration

No migration runs during application startup. First inspect history:

```powershell
.\backend\.venv\Scripts\python.exe -m alembic heads
.\backend\.venv\Scripts\python.exe -m alembic history
```

Guest ownership is introduced by additive revision `20261102_0006`:
`guest_sessions` is created and nullable `vehicle_assessments.guest_session_id`
is added with a foreign key and index. Existing assessments remain unchanged
with a null owner, so guest sessions cannot claim legacy records. Include this
revision when applying `alembic upgrade head` to a newly provisioned database;
do not run it automatically at application startup.

For a newly created Supabase database, set `DATABASE_URL` in a secure local
process environment (do not place it in source control or paste it into chat),
then run the one-time controlled migration:

```powershell
.\backend\.venv\Scripts\python.exe -m alembic upgrade head
```

Verify the resulting revision with `alembic current` before starting the API.
Do not run `upgrade head` against an existing database until its current schema
and revision have been checked and backed up. The application readiness endpoint
checks `SELECT 1`; it does not check migration currency.

## 2. Render Free FastAPI service

Create a **Web Service** connected to this repository with root directory `.`.
The root `.python-version` pins the runtime to Python `3.12.16`; remove or
update any Render `PYTHON_VERSION` override so it does not select a different
interpreter. Configure:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health/ready`
- Set all production variables below in Render's environment settings.
- Do not run migrations as a build/start command. Apply them as the controlled
  step above before directing traffic.

Render Free web services have an ephemeral filesystem, suspend after 15 minutes
without inbound activity, and may need about a minute to wake. Render documents
that Free services cannot attach persistent disks. This is why production
configuration rejects local image storage. Supabase Storage is the durable
object store; the Render local filesystem must not be used for uploaded images.
Free instance hours and bandwidth/build limits apply; Render documents that
services may be suspended when free usage is exhausted if no payment method is
present. Do not treat this tier as production availability or an SLA.

## 3. Backend environment variables

Required for production startup:

| Variable | Meaning |
| --- | --- |
| `APP_ENV=production` | Enables strict startup validation and production security behavior. |
| `DATABASE_URL` | Supabase SQLAlchemy URL using `postgresql+psycopg://`; encode password URL characters and set `sslmode=require` or stronger. |
| `API_ACCESS_TOKEN` | Random secret of at least 32 characters for internal callers. Keep server-side only; the static frontend uses guest sessions instead. |
| `GUEST_SESSION_TTL_DAYS` | Guest credential lifetime (default 30 days; allowed 1–365). Clearing browser storage loses access to that guest's history. |
| `GUEST_SESSION_CREATE_LIMIT_PER_HOUR` | Per-IP limit for guest session creation and renewal (default 10). |
| `GUEST_EXPENSIVE_LIMIT_PER_HOUR` | Per-guest limit for inference, lookup, pricing and workflow requests (default 30). |
| `GUEST_EXPENSIVE_IP_LIMIT_PER_HOUR` | Additional per-IP limit for expensive guest requests (default 120). |
| `IMAGE_STORAGE_PROVIDER=supabase` | Prevents ephemeral local disk use. |
| `SUPABASE_URL` | Project HTTPS URL. |
| `SUPABASE_STORAGE_BUCKET` | Existing private bucket name, normally `vehicle-images`. |
| `SUPABASE_SERVICE_ROLE_KEY` | Privileged server-only storage key. Never expose it to Vercel. |
| `SUPABASE_STORAGE_TIMEOUT_SECONDS` | Bounded request timeout for private image operations (default 30 seconds). |
| `CORS_ALLOWED_ORIGINS` | Comma-separated exact HTTPS origin(s), for example the Vercel production origin and any explicitly used preview origin. No paths or trailing slash. |
| `OCR_SERVICE_PROVIDER` | Select a remote configured provider: `huggingface_gradio` or `baidu_unlimited_ocr`. `paddleocr_local` is rejected in production. |
| `OCR_GRADIO_SPACE_URL` | HTTPS Space base URL when using `huggingface_gradio`; public service availability is not guaranteed. |
| `OCR_SERVICE_URL` | Required HTTPS endpoint when `OCR_SERVICE_PROVIDER=baidu_unlimited_ocr`. |
| `OCR_TIMEOUT_SECONDS` | Bounded remote OCR request timeout (default 180 seconds). |
| `DAMAGE_INFERENCE_ENABLED=false` | Set false on Render Free or any deployment without the isolated CPU runtime and checkpoint. Disabled scans report `unavailable` with `inference_disabled`; no worker process is started. In development, omission defaults to enabled. |

For `baidu_unlimited_ocr`, set `OCR_SERVICE_URL` to its authorized HTTPS service
instead of the Gradio URL. Never configure an unencrypted production endpoint.
The public Hugging Face Space and any Baidu service must be independently
checked for access, terms, uptime, rate limits, and image-handling requirements.

Optional capacity/model settings:

- `MAX_IMAGE_UPLOAD_BYTES` (default 10 MiB), `MAX_EXTERNAL_RESPONSE_BYTES`
  (default 2 MiB), `SUPABASE_STORAGE_TIMEOUT_SECONDS` (default 30),
  `OCR_TIMEOUT_SECONDS` (default 180), and
  `API_RATE_LIMIT_PER_MINUTE` (default 600 per IP per process).
- `PRICE_MODEL_PATH` (default `artifacts/vehicle_iq_price_model.joblib`). The
  model is loaded lazily. If the file is absent or invalid, pricing reports a
  safe provider failure; do not create placeholder model output.
- `LLM_ENABLED` and `LLM_API_KEY` are optional. LLM narrative remains off unless
  explicitly configured. `TAVILY_API_KEY` is optional for market discovery;
  no key means market valuation remains unavailable. Keep vehicle details,
  Challan, and FASTag flags false unless their use is authorized and verified.
- `DAMAGE_INFERENCE_PYTHON` and `DAMAGE_CHECKPOINT_PATH` configure local
  inference. The default executable path is platform-specific (`Scripts/python.exe`
  on Windows, `bin/python` elsewhere). They are not used when inference is
  disabled. An enabled deployment must provide the isolated runtime, runner, and
  checkpoint; model presence on the developer laptop does not imply Render access.

## 4. Vercel static frontend

Create a Vercel project with this repository and set the project root directory
to `frontend/`. No frontend build is needed; publish the static files directly.
Set `frontend/config.js`'s `apiBaseUrl` to the Render HTTPS origin (origin only,
no `/api/v1` suffix). The default empty value uses the page's own origin for
same-origin local serving. The file is public by design and must contain only a
backend URL, never a bearer token or Supabase key.

Add the exact Vercel origin to Render's `CORS_ALLOWED_ORIGINS`. CORS is not
authentication; assessment APIs validate guest ownership independently. Keep
the internal API token and Supabase service-role key out of Vercel and all
browser-visible files.

## 5. Local development

Copy `.env.example` to `.env`, configure a local PostgreSQL URL and local image
storage, then install the root backend dependencies and run:

```powershell
.\backend\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\backend\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8081
```

For a separate static frontend server, set `frontend/config.js` to
`http://127.0.0.1:8081`, serve the `frontend/` directory on port 5500, and add
that exact origin to local `CORS_ALLOWED_ORIGINS` if overriding defaults. The
backend also serves the frontend at `/` and serves its root-relative assets.
Install the optional local Paddle runtime only for local OCR development:

```powershell
.\backend\.venv\Scripts\python.exe -m pip install -r requirements-local-ocr.txt
```

Local YOLO requires its dedicated damage virtual environment and checkpoint;
neither is downloaded or initialized by this app.

## 6. Smoke checks after manual provisioning

1. `GET /health/live` should respond without checking external providers.
2. `GET /health/ready` should report database connectivity.
3. Check the database migration revision independently (`alembic current`).
4. Upload a non-sensitive test image and verify metadata records provider
   `supabase`; restart/redeploy and confirm the image can still be read.
5. Confirm disabled damage results are `unavailable`/`inspection_unavailable`,
   not a successful empty scan.
6. Verify remote OCR with an authorized test image and inspect provider
   timeout/limit behavior. External services are not covered by mocked tests.
7. Verify a fresh browser receives a guest session, can create and revisit only
   its own assessments, and cannot read another guest's assessment. Confirm
   expired/revoked sessions are rejected and storage-loss messaging is clear.

## Free-tier and readiness limits

Supabase Free currently documents 500 MB database and 1 GB file storage, with
projects eligible for inactivity pausing after a week; provider quotas and
account eligibility can change. Vercel Hobby is a free individual-development
plan with usage caps and plan terms. Render Free cold starts and ephemeral local
storage make it unsuitable for guaranteed continuous availability. These are
project/demo limits and do not establish that account signup needs no card; check
the current signup/billing screens before creating services.

This repository is prepared for manual configuration, not deployed or
production-ready. Blocking checks remain: no live Supabase project/storage bucket
or migration test has been performed; no Render/Vercel origin or environment is
configured; model artifact deployment remains unresolved; and remote OCR,
market, and optional LLM services have not been verified from the deployment.
