# Vehicle IQ backend deployment notes

## Required production configuration

Set these values in the deployment's secret/configuration manager. The backend
does not need committed credentials.

- `APP_ENV=production`
- `DATABASE_URL=postgresql+psycopg://.../vehicle_iq` — supply a valid URL with
  reserved password characters URL-encoded. The value is never included in
  production validation errors.
- `API_ACCESS_TOKEN` — a random bearer token at least 32 characters long. The
  current API uses one shared token; it does not implement user accounts or
  per-assessment ownership.

Production startup fails if the token is too short or missing, or the database
URL does not use `postgresql+psycopg`. `DATABASE_URL` is required in every
environment by application settings.

Optional capacity settings:

- `MAX_IMAGE_UPLOAD_BYTES` (default 10 MiB; maximum 100 MiB)
- `MAX_EXTERNAL_RESPONSE_BYTES` (default 2 MiB; maximum 10 MiB)
- `API_RATE_LIMIT_PER_MINUTE` (default 600 per client IP and process)

Vehicle Details, Challan, and FASTag candidate providers remain disabled unless
their respective `*_LOOKUP_ENABLED` setting is explicitly enabled. Do not enable
them until authorized use and provider compatibility are confirmed.

## Database migrations

The checked-in migration chain is linear:

1. `20261027_0001` creates the core assessment tables.
2. `20261028_0002` allows a damage finding's component to be null.
3. `20261029_0003` adds vehicle specifications.
4. `20261030_0004` adds vehicle specification identity.
5. `20261101_0005` adds market listings.
6. `20261103_0006` adds nullable PostgreSQL image bytes and changes the default
   storage provider for new records. Existing image metadata and filesystem
   object references are retained; their bytes are not backfilled. Such legacy
   images remain unavailable through image-content and processing APIs until
   their bytes are restored separately.

For a newly provisioned PostgreSQL database, configure `DATABASE_URL`, then run
the migration as a controlled release step from the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -m alembic upgrade head
```

Do not run migrations automatically in each web-worker startup. The API
readiness endpoint checks database connectivity only; it does not confirm that
the database is at the latest Alembic revision. Confirm the applied revision as
part of deployment/release verification.

## Image persistence and hosting

New image bytes and their metadata are stored together in PostgreSQL
(`BYTEA`). Image metadata listing defers the byte column, while the content,
OCR, and damage-processing paths load bytes only when required. This avoids
dependence on ephemeral backend filesystems and Render persistent disks.

The migration intentionally does not populate bytes for existing records that
only reference local files or external objects. Those metadata rows are
preserved, and their images are unavailable if their original content cannot
be restored. PostgreSQL storage increases database usage by up to the configured
10 MiB per uploaded image; monitor database capacity accordingly.

## Start and health checks

After configuration and the migration release step, start the ASGI app with:

```powershell
.\backend\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8081
```

- `GET /health/live` checks that the process responds.
- `GET /health/ready` runs a read-only `SELECT 1`; it returns 503 when the DB is
  unavailable, but does not verify schema/migration currency.
- `/api/v1/*` requires the configured bearer token in production.

### Render native Python service

The repository root `.python-version` pins the Render runtime to Python 3.13.16.
Configure the Render service with the repository root as its root directory and
these commands:

- Build: `pip install -r requirements.txt`
- Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

Render supplies `PORT`; do not hardcode the local development port in its start
command. Set the production variables listed above in the Render environment.
The tracked `models/vehicle_iq_price_model.joblib` is loaded lazily through
`PRICE_MODEL_PATH`, which defaults to that repository-relative artifact. An
optional override can point to another mounted artifact. The estimate remains
research-only and is not a verified market valuation.

The in-process rate limiter is per worker and can be bypassed across multiple
workers. Public deployments should also apply rate limiting at a trusted reverse
proxy/API gateway. TLS termination, persistent-volume behavior, secret-manager
integration, and deployment health-check wiring must be verified on the chosen
hosting platform; no deployment platform is configured in this repository.
