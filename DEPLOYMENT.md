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
- `IMAGE_STORAGE_DIR` — an explicitly configured absolute path backed by durable
  storage. Startup checks that it was configured, but cannot determine whether
  the hosting platform actually mounts persistent storage there.

Production startup fails if the token is too short or missing, the database URL
does not use `postgresql+psycopg`, or the image storage path is not explicitly
set. `DATABASE_URL` is required in every environment by application settings.

Optional capacity settings:

- `MAX_IMAGE_UPLOAD_BYTES` (default 10 MiB; maximum 100 MiB)
- `MAX_EXTERNAL_RESPONSE_BYTES` (default 2 MiB; maximum 10 MiB)
- `API_RATE_LIMIT_PER_MINUTE` (default 600 per client IP and process)

Vehicle Details, Challan, and FASTag candidate providers remain disabled unless
their respective `*_LOOKUP_ENABLED` setting is explicitly enabled. Do not enable
them until authorized use and provider compatibility are confirmed.

## Database migrations

The checked-in migration chain is linear:

1. `20261027_0001` creates the seven core assessment tables.
2. `20261028_0002` allows a damage finding's component to be null.

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

Image bytes are stored by the local filesystem adapter under
`IMAGE_STORAGE_DIR`; PostgreSQL stores metadata and object keys only. The local
default `.local-storage/vehicle-images` is git-ignored and suitable for local
development. It is not guaranteed to survive container replacement, host
recreation, or deployment to an ephemeral filesystem.

Before deployment, provision a persistent mounted volume and point
`IMAGE_STORAGE_DIR` at its absolute mount path. This adapter is local-disk based;
multiple replicas require a shared filesystem or a future object-storage
adapter. No platform-specific volume configuration is included in this repo.

## Start and health checks

After configuration and the migration release step, start the ASGI app with:

```powershell
.\backend\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8081
```

- `GET /health/live` checks that the process responds.
- `GET /health/ready` runs a read-only `SELECT 1`; it returns 503 when the DB is
  unavailable, but does not verify schema/migration currency.
- `/api/v1/*` requires the configured bearer token in production.

The in-process rate limiter is per worker and can be bypassed across multiple
workers. Public deployments should also apply rate limiting at a trusted reverse
proxy/API gateway. TLS termination, persistent-volume behavior, secret-manager
integration, and deployment health-check wiring must be verified on the chosen
hosting platform; no deployment platform is configured in this repository.
