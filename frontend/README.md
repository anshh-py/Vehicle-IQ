# Vehicle IQ inspection experience

The FastAPI app serves the photo-led experience at `/`, with direct routes for
`/inspect`, `/inspection/{assessment_id}`, `/report/{assessment_id}`,
`/how-it-works`, and `/sample-report`. The sample report is static, visibly
labelled synthetic content and does not read database records or call APIs.
The former `/dashboard/` path remains available for compatibility. No frontend
package installation is needed.

## Local portfolio demo

1. Configure the local database and ensure its schema is already migrated.
2. For an unauthenticated **local-only** demo, use `APP_ENV=development` and
   leave `API_ACCESS_TOKEN` unset. In development, the backend permits API
   requests when that optional token is absent. Do not expose this configuration
   to a public network or tunnel.
3. Start the app from the repository root:

   ```powershell
   .\backend\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8081
   ```

4. Open <http://127.0.0.1:8081/>. Use **Sample report** to browse the static
   illustrative report, or choose **Inspect a car** to submit 1–10 JPEG, PNG,
   or WebP images. Registration number and asking price are optional.
5. If pricing needs vehicle specifications, enter them in the optional form.

The frontend intentionally has no access-code field and never sends a bearer
token. If `API_ACCESS_TOKEN` is configured, protected API requests remain
protected and the inspection UI will show a configuration error instead of
asking for the token. Production requires API authentication; see
[`DEPLOYMENT.md`](../DEPLOYMENT.md). The shared backend token is not a customer
account or per-assessment authorization system.

## Sample report and limitations

The sample uses synthetic Maruti Swift inputs and an illustrative research-model
estimate from those inputs. It contains no OCR or damage-inference result, no
customer record, and no market comparables. Its banner identifies it as
demonstration content; it is not evidence about a real vehicle.

The assessment workflow returns one synchronous JSON response. It does not
stream actual stage events, percentages, or narrative report text. The live
inspection therefore uses an indeterminate scanning animation. Real reports
display only returned OCR evidence, damage findings, stored vehicle details,
the research-only model estimate, and permitted market evidence when available.

## Checks

```powershell
node --check frontend\app.js
.\backend\.venv\Scripts\python.exe -m pytest tests\test_frontend_dashboard.py tests\test_security_middleware.py -q
```
