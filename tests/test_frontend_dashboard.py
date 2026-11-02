from pathlib import Path
import re
from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.config import settings
from app.main import app


FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


def test_landing_report_assets_and_hero_image_are_served():
    with TestClient(app) as client:
        page = client.get("/dashboard/")
        script = client.get("/dashboard/app.js")
        styles = client.get("/dashboard/styles.css")
        hero = client.get("/dashboard/assets/vehicle-hero.png")

    assert page.status_code == script.status_code == styles.status_code == hero.status_code == 200
    assert "Look closer." in page.text
    assert "hero-visual" in page.text
    assert 'id="photo-input" type="file"' in page.text
    assert "report-shell" in page.text
    assert 'href="./assets/vehicle-hero.png"' in page.text
    assert hero.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(hero.content) > 100_000
    assert "prefers-reduced-motion" in styles.text
    assert "renderMarket" in script.text
    assert "--emerald" in styles.text
    assert "fonts.googleapis.com" not in styles.text


def test_each_frontend_route_supports_direct_navigation_and_refresh():
    assessment_id = uuid4()
    routes = [
        ("/", 'data-route-page="landing"'),
        ("/inspect", 'data-route-page="inspect"'),
        ("/inspection/" + str(assessment_id), 'data-route-page="inspection"'),
        ("/report/" + str(assessment_id), 'data-route-page="report"'),
        ("/sample-report", 'data-route-page="report"'),
        ("/how-it-works", 'data-route-page="how"'),
    ]
    with TestClient(app) as client:
        for route, expected_page in routes:
            first = client.get(route)
            refreshed = client.get(route)
            assert first.status_code == refreshed.status_code == 200
            assert first.headers["content-type"].startswith("text/html")
            assert expected_page in refreshed.text


def test_hidden_state_assignments_reference_existing_dom_ids():
    script = (FRONTEND / "app.js").read_text(encoding="utf-8")
    page = (FRONTEND / "index.html").read_text(encoding="utf-8")
    html_ids = set(re.findall(r'\bid="([^"]+)"', page))
    hidden_ids = set(re.findall(r"\$\('([^']+)'\)\.hidden\s*=", script))

    assert "sample-report-banner" in html_ids
    assert "route-access-form" not in html_ids
    assert "api-token" not in html_ids
    assert hidden_ids <= html_ids


def test_public_portfolio_pages_are_open_while_api_authentication_remains(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "development")
    monkeypatch.setattr(settings, "api_access_token", SecretStr("portfolio-test-token"))

    with TestClient(app) as client:
        landing = client.get("/")
        how = client.get("/how-it-works")
        sample = client.get("/sample-report")
        protected_api = client.get("/api/v1/assessments")

    assert landing.status_code == how.status_code == sample.status_code == 200
    assert "Illustrative sample report" in sample.text
    assert protected_api.status_code == 401
    assert "portfolio-test-token" not in protected_api.text


def test_submission_reuses_existing_upload_workflow_and_pricing_contracts():
    script = (FRONTEND / "app.js").read_text(encoding="utf-8")
    page = (FRONTEND / "index.html").read_text(encoding="utf-8")
    initial_form = page.split('<form id="inspection-form"', maxsplit=1)[1].split("</form>", maxsplit=1)[0]

    assert "body.append('file', file, file.name)" in script
    assert "`/${encodeURIComponent(id)}/workflow`" in script
    assert "image_ids: imageIds" in script
    assert "`/${encodeURIComponent(id)}/images`" in script
    assert "`/${encodeURIComponent(id)}/report`" in script
    assert "window.addEventListener('hashchange', handleHashChange)" in script
    assert "function routeFor(hash)" in script
    assert "if (savedReport)" in script
    assert "const images = await get(`/${encodeURIComponent(id)}/images`)" in script
    assert "dropZone.classList.add('drag-over')" in script
    assert "dropZone.classList.remove('drag-over')" in script
    assert "missing_fields" in script
    assert "`/${encodeURIComponent(assessmentId)}/workflow`" in script
    assert "vehicle: pricingVehicle" in script
    assert "workflowPayload.image_types = selectedImageTypes" in script
    assert "imageTypes: workflowImageTypes" in script
    assert "photoTypes = photoTypes.filter" in script
    assert "let skipNextHashChange = '';" in script
    assert "if (location.hash === skipNextHashChange)" in script
    assert "pricing.status === 'available' && reportStage.status === 'available'" in script
    assert "navigate(`/report/${encodeURIComponent(assessmentId)}`)" in script
    for field in (
        "make",
        "model",
        "variant",
        "fuel_type",
        "registration_year",
        "registration_number",
        "asking_price",
        "seller_type",
        "odometer_km",
        "transmission",
        "mileage_kmpl",
        "max_power_bhp",
        "engine_cc",
        "seats",
    ):
        assert f'name="{field}"' in initial_form


def test_inspection_is_single_page_with_optional_vehicle_inputs_and_inline_report_editor():
    page = (FRONTEND / "index.html").read_text(encoding="utf-8")
    script = (FRONTEND / "app.js").read_text(encoding="utf-8")
    styles = (FRONTEND / "styles.css").read_text(encoding="utf-8")
    initial_form = page.split('<form id="inspection-form"', maxsplit=1)[1].split("</form>", maxsplit=1)[0]

    assert "<dialog" not in page
    assert "Complete vehicle details." not in page
    assert 'id="pricing-form"' in page
    assert 'id="pricing-form"' not in initial_form
    assert "required" not in initial_form
    assert 'id="start-button" type="submit"' in initial_form
    assert "workflowPayload.vehicle = vehicleDetails" in script
    assert "vehicle: pricingVehicle" in script
    assert ".vehicle-input-grid" in styles
    assert ".details-card { max-height: none; overflow: visible; }" in styles
    assert ".submission-layout { grid-template-columns: 1fr;" in styles


def test_pricing_completion_allows_manual_corrections_and_displays_provenance():
    script = (FRONTEND / "app.js").read_text(encoding="utf-8")
    page = (FRONTEND / "index.html").read_text(encoding="utf-8")
    definitions = script.split("const pricingFieldDefinitions = {", maxsplit=1)[1].split("};", maxsplit=1)[0]

    assert 'id="pricing-fields"></div>' in page
    assert "for (const name of missingFields)" in script
    for field in ("odometer_km", "seller_type", "transmission", "mileage_kmpl", "max_power_bhp"):
        assert f"{field}:" in definitions
    for field in ("make", "model", "registration_year", "fuel_type", "engine_cc", "seats"):
        assert f"{field}:" in definitions
    assert "owner_count:" not in definitions
    assert "const sources = [assessmentDetails, lookup, estimateVehicle, pricingVehicle || {}]" in script
    assert "source.registration_date" in script
    assert "Review or correct pricing details" in page
    assert "control.dataset.originalValue = control.value" in script
    assert "Prefilled from available vehicle data. Change only to correct the value." in script
    assert "Still needed: ${missingLabels.join(', ')}." in script
    assert "Correct these vehicle details before retrying: ${missingLabels.join(', ')}." in script
    assert "lookupResult.field_provenance" in script
    assert "report.vehicle_details_provenance" in script
    assert 'name="make"' in page and 'name="model"' in page
    assert 'name="registration_year"' in page and 'name="fuel_type"' in page
    assert 'name="engine_cc"' in page and 'name="seats"' in page
    assert "cleanStatus(report.challan_lookup?.status)" in script
    assert "cleanStatus(report.fastag_lookup?.status)" in script
    assert "Market valuation unavailable" in script
    assert "market.refresh_status === 'refresh_failed'" in script
    assert "provider_error" in script


def test_report_prefill_maps_manufacturing_year_to_registration_year():
    script = (FRONTEND / "app.js").read_text(encoding="utf-8")
    assert "'manufacturing_year'" in script
    assert "registration_year: ['registration_year', 'year', 'manufacturing_year']" in script


def test_report_highlights_model_estimate_as_centered_resale_price_band():
    script = (FRONTEND / "app.js").read_text(encoding="utf-8")
    page = (FRONTEND / "index.html").read_text(encoding="utf-8")
    styles = (FRONTEND / "styles.css").read_text(encoding="utf-8")

    assert "function resalePriceRange(raw)" in script
    assert "Math.floor(amount / 10000) * 10000" in script
    assert "lower + 10000" in script
    assert "resalePriceRange(estimate) || 'Not available'" in script
    assert "recommendation.ml_estimate_inr" in script
    assert "Rounded to the nearest ₹10,000 band from the model estimate; not a confidence interval." in script
    assert "Actual selling price may vary depending on vehicle condition, location, ownership and market demand." in script
    assert "This comparison is indicative and does not guarantee the final negotiated price." in script
    assert "Figures are estimates and may differ from actual outcomes." in script
    assert "Research model · not market value" not in script
    assert page.index('class="report-masthead"') < page.index('id="report-overview"')
    assert page.index('id="report-overview"') < page.index('id="report-load-note"')
    assert ".report-price-hero" in styles
    assert ".report-price-range" in styles
    assert "@media (max-width: 650px)" in styles


def test_report_prefill_normalizes_fuel_and_transmission_to_form_options():
    script = (FRONTEND / "app.js").read_text(encoding="utf-8")
    assert "fuelMap" in script
    assert "transmissionMap" in script
    assert "Petrol: 'petrol'" in script
    assert "Diesel: 'diesel'" in script
    assert "Manual: 'manual'" in script
    assert "Automatic: 'automatic'" in script


def test_report_prefill_uses_lookup_details_when_available():
    script = (FRONTEND / "app.js").read_text(encoding="utf-8")
    assert "const sources = [assessmentDetails, lookup, estimateVehicle, pricingVehicle || {}]" in script
    assert "source[candidate] !== null && source[candidate] !== undefined && source[candidate] !== ''" in script


def test_report_prefill_preserves_user_entered_values_over_lookup():
    script = (FRONTEND / "app.js").read_text(encoding="utf-8")
    assert "{ ...reportVehiclePrefill(), ...pricingVehicle }" in script


def test_market_display_labels_discovered_advertisements_and_lists_sources():
    script = (FRONTEND / "app.js").read_text(encoding="utf-8")
    assert "marketPriceLabel" in script
    assert "₹${(amount / 100000)" in script
    assert "listing.variant" in script
    assert "listing.year" in script
    assert "listing.priced_used_listing !== false" in script
    assert "listing.priced_comparable !== false" in script
    assert "function marketSourceLink(sourceUrl)" in script
    assert "link.href = sourceUrl" in script
    assert "link.textContent = 'View listing'" in script
    assert "marketSourceLink(listing.source_url)" in script
    assert "['http:', 'https:'].includes(parsed.protocol)" in script
    assert "No sourced used-car listings found." in script
    assert "Relevant market context" not in script
    assert "Differences or missing details" not in script


def test_market_listing_shows_unavailable_labels_for_missing_vehicle_details():
    script = (FRONTEND / "app.js").read_text(encoding="utf-8")
    assert "const variant = typeof listing.variant === 'string' && listing.variant.trim()" in script
    assert "'unavailable';" in script
    assert "const year = Number.isInteger(listing.year) ? String(listing.year) : 'unavailable';" in script
    assert "`Variant: ${variant} · Year: ${year}`" in script


def test_report_lookup_error_message_is_shown_when_lookup_fails():
    script = (FRONTEND / "app.js").read_text(encoding="utf-8")
    assert "lookup.error_code" in script
    assert "provider_unavailable: 'The vehicle details provider was unavailable" in script
    assert "provider_failed: 'The vehicle details provider did not return usable" in script


def test_pricing_completion_has_validation_duplicate_guard_and_failure_feedback():
    script = (FRONTEND / "app.js").read_text(encoding="utf-8")
    submit = script.split("$('pricing-form').addEventListener('submit'", maxsplit=1)[1].split("let submissionInProgress", maxsplit=1)[0]

    assert "if (pricingSubmissionInProgress) return" in submit
    assert "pricingForm.reportValidity()" in submit
    assert "pricingSubmissionInProgress = true" in submit
    assert "pricingSubmissionInProgress = false" in submit
    assert "finally" in submit
    assert "saved report could not be refreshed" in submit
    assert "model could not prepare an estimate" in submit
    assert "We could not complete the estimate." in submit

def test_fresh_submission_posts_workflow_without_frontend_access_code():
    script = (FRONTEND / "app.js").read_text(encoding="utf-8")
    submit = script.split("form.addEventListener('submit', async event =>", maxsplit=1)[1].split("$('new-inspection')", maxsplit=1)[0]
    resume = script.split("async function resumeInspection", maxsplit=1)[1].split("async function loadRoute", maxsplit=1)[0]
    request = script.split("async function post", maxsplit=1)[1].split("async function get", maxsplit=1)[0]

    assert submit.index("await post('', { json: assessmentBody })") < submit.index("await post(`/${assessmentId}/images`")
    assert submit.index("imageIds = uploaded.map(image => image.id)") < submit.index("await navigate(`/inspection/${encodeURIComponent(assessmentId)}`, {")
    assert "vehicle: pricingVehicle" in submit
    assert "if (Array.isArray(justUploadedImageIds) && justUploadedImageIds.length > 0)" in resume
    assert "imageIds = justUploadedImageIds" in resume
    assert "post(`/${encodeURIComponent(id)}/workflow`, {" in resume
    assert "json: workflowPayload" in resume
    assert "timeoutMs: 180_000" in resume
    assert resume.index("post(`/${encodeURIComponent(id)}/workflow`") < resume.index("navigate(`/report/${encodeURIComponent(id)}`)")
    assert "headers.Authorization" not in request
    assert "api-token" not in script
    assert "route-access-form" not in script


def test_submission_failure_clears_loading_and_duplicate_guard():
    script = (FRONTEND / "app.js").read_text(encoding="utf-8")
    submit = script.split("form.addEventListener('submit', async event =>", maxsplit=1)[1].split("$('new-inspection')", maxsplit=1)[0]
    resume = script.split("async function resumeInspection", maxsplit=1)[1].split("async function loadRoute", maxsplit=1)[0]

    assert "if (submissionInProgress) return" in submit
    assert "finally" in submit
    assert "setBusy(false)" in submit
    assert "submissionInProgress = false" in submit
    assert "await closeOverlay({ message: 'We could not continue this review' })" in resume
    assert "setPageError(" in resume
    assert "fetchWithTimeout" in script
    assert "This step took too long. Check its status and try again." in script


def test_progress_is_indeterminate_until_completed_response_and_errors_are_sanitized():
    script = (FRONTEND / "app.js").read_text(encoding="utf-8")
    close_overlay = script.split("async function closeOverlay", maxsplit=1)[1].split("overlay.addEventListener", maxsplit=1)[0]

    assert "textContent = '—'" in script
    assert "if (complete)" in close_overlay
    assert "textContent = '100'" in close_overlay
    assert "response.status === 401" in script
    assert "response.status === 413" in script
    assert "data.detail" not in script
    assert "No sourced used-car listings found." in script


def test_report_sections_render_real_payload_fields_and_token_is_not_persisted():
    script = (FRONTEND / "app.js").read_text(encoding="utf-8")

    assert "entry.text" in script
    assert "finding.damage_type" in script
    assert "finding.confidence" in script
    assert "prediction.amount" in script
    assert "recommendation.explanation" in script
    assert "report.vehicle_details_lookup" in script
    assert "report.assessment?.vehicle_details" in script
    assert "finding.image_id" in script
    assert "finding-evidence" in script
    assert "images/${encodeURIComponent(imageId)}/content" in script
    assert "recommendation.absolute_difference_inr" in script
    assert "market.market_asking_price_range" not in script
    assert "market.market_research_results" in script
    assert "localStorage" not in script
    assert "sessionStorage" not in script
    assert "console.log" not in script
    assert "Authorization" not in script
    assert "route-access-input" not in script
    assert "sample-report" in script


def test_condition_review_appends_dom_nodes_instead_of_stringifying_them():
    script = (FRONTEND / "app.js").read_text(encoding="utf-8")
    condition = script.split("function renderCondition(workflow)", maxsplit=1)[1].split("function pricingNeedsFeatures", maxsplit=1)[0]

    assert "const emptyFindings = node('div', 'empty-findings')" in condition
    assert "emptyFindings.append(node('span', '', '✦'), line)" in condition
    assert "node('div', 'empty-findings', node(" not in condition
    assert "finding.damage_type || 'Visible finding'" in condition
    assert "node('span', 'confidence', confidence)" in condition
    assert "item.append(evidencePhoto)" in condition


def test_damage_status_mapping_never_treats_unavailable_as_no_findings():
    script = (FRONTEND / "app.js").read_text(encoding="utf-8")
    condition = script.split("function renderCondition(workflow)", maxsplit=1)[1].split("function pricingNeedsFeatures", maxsplit=1)[0]
    workflow_mapping = script.split("function workflowFromReport(reportResponse)", maxsplit=1)[1].split("function illustrativeSampleReport", maxsplit=1)[0]

    for status in (
        "completed_with_findings",
        "completed_no_findings",
        "unavailable",
        "failed",
    ):
        assert status in script
    assert "stage.status === 'completed_no_findings'" in condition
    assert "stage.status === 'unavailable'" not in condition
    assert "This does not establish that the vehicle is damage-free." in condition
    assert "results_available" in workflow_mapping
    assert ": 'unavailable'" in workflow_mapping


def test_pricing_report_displays_saved_missing_fields_and_failure_status():
    script = (FRONTEND / "app.js").read_text(encoding="utf-8")

    assert "missing_vehicle_attributes" in script
    assert "Still needed:" in script
    assert "const savedPricing = sourceStages.pricing || {}" in script
    assert "savedPricing.missing_fields" in script
    assert "savedPricing.error_code" in script


def test_report_explains_registration_lookup_failures_without_provider_payloads():
    script = (FRONTEND / "app.js").read_text(encoding="utf-8")
    assert "lookup.error_code" in script
    assert "provider_disabled: 'Vehicle details lookup is disabled" in script
    assert "provider_unavailable: 'The vehicle details provider was unavailable" in script
    assert "provider_failed: 'The vehicle details provider did not return usable" in script
    assert "error_message" not in script


def test_dark_mobile_layout_and_accessible_upload_controls():
    page = (FRONTEND / "index.html").read_text(encoding="utf-8")
    styles = (FRONTEND / "styles.css").read_text(encoding="utf-8")

    assert '<a class="header-action" href="#/inspect">' in page
    assert 'aria-label="Main navigation"' in page
    assert 'aria-labelledby="photos-title"' in page
    assert 'aria-describedby="upload-help"' in page
    assert 'aria-live="polite"' in page
    assert "color-scheme: dark" in styles
    assert "@media (max-width: 650px)" in styles
    assert "@media (prefers-reduced-motion: reduce)" in styles
