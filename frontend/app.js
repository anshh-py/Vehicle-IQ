(() => {
  'use strict';

  const API = `https://vehicle-iq-8wzk.onrender.com/api/v1/assessments`;
  const form = document.getElementById('inspection-form');
  const photoInput = document.getElementById('photo-input');
  const photoGrid = document.getElementById('photo-grid');
  const overlay = document.getElementById('inspection-overlay');
  let photos = [];
  let photoCategories = [];
  let previewUrls = [];
  let assessmentId = null;
  let imageIds = [];
  let latestWorkflow = null;
  let pricingVehicle = null;
  let workflowAssessmentId = null;
  let sampleReportActive = false;
  let routeLoadSequence = 0;

  const $ = id => document.getElementById(id);
  const node = (tag, className, text) => {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (text !== undefined && text !== null) item.textContent = String(text);
    return item;
  };

  function setMessage(target, message) {
    target.textContent = message;
    target.hidden = !message;
  }

  function normalizeRegistration(value) {
    return String(value || '').trim().replace(/\s+/g, '').toUpperCase();
  }

  function isSupportedRegistration(value) {
    return /^[A-Z]{2}[0-9]{1,2}[A-Z]{1,4}[0-9]{1,4}$/.test(value)
      || /^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$/.test(value);
  }

  function workflowHasValidOcrRegistration(workflow) {
    const report = workflow?.stages?.report?.result?.report_data || {};
    const registration = report.registration_ocr_assessment || {};
    return registration.status === 'candidate'
      && isSupportedRegistration(normalizeRegistration(registration.normalized_candidate));
  }

  function requestError(message, status) {
    const error = new Error(message);
    error.status = status;
    return error;
  }

  function fetchWithTimeout(url, options, timeoutMs) {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), timeoutMs);
    return fetch(url, { ...options, signal: controller.signal })
      .finally(() => window.clearTimeout(timer));
  }

  function showPage(name) {
    document.querySelectorAll('[data-route-page]').forEach(page => {
      page.hidden = page.dataset.routePage !== name;
    });
    document.body.dataset.page = name;
    const titles = {
      landing: 'Vehicle IQ — Inspect with more clarity',
      inspect: 'Start an inspection — Vehicle IQ',
      inspection: 'Review in progress — Vehicle IQ',
      report: 'Your inspection report — Vehicle IQ',
      how: 'How it works — Vehicle IQ',
      error: 'Review unavailable — Vehicle IQ',
    };
    document.title = titles[name] || titles.error;
  }

  function showRouteError(title, message, { continuePath = '' } = {}) {
    showPage('error');
    $('route-error-title').textContent = title;
    $('route-error-copy').textContent = message;
    $('route-continue').hidden = !continuePath;
    if (continuePath) $('route-continue').href = continuePath;
  }

  function updatePhotoPreview() {
    previewUrls.forEach(url => URL.revokeObjectURL(url));
    previewUrls = [];
    photoGrid.replaceChildren();
    $('photo-count').innerHTML = '';
    $('photo-count').append(document.createTextNode(String(photos.length)), node('i', '', ' / 10'));
    photos.forEach((photo, index) => {
      const card = node('div', 'photo-thumb');
      const image = document.createElement('img');
      image.src = URL.createObjectURL(photo);
      image.alt = `Selected car photo ${index + 1}: ${photo.name}`;
      previewUrls.push(image.src);
      const remove = node('button', 'photo-remove', '×');
      remove.type = 'button';
      remove.setAttribute('aria-label', `Remove photo ${index + 1}`);
      remove.addEventListener('click', () => {
        photos = photos.filter((_, photoIndex) => photoIndex !== index);
        photoCategories = photoCategories.filter((_, photoIndex) => photoIndex !== index);
        updatePhotoPreview();
      });
      const categoryLabel = node('label', 'photo-category-field');
      categoryLabel.append(node('span', '', 'Photo type'));
      const category = document.createElement('select');
      category.setAttribute('aria-label', `Photo ${index + 1} type`);
      [
        ['unknown', 'Choose type'],
        ['vehicle_exterior', 'Vehicle exterior'],
        ['vehicle_interior', 'Vehicle interior'],
        ['damage_closeup', 'Damage close-up'],
        ['registration_plate', 'Registration plate'],
        ['document', 'Document'],
      ].forEach(([value, label]) => {
        const option = node('option', '', label);
        option.value = value;
        category.append(option);
      });
      category.value = photoCategories[index] || 'unknown';
      category.addEventListener('change', () => { photoCategories[index] = category.value; });
      categoryLabel.append(category);
      card.append(image, node('span', 'photo-thumb-name', photo.name), categoryLabel, remove);
      photoGrid.append(card);
    });
  }

  function addPhotos(fileList) {
    const incoming = [...fileList];
    const acceptedTypes = new Set(['image/jpeg', 'image/png', 'image/webp']);
    const invalid = incoming.find(file => !acceptedTypes.has(file.type));
    if (invalid) {
      setMessage($('form-message'), `${invalid.name} is not a JPEG, PNG, or WebP image.`);
      return;
    }
    if (photos.length + incoming.length > 10) {
      setMessage($('form-message'), 'Choose up to 10 photos for one inspection.');
      return;
    }
    setMessage($('form-message'), '');
    photos = [...photos, ...incoming];
    photoCategories = [...photoCategories, ...incoming.map(() => 'unknown')];
    updatePhotoPreview();
  }

  photoInput.addEventListener('change', event => {
    addPhotos(event.target.files);
    photoInput.value = '';
  });
  const dropZone = $('upload-zone');
  dropZone.addEventListener('dragover', event => {
    event.preventDefault();
    dropZone.classList.add('drag-over');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', event => {
    event.preventDefault();
    dropZone.classList.remove('drag-over');
    addPhotos(event.dataTransfer.files);
  });

  async function post(path, { json, file, imageType, timeoutMs = 60_000 } = {}) {
    const headers = {};
    const options = { method: 'POST', headers };
    if (file) {
      const body = new FormData();
      body.append('file', file, file.name);
      if (imageType) body.append('image_type', imageType);
      options.body = body;
    } else {
      headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(json ?? {});
    }

    let response;
    try {
      response = await fetchWithTimeout(`${API}${path}`, options, timeoutMs);
    } catch (error) {
      if (error?.name === 'AbortError') {
        throw requestError('This step took too long. Check its status and try again.', 504);
      }
      throw new Error('We could not connect. Check that the inspection service is available, then try again.');
    }
    let data = null;
    try { data = await response.json(); } catch { /* handled as a generic response below */ }
    if (!response.ok) {
      if (response.status === 401) throw requestError('This demo is not available with the current server configuration.', 401);
      if (response.status === 413) throw requestError('One of the photos is larger than the allowed size.', 413);
      if (response.status === 422 && file) throw requestError('One of the selected files is not a supported image. Choose a JPEG, PNG, or WebP photo.', 422);
      if (response.status === 422 && path.endsWith('/price-estimates')) throw requestError('Check the vehicle details and try the price estimate again.', 422);
      if (response.status === 422) throw requestError('Check the registration and asking price details, then try again.', 422);
      if (response.status === 409 && path.endsWith('/registration-number')) throw requestError('A registration number is already saved for this assessment. It was not changed.', 409);
      if (response.status === 404) throw requestError('This review could not be found. Start a new inspection and try again.', 404);
      if (response.status === 429) throw requestError('Please wait a moment before trying again.', 429);
      if (response.status === 503 || response.status === 504 || response.status === 502) throw requestError('One or more inspection services are temporarily unavailable. Your results will show which parts completed.', response.status);
      throw requestError('We could not complete that step. Please try again.', response.status);
    }
    return data;
  }

  async function get(path, { allow404 = false } = {}) {
    let response;
    try {
      response = await fetchWithTimeout(`${API}${path}`, {}, 45_000);
    } catch (error) {
      if (error?.name === 'AbortError') {
        throw requestError('Loading this review took too long. Please try again.', 504);
      }
      throw new Error('We could not connect. Check that the inspection service is available, then try again.');
    }
    if (allow404 && response.status === 404) return null;
    if (!response.ok) {
      const message = response.status === 401
        ? 'This demo is not available with the current server configuration.'
        : response.status === 404
          ? 'This review could not be found. Start a new inspection and try again.'
          : response.status === 429
            ? 'Please wait a moment before trying again.'
            : 'We could not load this review. Please try again.';
      throw requestError(message, response.status);
    }
    try {
      return await response.json();
    } catch {
      throw new Error('The review response could not be read. Please try again.');
    }
  }

  async function getImageContent(assessmentIdValue, imageId) {
    try {
      const response = await fetchWithTimeout(
        `${API}/${encodeURIComponent(assessmentIdValue)}/images/${encodeURIComponent(imageId)}/content`,
        {},
        30_000,
      );
      if (!response.ok) return null;
      return await response.blob();
    } catch {
      return null;
    }
  }

  function showOverlay(message) {
    $('progress-value').textContent = '—';
    $('overlay-status').textContent = message;
    $('overlay-footnote').textContent = 'Timing varies with the number of photos and the detail in each image.';
    $('progress-bar').style.width = '';
    overlay.classList.remove('complete', 'error');
    overlay.classList.remove('closing');
    overlay.hidden = false;
    overlay.setAttribute('aria-hidden', 'false');
    document.body.classList.add('modal-open');
    overlay.focus();
  }

  function updateOverlay(message) {
    $('overlay-status').textContent = message;
  }

  const delay = milliseconds => new Promise(resolve => window.setTimeout(resolve, milliseconds));

  async function closeOverlay({ complete = false, message = '' } = {}) {
    if (complete) {
      overlay.classList.add('complete');
      $('progress-value').textContent = '100';
      $('overlay-status').textContent = message || 'Your review is ready';
      $('overlay-footnote').textContent = 'Inspection response received.';
      await delay(650);
    } else if (message) {
      overlay.classList.add('error');
      $('overlay-status').textContent = message;
      await delay(700);
    }
    overlay.classList.add('closing');
    await delay(220);
    overlay.hidden = true;
    overlay.classList.remove('closing');
    overlay.setAttribute('aria-hidden', 'true');
    overlay.classList.remove('complete', 'error');
    document.body.classList.remove('modal-open');
  }

  overlay.addEventListener('keydown', event => {
    if (event.key === 'Tab' || event.key === 'Escape') {
      event.preventDefault();
      overlay.focus();
    }
  });

  function setBusy(busy, label = 'Start my inspection') {
    const button = $('start-button');
    button.disabled = busy;
    button.querySelector('span:first-child').textContent = busy ? 'Preparing review…' : label;
  }

  function money(raw) {
    const amount = Number(raw);
    if (!Number.isFinite(amount)) return 'Unavailable';
    return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(amount);
  }

  function resalePriceRange(raw) {
    const amount = Number(raw);
    if (!Number.isFinite(amount) || amount <= 0) return null;
    const lower = Math.floor(amount / 10000) * 10000;
    return `${money(lower)} – ${money(lower + 10000)}`;
  }

  function marketSourceLink(sourceUrl) {
    if (typeof sourceUrl !== 'string') return null;
    try {
      const parsed = new URL(sourceUrl);
      if (!['http:', 'https:'].includes(parsed.protocol) || !parsed.hostname || parsed.username || parsed.password) {
        return null;
      }
      const link = document.createElement('a');
      link.href = sourceUrl;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = 'View listing';
      return link;
    } catch {
      return null;
    }
  }

  function marketPriceLabel(raw) {
    const amount = Number(raw);
    if (!Number.isFinite(amount) || amount <= 0) return null;
    if (amount >= 100000) {
      return `₹${(amount / 100000).toLocaleString('en-IN', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })} lakh`;
    }
    return money(amount);
  }

  function setResultStatus(id, text, tone = '') {
    const badge = $(id);
    badge.textContent = text;
    if (tone) badge.dataset.tone = tone;
    else delete badge.dataset.tone;
  }

  function cleanStatus(status) {
    if (status === 'available') return ['Ready', ''];
    if (status === 'partial') return ['Some results', 'warn'];
    if (status === 'failed') return ['Could not complete', 'warn'];
    if (status === 'completed_with_findings') return ['Findings returned', ''];
    if (status === 'completed_no_findings') return ['No findings returned', ''];
    return ['Unavailable', 'muted'];
  }

  function appendFact(container, label, value) {
    const fact = node('div', 'vehicle-fact');
    fact.append(node('span', 'fact-label', label), node('span', 'fact-value', value || 'Not provided'));
    container.append(fact);
  }

  function renderVehicle(workflow, submittedRegistration) {
    const content = $('vehicle-evidence');
    content.replaceChildren();
    const ocr = workflow.stages?.ocr || {};
    const report = workflow.stages?.report?.result?.report_data || {};
    const lookup = report.vehicle_details_lookup || {};
    const verifiedDetails = ['available', 'partial'].includes(lookup.status) ? (lookup.details || {}) : {};
    const submittedDetails = report.assessment?.vehicle_details || {};
    const [status, tone] = cleanStatus(ocr.status);
    setResultStatus('vehicle-status', status, tone);
    const facts = node('div', 'vehicle-facts');
    appendFact(facts, 'Registration reference', submittedRegistration || verifiedDetails.registration_number || report.assessment?.registration_number || 'Not provided');
    appendFact(facts, 'Photos submitted', sampleReportActive ? 'Illustrative sample' : `${imageIds.length}`);
    if (verifiedDetails.make || verifiedDetails.model) {
      appendFact(facts, 'Vehicle', [verifiedDetails.make, verifiedDetails.model, verifiedDetails.variant].filter(Boolean).join(' '));
      appendFact(facts, 'Year / fuel', [verifiedDetails.manufacturing_year, verifiedDetails.fuel_type].filter(Boolean).join(' · '));
      appendFact(facts, 'Specification source', 'vehicle_api');
      appendFact(facts, 'Vehicle class', verifiedDetails.vehicle_class || 'Not returned');
      appendFact(facts, 'Registration office', verifiedDetails.rto || 'Not returned');
    } else if (submittedDetails.make || submittedDetails.brand || submittedDetails.model) {
      appendFact(facts, 'Details provided', [submittedDetails.make || submittedDetails.brand, submittedDetails.model, submittedDetails.variant].filter(Boolean).join(' '));
      appendFact(facts, 'Year / fuel', [submittedDetails.registration_year || submittedDetails.year, submittedDetails.fuel_type || submittedDetails.fuel].filter(Boolean).join(' · '));
      appendFact(facts, 'Specification source', 'user_input');
    }
    content.append(facts);
    if (sampleReportActive) {
      content.append(node('p', 'quiet-copy', 'This sample includes no OCR output and does not identify a real vehicle.'));
      return;
    }
    const lookupMessages = {
      provider_disabled: 'Vehicle details lookup is disabled; no external lookup was made.',
      provider_unavailable: 'The vehicle details provider was unavailable. No registration details were retrieved.',
      provider_failed: 'The vehicle details provider did not return usable registration details.',
      lookup_failed: 'Vehicle details lookup could not be completed.',
      lookup_storage_failed: 'The lookup result could not be read or saved.',
      registration_not_found: 'No supported registration number was found in the photo text, so lookup was skipped.',
      registration_needs_review: 'The photo text did not provide a clear registration candidate, so lookup was skipped.',
      invalid_registration: 'The registration candidate was not in a supported format, so lookup was skipped.',
    };
    if (lookup.error_code) {
      content.append(node('p', 'quiet-copy', lookupMessages[lookup.error_code] || 'Vehicle registration details are unavailable.'));
    }
    const recognized = Array.isArray(ocr.result) ? ocr.result.filter(item => item.status === 'completed' && item.text) : [];
    if (recognized.length) {
      recognized.forEach((entry, index) => {
        const evidence = node('p', 'evidence-copy');
        evidence.append(node('small', '', `TEXT FOUND IN PHOTO ${index + 1}`));
        evidence.append(document.createTextNode(entry.text));
        content.append(evidence);
      });
    } else {
      content.append(node('p', 'quiet-copy', 'No readable text was returned from the photos. Text recognition does not verify vehicle identity.'));
    }
    content.append(node('p', 'quiet-copy', 'Photo text is evidence for your review. Verify registration documents in person.'));
  }

  function renderVehicleSpecs(workflow) {
    const content = $('vehicle-specs');
    if (!content) return;
    content.replaceChildren();
    const report = workflow.stages?.report?.result?.report_data || {};
    const lookupResult = report.vehicle_details_lookup || {};
    const lookup = lookupResult.details || {};
    const assessmentDetails = report.assessment?.vehicle_details || {};
    const estimates = Array.isArray(report.price_estimates) ? report.price_estimates : [];
    const estimateVehicle = estimates.at(-1)?.vehicle || {};
    const provenance = {
      ...(report.vehicle_details_provenance || {}),
      ...(lookupResult.field_provenance || {}),
      ...(estimates.at(-1)?.provenance || {}),
    };
    const specs = [
      { key: 'make', label: 'Make', source: lookup.make || estimateVehicle.make || assessmentDetails.make },
      { key: 'model', label: 'Model', source: lookup.model || estimateVehicle.model || assessmentDetails.model },
      { key: 'variant', label: 'Variant', source: lookup.variant || estimateVehicle.variant || assessmentDetails.variant },
      { key: 'registration_year', label: 'Year', source: lookup.registration_date ? String(lookup.registration_date).slice(0, 4) : (estimateVehicle.registration_year || assessmentDetails.registration_year || assessmentDetails.year) },
      { key: 'fuel_type', label: 'Fuel type', source: lookup.fuel_type || estimateVehicle.fuel_type || assessmentDetails.fuel_type },
      { key: 'engine_cc', label: 'Engine capacity · cc', source: lookup.engine_cc || estimateVehicle.engine_cc || assessmentDetails.engine_cc },
      { key: 'max_power_bhp', label: 'Maximum power · bhp', source: lookup.max_power_bhp || estimateVehicle.max_power_bhp || assessmentDetails.max_power_bhp },
      { key: 'transmission', label: 'Transmission', source: lookup.transmission || estimateVehicle.transmission || assessmentDetails.transmission },
      { key: 'seats', label: 'Seats', source: lookup.seats || estimateVehicle.seats || assessmentDetails.seats },
      { key: 'mileage_kmpl', label: 'Mileage · km/l', source: lookup.mileage_kmpl || estimateVehicle.mileage_kmpl || assessmentDetails.mileage_kmpl },
      { key: 'owner_count', label: 'Owner count', source: lookup.owner_count || estimateVehicle.owner_count || assessmentDetails.owner_count },
    ];
    const list = node('div', 'vehicle-specs-list');
    specs.forEach(spec => {
      if (spec.source === null || spec.source === undefined || spec.source === '') return;
      const item = node('div', 'vehicle-spec-item');
      const source = provenance[spec.key];
      const sourceLabel = source === 'vehicle_api' ? 'Vehicle API' : source === 'model_default' ? 'Model default' : source === 'generic_fallback' ? 'Generic fallback' : source === 'user_input' ? 'User provided — not independently verified' : 'Unknown';
      if (sourceLabel === 'Unknown') return;
      item.append(node('span', 'spec-label', spec.label), node('span', 'spec-value', String(spec.source)), node('span', 'spec-source', `(${sourceLabel})`));
      list.append(item);
    });
    if (!list.childElementCount) {
      content.append(node('p', 'quiet-copy', 'No sourced vehicle specifications are available.'));
      return;
    }
    content.append(list);
  }

  function renderCondition(workflow) {
    const content = $('condition-findings');
    content.replaceChildren();
    const stage = workflow.stages?.damage_detection || {};
    const [status, tone] = cleanStatus(stage.status);
    setResultStatus('condition-status', status, tone);
    if (sampleReportActive) {
      content.append(node('p', 'quiet-copy', 'No damage inference is included in this illustrative sample.'));
      return;
    }
    const scans = stage.result?.results || [];
    const reportData = workflow.stages?.report?.result?.report_data || {};
    const condition = reportData.condition_assessment || {};
    if (condition.category) {
      content.append(node('p', 'condition-category', `Condition evidence: ${condition.category.replace(/_/g, ' ')}`));
    }
    const findings = Array.isArray(stage.result?.findings)
      ? stage.result.findings
      : scans.flatMap(scan => scan.findings || []);
    if (!findings.length) {
      const line = node('p', 'quiet-copy', stage.status === 'completed_no_findings'
        ? 'No damage was detected in the supplied images. This does not establish that the vehicle is damage-free.'
        : stage.status === 'failed'
          ? 'Damage inspection failed before it could be completed. No conclusion about damage can be drawn from these images.'
          : 'Damage inspection was unavailable for these images. No conclusion about damage can be drawn.');
      const emptyFindings = node('div', 'empty-findings');
      emptyFindings.append(node('span', '', '✦'), line);
      content.append(emptyFindings);
      return;
    }
    const list = node('div', 'finding-list');
    const evidenceImages = new Set();
    findings.forEach(finding => {
      const item = node('div', 'finding-item');
      const label = node('div', '');
      label.append(node('strong', '', finding.damage_type || 'Visible finding'));
      if (finding.component) label.append(node('small', '', finding.component));
      const confidence = Number.isFinite(finding.confidence)
        ? `${(finding.confidence * 100).toFixed(1)}% model confidence`
        : 'Confidence unavailable';
      item.append(label, node('span', 'confidence', confidence));
      if (finding.description) item.append(node('p', 'finding-description', finding.description));
      const photoIndex = imageIds.indexOf(finding.image_id);
      if (photoIndex >= 0 && !evidenceImages.has(finding.image_id)) {
        evidenceImages.add(finding.image_id);
        const evidencePhoto = document.createElement('img');
        evidencePhoto.className = 'finding-evidence';
        const preview = previewUrls[photoIndex];
        if (preview) evidencePhoto.src = preview;
        else evidencePhoto.dataset.imageId = finding.image_id;
        evidencePhoto.hidden = !preview;
        evidencePhoto.alt = `Submitted photo ${photoIndex + 1} associated with this finding`;
        item.append(evidencePhoto);
      }
      list.append(item);
    });
    content.append(list);
    content.append(node('p', 'quiet-copy', 'Findings are model outputs, not a substitute for an in-person inspection.'));
  }

  function pricingNeedsFeatures(stage) {
    const unresolved = [
      ...(stage?.result?.missing_fields || []),
      ...(stage?.result?.invalid_fields || []),
    ];
    return ['odometer_km', 'seller_type', 'transmission', 'mileage_kmpl', 'max_power_bhp']
      .some(field => unresolved.includes(field));
  }

  function missingPricingLabels(stage) {
    const names = [
      ...(stage?.result?.missing_fields || []),
      ...(stage?.result?.invalid_fields || []),
    ];
    const labels = {
      odometer_km: 'odometer reading',
      seller_type: 'seller type',
      make: 'make',
      model: 'model',
      registration_year: 'registration year',
      fuel_type: 'fuel type',
      transmission: 'transmission',
      mileage_kmpl: 'mileage',
      engine_cc: 'engine capacity',
      max_power_bhp: 'maximum power',
      seats: 'seating capacity',
    };
    return names.map(name => labels[name] || name).filter(name => name in labels);
  }

  const pricingFieldDefinitions = {
    odometer_km: { label: 'Odometer · km', type: 'number', min: 0, max: 2000000, step: 1, placeholder: '' },
    seller_type: { label: 'Seller type', options: [['Individual', 'Individual'], ['Dealer', 'Dealer'], ['Trustmark Dealer', 'Trustmark Dealer']] },
    transmission: { label: 'Transmission type', options: [['manual', 'Manual'], ['automatic', 'Automatic'], ['amt', 'AMT'], ['cvt', 'CVT'], ['dct', 'DCT']] },
    mileage_kmpl: { label: 'Mileage · km/l', type: 'number', min: 0.1, max: 100, step: 'any', placeholder: '' },
    max_power_bhp: { label: 'Maximum power · bhp', type: 'number', min: 0.1, max: 2000, step: 'any', placeholder: '' },
  };

  const fuelMap = { Petrol: 'petrol', Diesel: 'diesel', CNG: 'cng', LPG: 'lpg', Electric: 'electric' };
  const transmissionMap = {
    Manual: 'manual', Automatic: 'automatic', AMT: 'amt', CVT: 'cvt', DCT: 'dct',
    manual: 'manual', automatic: 'automatic',
    amt: 'amt', cvt: 'cvt', dct: 'dct',
  };

  function reportVehiclePrefill() {
    const report = latestWorkflow?.stages?.report?.result?.report_data || {};
    const lookup = report.vehicle_details_lookup?.details || {};
    const assessmentDetails = report.assessment?.vehicle_details || {};
    const estimates = Array.isArray(report.price_estimates) ? report.price_estimates : [];
    const estimateVehicle = estimates.at(-1)?.vehicle || {};
    const sources = [assessmentDetails, lookup, estimateVehicle, pricingVehicle || {}];
    const aliases = {
    make: ['make'],
    model: ['model'],
    registration_year: ['registration_year', 'year', 'manufacturing_year'],
    odometer_km: ['odometer_km', 'mileage_km'],
    fuel_type: ['fuel_type'],
    transmission: ['transmission'],
    seller_type: ['seller_type'],
    mileage_kmpl: ['mileage_kmpl'],
    engine_cc: ['engine_cc'],
    max_power_bhp: ['max_power_bhp'],
    seats: ['seats'],
    owner_count: ['owner_count'],
  };
  const values = {};
  for (const source of sources) {
    if (source && source.registration_date && !source.registration_year) {
      const parsedDate = new Date(source.registration_date);
      if (!Number.isNaN(parsedDate.getTime())) source.registration_year = parsedDate.getFullYear();
    }
  }
    for (const [field, keys] of Object.entries(aliases)) {
      for (const source of sources) {
        const key = keys.find(candidate => source[candidate] !== null && source[candidate] !== undefined && source[candidate] !== '');
        if (key) {
          let value = source[key];
          if (field === 'fuel_type') value = fuelMap[value] || value;
          if (field === 'transmission') value = transmissionMap[value] || value;
          values[field] = value;
          break;
        }
      }
    }
    return values;
  }

  function renderPricingFields(missingFields, prefill = reportVehiclePrefill()) {
    const fields = $('pricing-fields');
    fields.replaceChildren();
    for (const name of missingFields) {
      const definition = pricingFieldDefinitions[name];
      if (!definition || definition.readOnly) continue;
      const label = node('label', 'field');
      label.append(node('span', '', definition.label));
      let control;
      if (definition.options) {
        control = document.createElement('select');
        const placeholder = node('option', '', 'Select');
        placeholder.value = '';
        control.append(placeholder);
        for (const [value, text] of definition.options) {
          const option = node('option', '', text);
          option.value = value;
          control.append(option);
        }
      } else {
        control = document.createElement('input');
        control.type = definition.type;
        if (definition.maxLength) control.maxLength = definition.maxLength;
        if (definition.min !== undefined) control.min = String(definition.min);
        if (definition.max !== undefined) control.max = String(definition.max);
        if (definition.step !== undefined) control.step = String(definition.step);
        control.placeholder = definition.placeholder || '';
      }
      control.name = name;
      control.required = true;
      control.value = prefill[name] === undefined ? '' : String(prefill[name]);
      control.dataset.originalValue = control.value;
      if (name === 'max_power_bhp') label.append(node('small', 'field-help', 'Refer to official vehicle documents or manufacturer specifications rather than guess.'));
      if (['transmission', 'mileage_kmpl', 'max_power_bhp'].includes(name)) {
        label.append(node('small', 'field-help', 'Prefilled from available vehicle data. Change only to correct the value.'));
      }
      label.append(control);
      fields.append(label);
    }
  }

  function renderPrice(workflow) {
    const content = $('price-content');
    const prompt = $('pricing-details-prompt');
    content.replaceChildren();
    prompt.hidden = true;
    const stage = workflow.stages?.pricing || {};
    const report = workflow.stages?.report?.result?.report_data;
    const decision = report?.pricing_decision || {};
    const [status, tone] = cleanStatus(stage.status);
    setResultStatus('price-status', status, tone);
    const prediction = stage.result;
    if (decision.pricing_eligible !== true) {
      content.append(node('p', 'valuation-withheld', decision.withholding_message || 'A reliable condition-adjusted resale estimate is unavailable.'));
      if (decision.condition_category) content.append(node('p', 'quiet-copy', `Condition status: ${String(decision.condition_category).replace(/_/g, ' ')}. The base model prediction is retained for audit and is not shown as a resale estimate.`));
      if (decision.human_review_required) content.append(node('p', 'quiet-copy', 'A qualified in-person inspection is recommended before relying on a resale valuation.'));
    } else if (prediction && stage.status === 'available') {
      const highlight = node('div', 'price-highlight');
      highlight.append(node('small', '', 'RESEARCH MODEL ESTIMATE · INR'));
      highlight.append(node('strong', '', money(prediction.amount)));
      highlight.append(node('p', '', prediction.model_name || 'Research-only model'));
      content.append(highlight);
    } else {
      content.append(node('p', 'quiet-copy', pricingNeedsFeatures(stage)
        ? `A price estimate can be prepared after you provide or correct: ${missingPricingLabels(stage).join(', ')}.`
        : 'No model estimate is available. Check the required vehicle specifications and provide corrections where needed.'));
      const missing = missingPricingLabels(stage);
      if (missing.length) content.append(node('p', 'quiet-copy', `Still needed: ${missing.join(', ')}.`));
    }
    const estimates = report?.price_estimates || [];
    const inputSources = estimates.at(-1)?.provenance || stage.result?.provenance || stage.result?.input_provenance || {};
    const sourceEntries = Object.entries(inputSources);
    if (sourceEntries.length) {
      const sourceLabels = sourceEntries.map(([field, source]) => `${field}: ${source}`);
      content.append(node('p', 'quiet-copy', `Pricing input sources: ${sourceLabels.join('; ')}.`));
    }

    const recommendation = report?.buyer_recommendation || {};
    const sellerAsking = recommendation.asking_price_inr;
    if (sellerAsking) {
      const asking = node('div', 'asking-row');
      asking.append(node('span', '', "Seller's asking price"), node('strong', '', money(sellerAsking)));
      content.append(asking);
      content.append(node('p', 'quiet-copy', 'Seller asking price source: user_input.'));
    }
    if (recommendation.absolute_difference_inr !== null && recommendation.absolute_difference_inr !== undefined) {
      const difference = node('div', 'price-difference');
      const percentage = recommendation.difference_percent_of_ml_estimate;
      const percentageNumber = Number(percentage);
      const percentageLabel = percentage !== null && percentage !== undefined && Number.isFinite(percentageNumber)
        ? ` · ${percentageNumber.toFixed(2)}%`
        : '';
      difference.append(
        node('span', '', 'Difference from model estimate'),
        node('strong', '', `${money(recommendation.absolute_difference_inr)}${percentageLabel}`),
      );
      content.append(difference);
      content.append(node(
        'p',
        'price-disclaimer',
        'This comparison is indicative and does not guarantee the final negotiated price.',
      ));
    }
    content.append(node(
      'p',
      'price-disclaimer',
      'Figures are estimates and may differ from actual outcomes. The model estimate is not verified market value or a guaranteed sale price.',
    ));
    $('pricing-form').hidden = true;
    $('edit-pricing-details').hidden = false;
    $('pricing-fields').replaceChildren();
    setMessage($('pricing-message'), '');
    prompt.hidden = false;
  }

  function renderLLMNarrative(workflow) {
    const report = workflow?.stages?.report?.result?.report_data || {};
    const narrative = report.llm_narrative || report.llmNarrative;
    const container = $('report-shell');
    if (!container || !narrative) return;
    container.querySelector('section.llm-narrative')?.remove();
    const section = node('section', 'report-section llm-narrative');
    section.append(node('h3', '', 'AI-generated inspection narrative'));
    section.append(node('p', 'quiet-copy', 'AI-generated explanation based only on the structured assessment evidence. This is not a professional mechanical inspection.'));
    const labels = {
      vehicle_overview: 'Vehicle overview',
      overall_condition_summary: 'Overall condition',
      detected_damage_explanation: 'Detected damage',
      potential_inspection_risks: 'Potential risks',
      buyer_recommendations: 'Buyer recommendations',
      price_estimate_explanation: 'Price estimate',
      missing_information_and_limitations: 'Limitations',
      vehicle_identity_and_lookup_summary: 'Vehicle identity and lookup evidence',
      market_valuation_explanation: 'Market research',
    };
    Object.entries(labels).forEach(([key, label]) => {
      if (!narrative[key]) return;
      const block = node('div', 'narrative-block');
      block.append(node('h4', '', label));
      String(narrative[key]).split(/\n+/).forEach(line => {
        const trimmed = line.trim();
        if (!trimmed) return;
        if (/^[-*]\s+/.test(trimmed)) {
          const item = node('li', '', trimmed.replace(/^[-*]\s+/, ''));
          let list = block.querySelector('ul.narrative-list');
          if (!list) {
            list = node('ul', 'narrative-list');
            block.append(list);
          }
          list.append(item);
        } else {
          block.append(node('p', '', trimmed));
        }
      });
      section.append(block);
    });
    container.append(section);
  }


  function renderRecommendation(workflow) {
    const content = $('recommendation-content');
    content.replaceChildren();
    const reportStage = workflow.stages?.report;
    const report = reportStage?.result?.report_data;
    const recommendation = report?.buyer_recommendation;
    const [status, tone] = cleanStatus(reportStage?.status);
    if (reportStage) setResultStatus('report-status', status, tone);
    if (!recommendation) {
      content.append(node('p', 'quiet-copy', 'The summary could not be prepared. The results above reflect only completed checks.'));
      return;
    }
    const box = node('div', 'recommendation-box');
    box.dataset.kind = recommendation.status || 'insufficient_evidence';
    box.append(node('h4', '', (recommendation.status || 'insufficient_evidence').replace(/_/g, ' ')));
    box.append(node('p', '', recommendation.explanation || 'No recommendation explanation was returned.'));
    if (Array.isArray(recommendation.next_steps) && recommendation.next_steps.length) {
      const list = node('ul', '');
      recommendation.next_steps.forEach(step => list.append(node('li', '', step)));
      box.append(list);
    }
    content.append(box);
  }

  function renderMarket(workflow) {
    const overview = $('report-overview');
    overview.replaceChildren();
    const reportStage = workflow.stages?.report;
    const report = reportStage?.result?.report_data;
    $('sample-report-banner').hidden = !sampleReportActive;
    const market = report?.market_valuation || {};
    const marketListings = Array.isArray(market.market_research_results)
      ? market.market_research_results.filter(listing =>
        listing
        && listing.priced_used_listing !== false
        && listing.priced_comparable !== false
      )
      : Array.isArray(market.comparables)
        ? market.comparables.map(listing => ({
          source_url: listing.source_url,
          asking_price_inr: listing.advertised_price_inr,
          variant: listing.variant,
          year: listing.year,
        }))
        : [];
    marketListings.sort((left, right) => {
      const target = report?.assessment?.vehicle_details || {};
      const targetYear = target.registration_year || target.year || target.manufacturing_year;
      const leftExact = Number(Boolean(target.variant && String(left.variant || '').toLowerCase() === String(target.variant).toLowerCase()))
        + Number(Boolean(targetYear && Number(left.year) === Number(targetYear)));
      const rightExact = Number(Boolean(target.variant && String(right.variant || '').toLowerCase() === String(target.variant).toLowerCase()))
        + Number(Boolean(targetYear && Number(right.year) === Number(targetYear)));
      return rightExact - leftExact;
    });
    const recommendation = report?.buyer_recommendation || {};
    const decision = report?.pricing_decision || {};
    const estimate = decision.pricing_eligible === true
      ? (decision.customer_facing_estimate_inr || recommendation.ml_estimate_inr)
      : null;
    const estimateTile = node('section', 'report-price-hero');
    estimateTile.setAttribute('aria-label', 'Estimated resale price');
    estimateTile.append(node('p', 'report-price-label', decision.pricing_eligible === true ? 'ESTIMATED RESALE PRICE' : 'RESALE VALUATION WITHHELD'));
    estimateTile.append(node(
      'strong',
      'report-price-range',
      resalePriceRange(estimate) || 'Not available',
    ));
    estimateTile.append(node(
      'p',
      'report-price-method',
      decision.pricing_eligible !== true
        ? (decision.withholding_message || 'A condition-adjusted resale estimate is not available.')
        : 'Rounded to the nearest ₹10,000 band from the research model estimate; not a confidence interval.',
    ));
    estimateTile.append(node(
      'p',
      'report-price-disclaimer',
      'Actual selling price may vary depending on vehicle condition, location, ownership and market demand.',
    ));
    overview.append(estimateTile);

    const shell = $('report-shell');
    const submittedRegistration = document.getElementById('registration-number').value.trim();
    const reportData = report || {};
    const registration = submittedRegistration || reportData.assessment?.registration_number;
    $('report-subtitle').textContent = sampleReportActive
      ? 'Synthetic demonstration content only · no real photos or vehicle assessment.'
      : registration
      ? `Registration reference: ${registration}. Check all details against the original documents.`
      : `${imageIds.length} photo${imageIds.length === 1 ? '' : 's'} reviewed. Vehicle identity details were not provided.`;
    renderVehicle(workflow, submittedRegistration);
    renderVehicleSpecs(workflow);
    renderCondition(workflow);
    renderPrice(workflow);
    renderRecommendation(workflow);
    renderLLMNarrative(workflow);

    const marketRegion = document.createElement('section');
    marketRegion.className = 'market-unavailable';
    const marketCopy = node('div', '');
    const listingList = node('ul', 'market-listings');
    marketListings.forEach(listing => {
      const label = marketPriceLabel(
        listing.asking_price_inr || listing.advertised_price_inr
      );
      const sourceLink = marketSourceLink(listing.source_url);
      if (!label || !sourceLink) return;
      const entry = node('li', 'market-listing');
      entry.append(node('span', 'market-listing-price', label));
      const variant = typeof listing.variant === 'string' && listing.variant.trim()
        ? listing.variant.trim()
        : 'unavailable';
      const year = Number.isInteger(listing.year) ? String(listing.year) : 'unavailable';
      entry.append(node('small', '', `Variant: ${variant} · Year: ${year}`));
      entry.append(node('span', '', ' — '), sourceLink);
      listingList.append(entry);
    });
    if (listingList.childElementCount) {
      marketCopy.append(listingList);
    } else {
      marketCopy.append(node('p', '', 'No sourced used-car listings found.'));
    }
    marketRegion.append(marketCopy);
    const priceCard = document.querySelector('.valuation-card');
    priceCard.querySelector('.market-unavailable')?.remove();
    priceCard.append(marketRegion);
    shell.hidden = false;
  }

  function workflowFromReport(reportResponse) {
    const data = reportResponse?.report_data || {};
    const sourceStages = data.stages || {};
    const sourceDamage = sourceStages.damage_detection || {};
    const sourceDamageStatus = sourceDamage.status || 'unavailable';
    const reportFindings = Array.isArray(data.damage_findings) ? data.damage_findings : [];
    const legacyDamageStatus = sourceDamageStatus === 'results_available'
      ? (reportFindings.length ? 'completed_with_findings' : 'unavailable')
      : sourceDamageStatus === 'completed_no_findings'
        ? 'completed_no_findings'
        : sourceDamageStatus === 'failed'
          ? 'failed'
          : sourceDamageStatus === 'completed_with_findings'
            ? (reportFindings.length ? 'completed_with_findings' : 'unavailable')
            : 'unavailable';
    const latestEstimate = Array.isArray(data.price_estimates) ? data.price_estimates.at(-1) : null;
    const pricingDecision = data.pricing_decision || {};
    const savedPricing = sourceStages.pricing || {};
    const missingFields = Array.isArray(savedPricing.missing_fields) ? savedPricing.missing_fields : [];
    const savedPricingStatus = savedPricing.status;
    const pricingStatus = savedPricingStatus === 'available'
      ? (latestEstimate ? 'available' : 'unavailable')
      : savedPricingStatus === 'failed' ? 'failed'
        : savedPricingStatus === 'skipped' || savedPricingStatus === 'unavailable' ? 'unavailable'
          : latestEstimate ? 'available' : 'unavailable';
    const pricingErrorCode = savedPricing.error_code || (
      savedPricingStatus === 'skipped' ? 'missing_vehicle_attributes' : null
    );
    const stages = {
      ocr: { status: sourceStages.ocr?.status || 'unavailable', result: data.ocr_results || [] },
      damage_detection: {
        status: legacyDamageStatus,
        result: { ...sourceDamage, findings: reportFindings },
      },
      pricing: {
        status: pricingStatus,
        result: pricingStatus === 'available'
          ? { amount: pricingDecision.pricing_eligible === true ? (pricingDecision.customer_facing_estimate_inr || latestEstimate.amount) : null, model_name: latestEstimate.model_name }
          : { missing_fields: missingFields },
        errors: pricingErrorCode ? [{ code: pricingErrorCode }] : [],
      },
      report: { status: 'available', result: reportResponse },
    };
    const stageStatuses = [stages.ocr.status, stages.damage_detection.status, stages.pricing.status];
    const isComplete = value => value === 'available'
      || value === 'completed_with_findings'
      || value === 'completed_no_findings';
    const availableCount = stageStatuses.filter(isComplete).length;
    const allAvailable = stageStatuses.every(isComplete);
    return {
      assessment_id: reportResponse.assessment_id,
      status: allAvailable ? 'completed' : availableCount ? 'partial' : 'failed',
      stages,
    };
  }

  function illustrativeSampleReport() {
    const vehicle = {
      make: 'Maruti Suzuki', model: 'Swift', variant: 'VXi', registration_year: 2021,
      odometer_km: 42000, fuel_type: 'petrol', transmission: 'manual', seller_type: 'Individual',
      mileage_kmpl: 21.5, engine_cc: 1197, max_power_bhp: 88.5, seats: 5, owner_count: 1,
    };
    return {
      id: 'illustrative-sample',
      assessment_id: 'illustrative-sample',
      status: 'draft',
      summary: 'Illustrative synthetic sample; not a real assessment.',
      generated_at: null,
      report_data: {
        assessment: {
          id: 'illustrative-sample', status: 'draft', registration_number: null,
          vehicle_details: vehicle, asking_price_inr: null,
        },
        images: [],
        stages: {
          ocr: { status: 'unavailable', result_count: 0 },
          damage_detection: { status: 'unavailable', finding_count: 0 },
          pricing: { status: 'available', estimate_count: 1 },
        },
        condition_assessment: {
          category: 'inspection_unavailable', inspection_status: 'unavailable',
          requires_human_review: true, finding_count: 0,
        },
        pricing_decision: {
          status: 'withheld_inspection_unavailable', pricing_eligible: false,
          customer_facing_estimate_inr: null,
          withholding_message: 'Condition assessment unavailable. A reliable condition-adjusted resale estimate cannot be provided.',
          condition_category: 'inspection_unavailable', human_review_required: true,
        },
        ocr_results: [],
        registration_lookup_results: [],
        vehicle_details_lookup: { status: 'unavailable', details: null },
        damage_findings: [],
        price_estimates: [{
          id: 'illustrative-estimate', amount: '524629.88', currency: 'INR',
          method: 'xgboost_pipeline_log1p_expm1',
          model_name: 'Vehicle IQ research model · synthetic input example',
          vehicle,
        }],
        market_valuation: {
          status: 'unavailable', evidence_count: 0, market_asking_price_range: null,
          limitations: ['No comparable listings are included in this synthetic sample.'],
        },
        buyer_recommendation: {
          status: 'insufficient_evidence', asking_price_inr: null,
          ml_estimate_inr: '524629.88', absolute_difference_inr: null,
          difference_percent_of_ml_estimate: null, market_evidence_available: false,
          explanation: 'This synthetic sample illustrates a research-model estimate separately from unavailable market evidence. No seller asking price is supplied.',
          next_steps: ['Use the sample only to understand report layout; submit a real assessment for vehicle-specific results.'],
          limitations: ['All vehicle details are synthetic demonstration inputs.', 'The model estimate is not verified market value.'],
        },
      },
    };
  }

  function loadSampleReport() {
    sampleReportActive = true;
    assessmentId = null;
    imageIds = [];
    pricingVehicle = null;
    workflowAssessmentId = 'illustrative-sample';
    latestWorkflow = workflowFromReport(illustrativeSampleReport());
    renderMarket(latestWorkflow);
    $('report-load-note').hidden = true;
    $('report-shell').focus({ preventScroll: true });
  }

  function routeFor(hash) {
    const path = hash.replace(/^#/, '') || '/';
    if (path === '/' || path === '/dashboard' || path === '/dashboard/') return { page: 'landing' };
    if (path === '/inspect') return { page: 'inspect' };
    if (path === '/how-it-works') return { page: 'how' };
    const inspection = path.match(/^\/inspection\/([^/]+)\/?$/);
    if (inspection) return { page: 'inspection', id: decodeURIComponent(inspection[1]) };
    if (path === '/sample-report' || path === '/sample-report/') return { page: 'report', id: 'sample', sample: true };
    const report = path.match(/^\/report\/([^/]+)\/?$/);
    if (report) return { page: 'report', id: decodeURIComponent(report[1]) };
    return { page: 'error', invalid: true };
  }

  function setPageError(title, message, options = {}) {
    showRouteError(title, message, options);
  }

  async function loadPersistedReport(id, sequence) {
    sampleReportActive = false;
    assessmentId = id;
    const assessment = await get(`/${encodeURIComponent(id)}`);
    const savedReport = await get(`/${encodeURIComponent(id)}/report`, { allow404: true });
    if (sequence !== routeLoadSequence) return;
    if (savedReport) {
      latestWorkflow = workflowFromReport(savedReport);
      workflowAssessmentId = id;
      imageIds = (savedReport.report_data?.images || []).map(image => image.id);
      pricingVehicle = null;
      renderMarket(latestWorkflow);
      await loadPersistedEvidence(id);
      $('report-load-note').hidden = true;
      $('report-shell').focus({ preventScroll: true });
      return;
    }
    if (latestWorkflow && workflowAssessmentId === id) {
      renderMarket(latestWorkflow);
      $('report-load-note').textContent = 'A saved report is not available. These results are from the current review session only.';
      $('report-load-note').hidden = false;
      return;
    }
    setPageError('Your report is not ready yet.', 'We could not find a saved review for this car. Continue the inspection to prepare one.', { continuePath: `/inspection/${encodeURIComponent(assessment.id)}` });
  }

  async function loadPersistedEvidence(id) {
    const images = [...document.querySelectorAll('.finding-evidence[data-image-id]')];
    await Promise.all(images.map(async image => {
      const content = await getImageContent(id, image.dataset.imageId);
      if (!content || !image.isConnected) return;
      const url = URL.createObjectURL(content);
      previewUrls.push(url);
      image.src = url;
      image.hidden = false;
    }));
  }

  async function resumeInspection(id, sequence, justUploadedImageIds = null, vehicle = null, registrationPrefill = '') {
    assessmentId = id;
    showOverlay('Getting your review ready');
    let manualRegistrationAlreadySaved = false;
    try {
      if (Array.isArray(justUploadedImageIds) && justUploadedImageIds.length > 0) {
        imageIds = justUploadedImageIds;
      } else {
        const savedReport = await get(`/${encodeURIComponent(id)}/report`, { allow404: true });
        if (sequence !== routeLoadSequence) return;
        if (savedReport) {
          await closeOverlay({ complete: true, message: 'Your saved review is ready' });
          if (sequence === routeLoadSequence) {
            location.hash = `/report/${encodeURIComponent(id)}`;
          }
          return;
        }

        const assessment = await get(`/${encodeURIComponent(id)}`);
        manualRegistrationAlreadySaved = assessment.vehicle_details?.registration_number_source === 'manual';
        if (!registrationPrefill && manualRegistrationAlreadySaved) {
          registrationPrefill = assessment.registration_number || '';
        }
        const images = await get(`/${encodeURIComponent(id)}/images`);
        if (sequence !== routeLoadSequence) return;
        if (!Array.isArray(images) || images.length < 1) {
          await closeOverlay({ message: 'No photos were found for this review.' });
          setPageError('No photos to review.', 'This inspection does not have any uploaded photos. Start a new review and add at least one image.');
          return;
        }
        imageIds = images.map(image => image.id);
      }
      updateOverlay('Taking a closer look at your photos');
      const workflowPayload = { image_ids: imageIds };
      const vehicleDetails = vehicle || pricingVehicle;
      if (vehicleDetails && Object.keys(vehicleDetails).length) {
        workflowPayload.vehicle = vehicleDetails;
      }
      const workflow = await post(`/${encodeURIComponent(id)}/workflow`, {
        json: workflowPayload,
        timeoutMs: 180_000,
      });
      if (sequence !== routeLoadSequence) return;
      latestWorkflow = workflow;
      workflowAssessmentId = id;
      if (!manualRegistrationAlreadySaved && !workflowHasValidOcrRegistration(workflow)) {
        await closeOverlay();
        if (sequence === routeLoadSequence) showRegistrationDialog(registrationPrefill);
        return;
      }
      const complete = workflow.status === 'completed' || workflow.status === 'partial';
      await closeOverlay({ complete, message: complete ? 'Your review is ready' : 'Some review steps could not be completed' });
      if (sequence === routeLoadSequence) await navigate(`/report/${encodeURIComponent(id)}`);
    } catch (error) {
      if (sequence !== routeLoadSequence) return;
      await closeOverlay({ message: 'We could not continue this review' });
      setPageError(
        error?.status === 401 ? 'This review is not available in the current demo configuration.' : 'We could not continue this review.',
        error instanceof Error ? error.message : 'Check the link and try again.',
        {
          continuePath: error?.status === 401 || error?.status === 404 ? '' : `/inspection/${encodeURIComponent(id)}`,
        },
      );
    }
  }

  async function loadRoute(options = {}) {
    const sequence = ++routeLoadSequence;
    const route = routeFor(window.location.hash);
    if (route.page !== 'inspection' && registrationDialog.open) registrationDialog.close();
    if (route.page !== 'inspection' && !overlay.hidden) {
      overlay.hidden = true;
      overlay.setAttribute('aria-hidden', 'true');
      overlay.classList.remove('complete', 'error', 'closing');
      document.body.classList.remove('modal-open');
    }
    showPage(route.page);
    if (!route.sample) sampleReportActive = false;
    if (route.page === 'error') {
      setPageError('That page could not be found.', 'Check the address or return home to start a new inspection.');
      return;
    }
    if (route.page === 'inspection') {
      await resumeInspection(
        route.id,
        sequence,
        options.imageIds || null,
        options.vehicle || null,
        options.registrationPrefill || '',
      );
      return;
    }
    if (route.sample) {
      loadSampleReport();
      return;
    }
    if (route.page === 'report') {
      try {
        await loadPersistedReport(route.id, sequence);
      } catch (error) {
        if (sequence !== routeLoadSequence) return;
        setPageError(
          error?.status === 401 ? 'This report is not available in the current demo configuration.' : 'We could not load this report.',
          error instanceof Error ? error.message : 'Check the link and try again.',
          {
            continuePath: error?.status && error.status < 500 ? '' : `/report/${encodeURIComponent(route.id)}`,
          },
        );
      }
    }
  }

  const registrationDialog = $('registration-dialog');
  const registrationForm = $('registration-form');
  let registrationSubmissionInProgress = false;

  function showRegistrationDialog(prefill = '') {
    $('manual-registration').value = prefill;
    setMessage($('registration-message'), '');
    if (!registrationDialog.open) registrationDialog.showModal();
    document.body.classList.add('modal-open');
    $('manual-registration').focus({ preventScroll: true });
  }

  registrationDialog.addEventListener('cancel', event => event.preventDefault());

  registrationForm.addEventListener('submit', async event => {
    event.preventDefault();
    if (registrationSubmissionInProgress) return;
    const input = $('manual-registration');
    const registrationNumber = normalizeRegistration(input.value);
    input.value = registrationNumber;
    if (!isSupportedRegistration(registrationNumber)) {
      setMessage($('registration-message'), 'Enter a valid Indian registration number, such as RJ14AB1234.');
      input.focus();
      return;
    }

    const button = registrationForm.querySelector('[type="submit"]');
    registrationSubmissionInProgress = true;
    button.disabled = true;
    input.disabled = true;
    setMessage($('registration-message'), '');
    const id = assessmentId;
    const sequence = routeLoadSequence;
    try {
      await post(`/${encodeURIComponent(id)}/registration-number`, {
        json: { registration_number: registrationNumber },
      });
      registrationDialog.close();
      document.body.classList.remove('modal-open');
      showOverlay('Checking registration details');
      const workflow = await post(`/${encodeURIComponent(id)}/workflow`, {
        json: { image_ids: imageIds, ...(pricingVehicle ? { vehicle: pricingVehicle } : {}) },
        timeoutMs: 180_000,
      });
      if (sequence !== routeLoadSequence) return;
      latestWorkflow = workflow;
      workflowAssessmentId = id;
      if (workflow.stages?.report?.status !== 'available') {
        throw requestError('The registration was saved, but the report could not be refreshed. Try continuing again.', 502);
      }
      await closeOverlay({ complete: true, message: 'Your review is ready' });
      await navigate(`/report/${encodeURIComponent(id)}`);
    } catch (error) {
      if (sequence !== routeLoadSequence) return;
      if (!overlay.hidden) await closeOverlay({ message: 'We could not complete the registration check' });
      document.body.classList.remove('modal-open');
      showRegistrationDialog(registrationNumber);
      setMessage(
        $('registration-message'),
        error instanceof Error
          ? `${error.message} You can retry; your assessment and photos are kept.`
          : 'We could not complete the registration check. Please try again.',
      );
    } finally {
      registrationSubmissionInProgress = false;
      button.disabled = false;
      input.disabled = false;
    }
  });

  let skipNextHashChange = '';

  function navigate(path, options = {}) {
    const nextHash = `#${path}`;
    if (location.hash === nextHash) return loadRoute(options);
    skipNextHashChange = nextHash;
    location.hash = path;
    return loadRoute(options);
  }

  function handleHashChange() {
    if (location.hash === skipNextHashChange) {
      skipNextHashChange = '';
      return;
    }
    skipNextHashChange = '';
    loadRoute();
  }

  function openPricingEditor() {
    const pricing = latestWorkflow?.stages?.pricing || {};
    const missingFields = [
      ...(pricing.result?.missing_fields || []),
      ...(pricing.result?.invalid_fields || []),
    ];
    renderPricingFields([...new Set([
      ...missingFields,
      'transmission',
      'mileage_kmpl',
      'max_power_bhp',
    ])]);
    setMessage($('pricing-message'), '');
    $('pricing-form').hidden = false;
    $('edit-pricing-details').hidden = true;
    fieldsFocus();
  }
  function fieldsFocus() { $('pricing-fields').querySelector('input,select')?.focus({ preventScroll: true }); }
  $('edit-pricing-details').addEventListener('click', openPricingEditor);
  $('cancel-pricing').addEventListener('click', () => {
    $('pricing-form').hidden = true;
    $('edit-pricing-details').hidden = false;
    setMessage($('pricing-message'), '');
  });

  function readVehicleForm(targetForm) {
    const vehicle = {};
    const controls = targetForm.querySelectorAll('[name]');
    controls.forEach(control => {
      if (!Object.hasOwn(pricingFieldDefinitions, control.name) || !control.value) return;
      const value = control.type === 'number' ? Number(control.value) : control.value.trim();
      if (String(value) === control.dataset.originalValue) return;
      vehicle[control.name] = value;
    });
    return vehicle;
  }

  let pricingSubmissionInProgress = false;
  $('pricing-form').addEventListener('submit', async event => {
    event.preventDefault();
    const pricingForm = event.currentTarget;
    if (pricingSubmissionInProgress) return;
    if (!pricingForm.reportValidity()) {
      setMessage($('pricing-message'), 'Complete each required field with a valid value before continuing.');
      pricingForm.querySelector(':invalid')?.focus();
      return;
    }
    setMessage($('pricing-message'), '');
    const button = pricingForm.querySelector('[type="submit"]');
    pricingSubmissionInProgress = true;
    button.disabled = true;
    pricingForm.hidden = true;
    showOverlay('Preparing the price perspective');
    try {
      pricingVehicle = readVehicleForm(pricingForm);
      updateOverlay('Updating your inspection report');
      const workflow = await post(`/${encodeURIComponent(assessmentId)}/workflow`, {
        json: { image_ids: imageIds, vehicle: pricingVehicle },
        timeoutMs: 180_000,
      });
      latestWorkflow = workflow;
      workflowAssessmentId = assessmentId;
      const pricing = workflow.stages?.pricing || {};
      const reportStage = workflow.stages?.report || {};
      if (pricing.status === 'available' && reportStage.status === 'available') {
        await closeOverlay({ complete: true, message: 'Your updated review is ready' });
        await navigate(`/report/${encodeURIComponent(assessmentId)}`);
        return;
      }

      renderMarket(workflow);
      const missingFields = [
        ...(pricing.result?.missing_fields || []),
        ...(pricing.result?.invalid_fields || []),
      ];
      const editableMissingFields = missingFields.filter(name => pricingFieldDefinitions[name] && !pricingFieldDefinitions[name].readOnly);
      const unmappedMissingFields = missingFields.filter(name => !editableMissingFields.includes(name));
      if (editableMissingFields.length) renderPricingFields(editableMissingFields, { ...reportVehiclePrefill(), ...pricingVehicle });
      const codes = (pricing.errors || []).map(error => error.code);
      const missingLabels = missingFields.map(name => pricingFieldDefinitions[name]?.label || name);
      const unmappedLabels = unmappedMissingFields.map(name => pricingFieldDefinitions[name]?.label || name);
      const message = codes.includes('missing_vehicle_attributes')
        ? `Still needed: ${missingLabels.join(', ')}.${
          unmappedLabels.length
            ? ` These details cannot be corrected here: ${unmappedLabels.join(', ')}.`
            : ''
        }`
        : codes.includes('invalid_vehicle_attributes')
          ? `Correct these vehicle details before retrying: ${missingLabels.join(', ')}.${
            unmappedLabels.length
              ? ` These details cannot be corrected here: ${unmappedLabels.join(', ')}.`
              : ''
          }`
          : pricing.status === 'failed'
            ? 'The model could not prepare an estimate. Your other completed inspection results are preserved.'
            : reportStage.status !== 'available'
              ? 'The price step finished, but the saved report could not be refreshed. Please try again.'
              : 'A price estimate is still unavailable. No price has been added.';
      await closeOverlay({ message: 'The review was updated with available results' });
      pricingForm.hidden = false;
      $('edit-pricing-details').hidden = true;
      setMessage($('pricing-message'), message);
      fieldsFocus();
    } catch (error) {
      await closeOverlay({ message: 'The estimate could not be added' });
      pricingForm.hidden = false;
      $('edit-pricing-details').hidden = true;
      setMessage($('pricing-message'), error instanceof Error ? error.message : 'We could not complete the estimate. Check the details and try again.');
      $('pricing-message').focus();
    } finally {
      button.disabled = false;
      pricingSubmissionInProgress = false;
    }
  });

  let submissionInProgress = false;
  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (submissionInProgress) return;
    setMessage($('form-message'), '');
    if (photos.length < 1 || photos.length > 10) {
      setMessage($('form-message'), 'Choose between 1 and 10 car photos to continue.');
      photoInput.focus();
      return;
    }
    const askingInput = $('asking-price');
    if (askingInput.value && !askingInput.checkValidity()) {
      setMessage($('form-message'), 'Enter a positive asking price up to ₹10 crore, or leave it blank.');
      askingInput.focus();
      return;
    }
    const invalidVehicleInput = [...form.querySelectorAll('[name]')].find(
      input => Object.hasOwn(pricingFieldDefinitions, input.name)
        && input.value
        && !input.checkValidity(),
    );
    if (invalidVehicleInput) {
      setMessage($('form-message'), `Enter a valid ${pricingFieldDefinitions[invalidVehicleInput.name].label.toLowerCase()}, or leave it blank.`);
      invalidVehicleInput.focus();
      return;
    }

    sampleReportActive = false;
    submissionInProgress = true;
    pricingVehicle = readVehicleForm(form);
    setBusy(true);
    showOverlay('Preparing your inspection');
    let uploadCount = 0;
    try {
      const assessmentBody = {};
      const registration = $('registration-number').value;
      const askingPrice = $('asking-price').value;
      if (askingPrice) assessmentBody.asking_price_inr = Number(askingPrice);
      const assessment = await post('', { json: assessmentBody });
      assessmentId = assessment.id;

      const uploaded = [];
      for (let i = 0; i < photos.length; i += 1) {
        updateOverlay(`Adding photo ${i + 1} of ${photos.length}`);
        const image = await post(`/${assessmentId}/images`, { file: photos[i], imageType: photoCategories[i] || 'unknown' });
        uploaded.push(image);
        uploadCount += 1;
      }
      imageIds = uploaded.map(image => image.id);
      await navigate(`/inspection/${encodeURIComponent(assessmentId)}`, {
        imageIds,
        vehicle: pricingVehicle,
        registrationPrefill: registration,
      });
    } catch (error) {
      const context = assessmentId
        ? `Your review was started${uploadCount ? ` and ${uploadCount} photo${uploadCount === 1 ? '' : 's'} uploaded` : ''}, but it could not be completed. `
        : '';
      await closeOverlay({ message: 'We could not finish this request' });
      setMessage($('form-message'), `${context}${error instanceof Error ? error.message : 'Please try again.'}`);
      if (assessmentId && uploadCount) {
        const resume = node('a', 'inline-resume', 'Continue this review');
        resume.href = `#/inspection/${encodeURIComponent(assessmentId)}`;
        $('form-message').append(document.createTextNode(' '), resume);
      }
      $('form-message').focus();
    } finally {
      setBusy(false);
      submissionInProgress = false;
    }
  });

  $('new-inspection').addEventListener('click', () => {
    photos = [];
    photoCategories = [];
    previewUrls.forEach(url => URL.revokeObjectURL(url));
    previewUrls = [];
    imageIds = [];
    assessmentId = null;
    latestWorkflow = null;
    workflowAssessmentId = null;
    pricingVehicle = null;
    sampleReportActive = false;
    form.reset();
    updatePhotoPreview();
    $('report-load-note').hidden = true;
    $('pricing-form').reset();
    $('pricing-form').hidden = true;
    $('edit-pricing-details').hidden = false;
    $('pricing-message').hidden = true;
    setMessage($('form-message'), '');
    navigate('/inspect').then(() => $('registration-number').focus({ preventScroll: true }));
  });
  window.addEventListener('hashchange', handleHashChange);
  loadRoute();
})();
