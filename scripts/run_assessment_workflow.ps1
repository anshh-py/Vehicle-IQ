<#
.SYNOPSIS
Creates an assessment, uploads vehicle images, and runs the existing workflow API.

.DESCRIPTION
Requires the Vehicle IQ API to be running and reachable. Supply 1 to 10 JPEG,
PNG, or WebP image paths. An optional JSON file may contain the vehicle pricing
features directly (make, model, registration_year, odometer_km, fuel_type,
transmission, seller_type, mileage_kmpl, engine_cc, max_power_bhp, seats, and
owner_count). If omitted, pricing is reported as unavailable by the workflow.

If API_ACCESS_TOKEN exists in the current PowerShell process environment, it is
sent as a Bearer token. The token is never printed. No token is required when
the server is configured without API token authentication.

.EXAMPLE
./scripts/run_assessment_workflow.ps1 -ImagePath .\car-front.jpg,.\car-side.png

.EXAMPLE
./scripts/run_assessment_workflow.ps1 -ImagePath .\car.png `
  -VehicleJsonPath .\vehicle-features.json -RegistrationNumber RJ45CA1879 `
  -AskingPriceInr 650000
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string[]] $ImagePath,

    [ValidateNotNullOrEmpty()]
    [string] $BaseUrl = 'http://127.0.0.1:8081',

    [string] $VehicleJsonPath,
    [string] $RegistrationNumber,
    [string] $Vin,

    [ValidateRange(0.01, 100000000)]
    [decimal] $AskingPriceInr
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($ImagePath.Count -lt 1 -or $ImagePath.Count -gt 10) {
    throw 'Provide between 1 and 10 image files.'
}

try {
    $baseUri = [Uri]::new($BaseUrl.TrimEnd('/') + '/')
} catch {
    throw 'BaseUrl must be a valid absolute HTTP or HTTPS URL.'
}
if (-not $baseUri.IsAbsoluteUri -or $baseUri.Scheme -notin @('http', 'https')) {
    throw 'BaseUrl must be a valid absolute HTTP or HTTPS URL.'
}

$resolvedImages = foreach ($path in $ImagePath) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Image file not found: $path"
    }
    (Get-Item -LiteralPath $path).FullName
}

Add-Type -AssemblyName System.Net.Http
$httpClient = [System.Net.Http.HttpClient]::new()
$httpClient.Timeout = [TimeSpan]::FromMinutes(30)

if ($null -ne $env:API_ACCESS_TOKEN -and $env:API_ACCESS_TOKEN.Length -gt 0) {
    $apiToken = $env:API_ACCESS_TOKEN
} else {
    $apiToken = $null
}

$vehicleFeatures = $null
if ($PSBoundParameters.ContainsKey('VehicleJsonPath')) {
    if (-not (Test-Path -LiteralPath $VehicleJsonPath -PathType Leaf)) {
        throw "Vehicle feature JSON file not found: $VehicleJsonPath"
    }
    try {
        $vehicleFeatures = Get-Content -LiteralPath $VehicleJsonPath -Raw | ConvertFrom-Json
    } catch {
        throw 'VehicleJsonPath must contain valid JSON with vehicle pricing features.'
    }
    if ($vehicleFeatures -isnot [System.Management.Automation.PSCustomObject]) {
        throw 'VehicleJsonPath must contain a JSON object of vehicle pricing features.'
    }
}

function Invoke-VehicleIqApi {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('Get', 'Post')][string] $Method,
        [Parameter(Mandatory = $true)][string] $Path,
        [hashtable] $JsonBody,
        [hashtable] $Form
    )

    $uri = [Uri]::new($baseUri, $Path.TrimStart('/'))
    $request = $null
    try {
        $request = [System.Net.Http.HttpRequestMessage]::new(
            [System.Net.Http.HttpMethod]::new($Method),
            $uri
        )
        if ($null -ne $apiToken) {
            $request.Headers.Authorization = [System.Net.Http.Headers.AuthenticationHeaderValue]::new(
                'Bearer', $apiToken
            )
        }
        if ($null -ne $Form) {
            $multipart = [System.Net.Http.MultipartFormDataContent]::new()
            foreach ($name in $Form.Keys) {
                $file = $Form[$name]
                $stream = [System.IO.File]::OpenRead($file.FullName)
                $part = [System.Net.Http.StreamContent]::new($stream)
                $extension = [System.IO.Path]::GetExtension($file.Name).ToLowerInvariant()
                $mediaType = switch ($extension) {
                    '.jpg' { 'image/jpeg' }
                    '.jpeg' { 'image/jpeg' }
                    '.png' { 'image/png' }
                    '.webp' { 'image/webp' }
                    default { 'application/octet-stream' }
                }
                $part.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::new($mediaType)
                $multipart.Add($part, [string]$name, [string]$file.Name)
            }
            $request.Content = $multipart
        } elseif ($null -ne $JsonBody) {
            $body = ConvertTo-Json -InputObject $JsonBody -Depth 30 -Compress
            $request.Content = [System.Net.Http.StringContent]::new(
                $body, [System.Text.Encoding]::UTF8, 'application/json'
            )
        }
        $response = $httpClient.SendAsync($request).GetAwaiter().GetResult()
        $responseBody = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        if (-not $response.IsSuccessStatusCode) {
            $statusCode = [int]$response.StatusCode
            $detail = $null
            try {
                $errorBody = $responseBody | ConvertFrom-Json
                if ($errorBody.detail -is [string]) { $detail = $errorBody.detail }
            } catch { }
            if ($detail) {
                throw "Vehicle IQ API returned HTTP $statusCode for $Method $Path`: $detail"
            }
            throw "Vehicle IQ API returned HTTP $statusCode for $Method $Path."
        }
        if ([string]::IsNullOrWhiteSpace($responseBody)) { return $null }
        return $responseBody | ConvertFrom-Json
    } catch {
        if ($_.Exception.Message -like 'Vehicle IQ API returned HTTP *') {
            throw $_.Exception.Message
        }
        throw "Could not reach Vehicle IQ API for $Method $Path. Check that the server is running and BaseUrl is correct."
    } finally {
        if ($null -ne $request) { $request.Dispose() }
    }
}

$assessmentBody = @{}
if ($PSBoundParameters.ContainsKey('RegistrationNumber')) {
    $assessmentBody['registration_number'] = $RegistrationNumber
}
if ($PSBoundParameters.ContainsKey('Vin')) {
    $assessmentBody['vin'] = $Vin
}
if ($PSBoundParameters.ContainsKey('AskingPriceInr')) {
    $assessmentBody['asking_price_inr'] = $AskingPriceInr
}

try {
$assessment = Invoke-VehicleIqApi -Method Post -Path '/api/v1/assessments' -JsonBody $assessmentBody
$assessmentId = [string]$assessment.id
if ([string]::IsNullOrWhiteSpace($assessmentId)) {
    throw 'Assessment creation succeeded but the API did not return an assessment ID.'
}

$uploadedImages = @(
    foreach ($image in $resolvedImages) {
        $imageFile = Get-Item -LiteralPath $image
        Invoke-VehicleIqApi -Method Post `
            -Path "/api/v1/assessments/$assessmentId/images" `
            -Form @{ file = $imageFile }
    }
)

$workflowBody = @{
    image_ids = @($uploadedImages | ForEach-Object { [string]$_.id })
}
if ($PSBoundParameters.ContainsKey('VehicleJsonPath')) {
    $workflowBody['vehicle'] = $vehicleFeatures
}

$workflow = Invoke-VehicleIqApi -Method Post `
    -Path "/api/v1/assessments/$assessmentId/workflow" -JsonBody $workflowBody

$result = [ordered]@{
    assessment = $assessment
    uploaded_images = $uploadedImages
    workflow = $workflow
}
ConvertTo-Json -InputObject $result -Depth 100
} finally {
    $httpClient.Dispose()
}
