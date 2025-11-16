# Cloud Run Latency Test Script
# Energy Optimization API - Performance validation

param(
    [Parameter(Mandatory=$false)]
    [string]$ServiceUrl,
    
    [Parameter(Mandatory=$false)]
    [int]$RequestCount = 100,
    
    [Parameter(Mandatory=$false)]
    [int]$Concurrency = 1
)

# Color output functions
function Write-Success {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Green
}

function Write-Info {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Cyan
}

function Write-Warning {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Red
}

# Load .env file
$envFile = Join-Path $PSScriptRoot ".." ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}

# Get service URL if not provided
if (-not $ServiceUrl) {
    $ProjectId = $env:GCP_PROJECT_ID
    $Region = "us-central1"
    $ServiceName = "energy-optimization-api"
    
    if ($ProjectId) {
        Write-Info "Fetching service URL..."
        try {
            gcloud config set project $ProjectId 2>&1 | Out-Null
            $ServiceUrl = gcloud run services describe $ServiceName `
                --region $Region `
                --project $ProjectId `
                --format "value(status.url)" 2>&1
            
            if ($ServiceUrl) {
                Write-Success "Service URL: $ServiceUrl"
            } else {
                Write-Error "Could not retrieve service URL"
                exit 1
            }
        } catch {
            Write-Error "Failed to get service URL. Provide -ServiceUrl parameter."
            exit 1
        }
    } else {
        Write-Error "Service URL is required. Provide -ServiceUrl parameter or set GCP_PROJECT_ID in .env"
        exit 1
    }
}

Write-Info "=========================================="
Write-Info "Cloud Run Latency Test"
Write-Info "=========================================="
Write-Host ""
Write-Info "Service URL: $ServiceUrl"
Write-Info "Request Count: $RequestCount"
Write-Info "Concurrency: $Concurrency"
Write-Host ""

# Test payload
$testPayload = @{
    lagging_current_reactive_power_kvarh = 10.5
    leading_current_reactive_power_kvarh = 5.2
    co2_tco2 = 0.03
    lagging_current_power_factor = 0.85
    leading_current_power_factor = 0.90
    nsm = 36000
    week_status = "Weekday"
    day_of_week = "Monday"
    load_type = "Medium_Load"
} | ConvertTo-Json

$predictUrl = "$ServiceUrl/predict"

# Health check first
Write-Info "Running health check..."
try {
    $healthResponse = Invoke-RestMethod -Uri "$ServiceUrl/health" -Method Get -TimeoutSec 10
    if ($healthResponse.status -eq "healthy") {
        Write-Success "Health check passed"
    } else {
        Write-Warning "Health check returned: $($healthResponse.status)"
    }
} catch {
    Write-Error "Health check failed: $_"
    Write-Warning "Continuing with latency test anyway..."
}

Write-Host ""
Write-Info "Starting latency test..."
Write-Info "Making $RequestCount requests to $predictUrl"
Write-Host ""

# Arrays to store results
$latencies = @()
$successCount = 0
$failCount = 0

# Progress tracking
$progressInterval = [math]::Max(1, [math]::Floor($RequestCount / 10))

# Make requests
for ($i = 1; $i -le $RequestCount; $i++) {
    try {
        $startTime = Get-Date
        
        $response = Invoke-RestMethod -Uri $predictUrl `
            -Method Post `
            -Body $testPayload `
            -ContentType "application/json" `
            -TimeoutSec 30 `
            -ErrorAction Stop
        
        $endTime = Get-Date
        $latencyMs = ($endTime - $startTime).TotalMilliseconds
        
        $latencies += $latencyMs
        $successCount++
        
        if ($i % $progressInterval -eq 0) {
            $percent = [math]::Round(($i / $RequestCount) * 100)
            Write-Host "Progress: $percent% ($i/$RequestCount) - Last latency: $([math]::Round($latencyMs, 2))ms" -ForegroundColor Gray
        }
        
    } catch {
        $failCount++
        Write-Host "Request $i failed: $_" -ForegroundColor Red
    }
    
    # Small delay between requests if concurrency is 1
    if ($Concurrency -eq 1) {
        Start-Sleep -Milliseconds 50
    }
}

Write-Host ""
Write-Info "=========================================="
Write-Info "Test Results"
Write-Info "=========================================="
Write-Host ""

# Calculate statistics
if ($latencies.Count -gt 0) {
    $sortedLatencies = $latencies | Sort-Object
    $minLatency = $sortedLatencies[0]
    $maxLatency = $sortedLatencies[-1]
    $avgLatency = ($latencies | Measure-Object -Average).Average
    $medianLatency = $sortedLatencies[[math]::Floor($sortedLatencies.Count / 2)]
    
    # Calculate percentiles
    $p50Index = [math]::Floor($sortedLatencies.Count * 0.50)
    $p75Index = [math]::Floor($sortedLatencies.Count * 0.75)
    $p90Index = [math]::Floor($sortedLatencies.Count * 0.90)
    $p95Index = [math]::Floor($sortedLatencies.Count * 0.95)
    $p99Index = [math]::Floor($sortedLatencies.Count * 0.99)
    
    $p50 = $sortedLatencies[$p50Index]
    $p75 = $sortedLatencies[$p75Index]
    $p90 = $sortedLatencies[$p90Index]
    $p95 = $sortedLatencies[$p95Index]
    $p99 = $sortedLatencies[$p99Index]
    
    Write-Info "Request Statistics:"
    Write-Host "  Total Requests: $RequestCount"
    Write-Success "  Successful: $successCount"
    if ($failCount -gt 0) {
        Write-Error "  Failed: $failCount"
    }
    Write-Host ""
    
    Write-Info "Latency Statistics (milliseconds):"
    Write-Host "  Min:     $([math]::Round($minLatency, 2)) ms"
    Write-Host "  Max:     $([math]::Round($maxLatency, 2)) ms"
    Write-Host "  Average: $([math]::Round($avgLatency, 2)) ms"
    Write-Host "  Median:  $([math]::Round($medianLatency, 2)) ms"
    Write-Host ""
    
    Write-Info "Latency Percentiles:"
    Write-Host "  P50: $([math]::Round($p50, 2)) ms"
    Write-Host "  P75: $([math]::Round($p75, 2)) ms"
    Write-Host "  P90: $([math]::Round($p90, 2)) ms"
    Write-Success "  P95: $([math]::Round($p95, 2)) ms"
    Write-Host "  P99: $([math]::Round($p99, 2)) ms"
    Write-Host ""
    
    # Check against requirements
    $p95Seconds = $p95 / 1000
    if ($p95Seconds -lt 1.0) {
        Write-Success "P95 latency is under 1 second requirement"
    } else {
        Write-Warning "P95 latency exceeds 1 second requirement ($([math]::Round($p95Seconds, 2))s)"
    }
    
    Write-Host ""
    Write-Info "Throughput:"
    $totalTimeSeconds = ($latencies | Measure-Object -Sum).Sum / 1000
    $requestsPerSecond = $RequestCount / $totalTimeSeconds
    Write-Host "  Requests/second: $([math]::Round($requestsPerSecond, 2))"
    
} else {
    Write-Error "No successful requests to calculate statistics"
}

Write-Host ""
Write-Info "Test complete!"
