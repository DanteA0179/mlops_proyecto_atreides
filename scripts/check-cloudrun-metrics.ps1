# Cloud Run Metrics Checker
# Energy Optimization API - View service metrics

param(
    [Parameter(Mandatory=$false)]
    [string]$ProjectId = $env:GCP_PROJECT_ID,
    
    [Parameter(Mandatory=$false)]
    [string]$Region = "us-central1",
    
    [Parameter(Mandatory=$false)]
    [string]$ServiceName = "energy-optimization-api",
    
    [Parameter(Mandatory=$false)]
    [int]$Hours = 1
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
            if ($key -eq "GCP_PROJECT_ID" -and -not $ProjectId) {
                $ProjectId = $value
            }
        }
    }
}

# Validate Project ID
if (-not $ProjectId) {
    Write-Error "Project ID is required. Set GCP_PROJECT_ID in .env or pass as parameter."
    exit 1
}

Write-Info "=========================================="
Write-Info "Cloud Run Metrics Checker"
Write-Info "=========================================="
Write-Host ""
Write-Info "Project: $ProjectId"
Write-Info "Region: $Region"
Write-Info "Service: $ServiceName"
Write-Info "Time Range: Last $Hours hour(s)"
Write-Host ""

# Set project
gcloud config set project $ProjectId 2>&1 | Out-Null

# Get service details
Write-Info "Fetching service details..."
try {
    $serviceInfo = gcloud run services describe $ServiceName `
        --region $Region `
        --project $ProjectId `
        --format json 2>&1 | ConvertFrom-Json
    
    $serviceUrl = $serviceInfo.status.url
    $latestRevision = $serviceInfo.status.latestReadyRevisionName
    
    Write-Success "Service is running"
    Write-Info "  URL: $serviceUrl"
    Write-Info "  Latest Revision: $latestRevision"
    Write-Host ""
} catch {
    Write-Error "Failed to get service details: $_"
    exit 1
}

# Calculate time range
$endTime = Get-Date
$startTime = $endTime.AddHours(-$Hours)
$startTimeStr = $startTime.ToString("yyyy-MM-ddTHH:mm:ssZ")
$endTimeStr = $endTime.ToString("yyyy-MM-ddTHH:mm:ssZ")

Write-Info "Fetching metrics from $startTimeStr to $endTimeStr..."
Write-Host ""

# Request Count
Write-Info "Request Count:"
try {
    $requestMetrics = gcloud monitoring time-series list `
        --filter="resource.type=cloud_run_revision AND resource.labels.service_name=$ServiceName AND metric.type=run.googleapis.com/request_count" `
        --start-time=$startTimeStr `
        --end-time=$endTimeStr `
        --project=$ProjectId `
        --format="value(points[].value.int64Value)" 2>&1
    
    if ($requestMetrics) {
        $totalRequests = ($requestMetrics | Measure-Object -Sum).Sum
        Write-Success "  Total Requests: $totalRequests"
    } else {
        Write-Warning "  No request data available"
    }
} catch {
    Write-Warning "  Could not fetch request count"
}

# Request Latencies
Write-Info "Request Latency:"
try {
    $latencyMetrics = gcloud monitoring time-series list `
        --filter="resource.type=cloud_run_revision AND resource.labels.service_name=$ServiceName AND metric.type=run.googleapis.com/request_latencies" `
        --start-time=$startTimeStr `
        --end-time=$endTimeStr `
        --project=$ProjectId `
        --format="value(points[].value.distributionValue)" 2>&1
    
    if ($latencyMetrics) {
        Write-Success "  Latency data available"
        Write-Info "  (See Cloud Console for detailed percentiles)"
    } else {
        Write-Warning "  No latency data available"
    }
} catch {
    Write-Warning "  Could not fetch latency metrics"
}

# Container Instance Count
Write-Info "Container Instances:"
try {
    $instanceMetrics = gcloud monitoring time-series list `
        --filter="resource.type=cloud_run_revision AND resource.labels.service_name=$ServiceName AND metric.type=run.googleapis.com/container/instance_count" `
        --start-time=$startTimeStr `
        --end-time=$endTimeStr `
        --project=$ProjectId `
        --format="value(points[].value.int64Value)" 2>&1
    
    if ($instanceMetrics) {
        $maxInstances = ($instanceMetrics | Measure-Object -Maximum).Maximum
        $avgInstances = ($instanceMetrics | Measure-Object -Average).Average
        Write-Success "  Max Instances: $maxInstances"
        Write-Success "  Avg Instances: $([math]::Round($avgInstances, 2))"
    } else {
        Write-Warning "  No instance data available"
    }
} catch {
    Write-Warning "  Could not fetch instance metrics"
}

# CPU Utilization
Write-Info "CPU Utilization:"
try {
    $cpuMetrics = gcloud monitoring time-series list `
        --filter="resource.type=cloud_run_revision AND resource.labels.service_name=$ServiceName AND metric.type=run.googleapis.com/container/cpu/utilizations" `
        --start-time=$startTimeStr `
        --end-time=$endTimeStr `
        --project=$ProjectId `
        --format="value(points[].value.doubleValue)" 2>&1
    
    if ($cpuMetrics) {
        $maxCpu = ($cpuMetrics | Measure-Object -Maximum).Maximum
        $avgCpu = ($cpuMetrics | Measure-Object -Average).Average
        Write-Success "  Max CPU: $([math]::Round($maxCpu * 100, 2))%"
        Write-Success "  Avg CPU: $([math]::Round($avgCpu * 100, 2))%"
    } else {
        Write-Warning "  No CPU data available"
    }
} catch {
    Write-Warning "  Could not fetch CPU metrics"
}

# Memory Utilization
Write-Info "Memory Utilization:"
try {
    $memoryMetrics = gcloud monitoring time-series list `
        --filter="resource.type=cloud_run_revision AND resource.labels.service_name=$ServiceName AND metric.type=run.googleapis.com/container/memory/utilizations" `
        --start-time=$startTimeStr `
        --end-time=$endTimeStr `
        --project=$ProjectId `
        --format="value(points[].value.doubleValue)" 2>&1
    
    if ($memoryMetrics) {
        $maxMemory = ($memoryMetrics | Measure-Object -Maximum).Maximum
        $avgMemory = ($memoryMetrics | Measure-Object -Average).Average
        Write-Success "  Max Memory: $([math]::Round($maxMemory * 100, 2))%"
        Write-Success "  Avg Memory: $([math]::Round($avgMemory * 100, 2))%"
    } else {
        Write-Warning "  No memory data available"
    }
} catch {
    Write-Warning "  Could not fetch memory metrics"
}

Write-Host ""
Write-Info "Detailed metrics and charts available in Cloud Console:"
Write-Info "  https://console.cloud.google.com/run/detail/$Region/$ServiceName/metrics?project=$ProjectId"
Write-Host ""
Write-Info "To see metrics for a different time range, use -Hours parameter:"
Write-Info "  .\scripts\check-cloudrun-metrics.ps1 -Hours 24"
