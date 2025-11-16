# Cloud Run Logs Viewer
# Energy Optimization API - View real-time logs

param(
    [Parameter(Mandatory=$false)]
    [string]$ProjectId = $env:GCP_PROJECT_ID,
    
    [Parameter(Mandatory=$false)]
    [string]$Region = "us-central1",
    
    [Parameter(Mandatory=$false)]
    [string]$ServiceName = "energy-optimization-api",
    
    [Parameter(Mandatory=$false)]
    [int]$Limit = 100,
    
    [Parameter(Mandatory=$false)]
    [switch]$Follow,
    
    [Parameter(Mandatory=$false)]
    [string]$Filter
)

# Color output functions
function Write-Info {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Cyan
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
Write-Info "Cloud Run Logs Viewer"
Write-Info "=========================================="
Write-Host ""
Write-Info "Project: $ProjectId"
Write-Info "Region: $Region"
Write-Info "Service: $ServiceName"
if ($Filter) {
    Write-Info "Filter: $Filter"
}
Write-Host ""

# Set project
gcloud config set project $ProjectId 2>&1 | Out-Null

# Build gcloud command
$logArgs = @(
    "run", "services", "logs", "read", $ServiceName,
    "--region", $Region,
    "--project", $ProjectId,
    "--limit", $Limit,
    "--format", "table(timestamp,severity,textPayload)"
)

if ($Follow) {
    Write-Info "Following logs (Ctrl+C to stop)..."
    $logArgs += "--follow"
}

if ($Filter) {
    $logArgs += "--filter"
    $logArgs += $Filter
}

# Execute logs command
try {
    & gcloud @logArgs
} catch {
    Write-Error "Failed to retrieve logs: $_"
    exit 1
}

Write-Host ""
Write-Info "Logs display complete"
Write-Host ""
Write-Info "Common filters:"
Write-Info "  --Filter 'severity=ERROR'          # Show only errors"
Write-Info "  --Filter 'severity>=WARNING'       # Show warnings and errors"
Write-Info "  --Filter 'textPayload:predict'     # Show prediction logs"
Write-Info "  --Follow                           # Stream logs in real-time"
Write-Host ""
Write-Info "Full logs in Cloud Console:"
Write-Info "  https://console.cloud.google.com/run/detail/$Region/$ServiceName/logs?project=$ProjectId"
