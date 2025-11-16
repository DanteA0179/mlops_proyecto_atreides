# Cloud Run Rollback Script
# Energy Optimization API - Rollback to previous revision

param(
    [Parameter(Mandatory=$false)]
    [string]$ProjectId = $env:GCP_PROJECT_ID,
    
    [Parameter(Mandatory=$false)]
    [string]$Region = "us-central1",
    
    [Parameter(Mandatory=$false)]
    [string]$ServiceName = "energy-optimization-api",
    
    [Parameter(Mandatory=$false)]
    [string]$RevisionName,
    
    [Parameter(Mandatory=$false)]
    [switch]$ListRevisions
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
Write-Info "Cloud Run Rollback Script"
Write-Info "=========================================="
Write-Host ""
Write-Info "Project: $ProjectId"
Write-Info "Region: $Region"
Write-Info "Service: $ServiceName"
Write-Host ""

# Set project
gcloud config set project $ProjectId 2>&1 | Out-Null

# List revisions
Write-Info "Fetching revisions..."
$revisions = gcloud run revisions list `
    --service $ServiceName `
    --region $Region `
    --project $ProjectId `
    --format "table(name,status,trafficPercent,creationTimestamp)" 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to fetch revisions. Service may not exist."
    exit 1
}

Write-Host $revisions
Write-Host ""

if ($ListRevisions) {
    Write-Info "Use -RevisionName parameter to rollback to a specific revision"
    exit 0
}

# Get current revision
$currentRevision = gcloud run services describe $ServiceName `
    --region $Region `
    --project $ProjectId `
    --format "value(status.latestReadyRevisionName)" 2>&1

Write-Info "Current revision: $currentRevision"

# If no revision specified, get previous one
if (-not $RevisionName) {
    Write-Info "No revision specified. Finding previous revision..."
    
    $allRevisions = gcloud run revisions list `
        --service $ServiceName `
        --region $Region `
        --project $ProjectId `
        --format "value(name)" `
        --sort-by "~creationTimestamp" 2>&1 | Where-Object { $_ -ne $currentRevision }
    
    if ($allRevisions -and $allRevisions.Count -gt 0) {
        $RevisionName = $allRevisions[0]
        Write-Info "Previous revision found: $RevisionName"
    } else {
        Write-Error "No previous revision found to rollback to"
        exit 1
    }
}

# Confirm rollback
Write-Warning "About to rollback from:"
Write-Warning "  Current: $currentRevision"
Write-Warning "  Target:  $RevisionName"
Write-Host ""
$confirm = Read-Host "Continue with rollback? (yes/no)"

if ($confirm -ne "yes") {
    Write-Info "Rollback cancelled"
    exit 0
}

# Perform rollback
Write-Info "Rolling back to revision: $RevisionName"
try {
    gcloud run services update-traffic $ServiceName `
        --to-revisions "$RevisionName=100" `
        --region $Region `
        --project $ProjectId
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Rollback successful!"
        
        # Get service URL
        $serviceUrl = gcloud run services describe $ServiceName `
            --region $Region `
            --project $ProjectId `
            --format "value(status.url)" 2>&1
        
        Write-Host ""
        Write-Info "Testing health endpoint..."
        Start-Sleep -Seconds 5
        
        try {
            $response = Invoke-RestMethod -Uri "$serviceUrl/health" -Method Get -TimeoutSec 10
            Write-Success "Health check passed: $($response.status)"
        } catch {
            Write-Warning "Health check failed. Service may still be starting."
        }
        
        Write-Host ""
        Write-Success "Service is now running revision: $RevisionName"
        Write-Info "Service URL: $serviceUrl"
    } else {
        Write-Error "Rollback failed"
        exit 1
    }
} catch {
    Write-Error "Rollback failed: $_"
    exit 1
}
