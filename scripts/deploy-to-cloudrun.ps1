# Cloud Run Deployment Script
# Energy Optimization API - Automated CI/CD
# This script builds the Docker image in Cloud Build and deploys to Cloud Run

param(
    [Parameter(Mandatory=$false)]
    [string]$ProjectId = $env:GCP_PROJECT_ID,
    
    [Parameter(Mandatory=$false)]
    [string]$Region = "us-central1",
    
    [Parameter(Mandatory=$false)]
    [string]$ServiceName = "energy-optimization-api",
    
    [Parameter(Mandatory=$false)]
    [string]$RepositoryName = "energy-api-repo",
    
    [Parameter(Mandatory=$false)]
    [string]$Tag = "latest",
    
    [Parameter(Mandatory=$false)]
    [switch]$NoCache,
    
    [Parameter(Mandatory=$false)]
    [switch]$DryRun
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

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "[$([DateTime]::Now.ToString('HH:mm:ss'))] $Message" -ForegroundColor Magenta
}

# Main script
Write-Info "=========================================="
Write-Info "Cloud Run Deployment Script"
Write-Info "Energy Optimization API"
Write-Info "=========================================="
Write-Host ""

# Load .env file if exists
$envFile = Join-Path $PSScriptRoot ".." ".env"
if (Test-Path $envFile) {
    Write-Info "Loading environment variables from .env..."
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
    Write-Success "Environment variables loaded"
} else {
    Write-Warning ".env file not found. Using default values."
}

# Validate Project ID
if (-not $ProjectId) {
    Write-Error "Project ID is required. Set GCP_PROJECT_ID in .env or pass as parameter."
    exit 1
}

Write-Host ""
Write-Info "Configuration:"
Write-Info "  Project ID: $ProjectId"
Write-Info "  Region: $Region"
Write-Info "  Service Name: $ServiceName"
Write-Info "  Repository: $RepositoryName"
Write-Info "  Image Tag: $Tag"
Write-Info "  Dry Run: $DryRun"
Write-Host ""

if ($DryRun) {
    Write-Warning "DRY RUN MODE - No actual deployment will occur"
    Write-Host ""
}

# Check gcloud installation
Write-Step "Step 1: Validating gcloud CLI"
try {
    $gcloudVersion = gcloud version 2>&1 | Select-String "Google Cloud SDK"
    Write-Success "gcloud CLI is installed: $gcloudVersion"
} catch {
    Write-Error "gcloud CLI is not installed or not in PATH"
    exit 1
}

# Set project
Write-Info "Setting active project..."
gcloud config set project $ProjectId 2>&1 | Out-Null
Write-Success "Active project: $ProjectId"

# Check authentication
$account = gcloud config get-value account 2>&1
if (-not $account -or $account -eq "(unset)") {
    Write-Error "Not authenticated. Run: gcloud auth login"
    exit 1
}
Write-Success "Authenticated as: $account"

# Validate required files
Write-Step "Step 2: Validating required files"
$requiredFiles = @(
    "Dockerfile.api",
    ".gcloudignore",
    "src/api/main.py",
    "models/ensembles/ensemble_lightgbm_v1.pkl"
)

$missingFiles = @()
foreach ($file in $requiredFiles) {
    $fullPath = Join-Path $PSScriptRoot ".." $file
    if (Test-Path $fullPath) {
        Write-Success "Found: $file"
    } else {
        Write-Error "Missing: $file"
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Error "Missing required files. Cannot proceed."
    exit 1
}

# Build image tag
$imageTag = "$Region-docker.pkg.dev/$ProjectId/$RepositoryName/${ServiceName}:$Tag"
Write-Host ""
Write-Info "Image will be tagged as: $imageTag"

if ($DryRun) {
    Write-Warning "Dry run complete. Exiting without deployment."
    exit 0
}

# Build image in Cloud Build
Write-Step "Step 3: Building and Deploying via Cloud Build"
Write-Info "This may take 5-7 minutes..."

$substitutions = @(
    "_REGION=$Region",
    "_REPOSITORY=$RepositoryName",
    "_SERVICE_NAME=$ServiceName",
    "_TAG=$Tag",
    "_MIN_INSTANCES=0",
    "_MAX_INSTANCES=2",
    "_CONCURRENCY=80",
    "_CPU=1",
    "_MEMORY=2Gi",
    "_TIMEOUT=300",
    "_MODEL_PATH=/app/models/ensembles/ensemble_lightgbm_v1.pkl",
    "_MODEL_TYPE=stacking_ensemble"
) -join ","

$buildArgs = @(
    "builds", "submit",
    "--config", "cloudbuild.yaml",
    "--substitutions", $substitutions,
    "--project", $ProjectId
)

if ($NoCache) {
    $buildArgs += "--no-cache"
}

try {
    # The output can be quite long, so we don't capture it unless there's an error.
    & gcloud @buildArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Cloud Build pipeline executed successfully (Build, Push, Deploy)."
    } else {
        Write-Error "Cloud Build pipeline failed. Check the logs in the Google Cloud Console."
        exit 1
    }
} catch {
    Write-Error "An error occurred while running gcloud builds submit: $_"
    exit 1
}

# Deployment is now handled by the cloudbuild.yaml, so the old deploy step is removed.

# Get service URL
Write-Step "Step 4: Retrieving service URL"
try {
    $serviceUrl = gcloud run services describe $ServiceName `
        --region $Region `
        --project $ProjectId `
        --format "value(status.url)" 2>&1
    
    Write-Success "Service URL: $serviceUrl"
} catch {
    Write-Warning "Could not retrieve service URL"
    $serviceUrl = $null
}

# Validate deployment
if ($serviceUrl) {
    Write-Step "Step 5: Validating deployment"
    
    Write-Info "Waiting for service to be ready..."
    Start-Sleep -Seconds 15
    
    Write-Info "Testing health endpoint..."
    try {
        $healthUrl = "$serviceUrl/health"
        $response = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 60
        
        if ($response.status -eq "healthy") {
            Write-Success "Health check passed!"
            Write-Info "Response: $($response | ConvertTo-Json -Compress)"
        } else {
            Write-Warning "Health check returned unexpected status: $($response.status)"
        }
    } catch {
        Write-Warning "Health check failed (service may still be starting): $_"
        Write-Info "Try manually: curl $healthUrl"
    }
    
    Write-Host ""
    Write-Info "Testing prediction endpoint..."
    try {
        $predictUrl = "$serviceUrl/predict"
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
        
        $predictResponse = Invoke-RestMethod -Uri $predictUrl -Method Post -Body $testPayload -ContentType "application/json" -TimeoutSec 60
        
        Write-Success "Prediction test passed!"
        Write-Info "Predicted usage: $($predictResponse.predicted_usage_kwh) kWh"
    } catch {
        Write-Warning "Prediction test failed: $_"
        Write-Info "Try manually with test data"
    }
}

# Display summary
Write-Host ""
Write-Success "=========================================="
Write-Success "Deployment Complete!"
Write-Success "=========================================="
Write-Host ""
Write-Info "Service Details:"
Write-Info "  Name: $ServiceName"
Write-Info "  Region: $Region"
Write-Info "  URL: $serviceUrl"
Write-Info "  Image: $imageTag"
Write-Host ""
Write-Info "Configuration:"
Write-Info "  Min Instances: 0 (scale-to-zero)"
Write-Info "  Max Instances: 2"
Write-Info "  Concurrency: 80"
Write-Info "  CPU: 1 vCPU"
Write-Info "  Memory: 2 GiB"
Write-Info "  Timeout: 300s"
Write-Host ""
Write-Info "Available Endpoints:"
Write-Info "  GET  $serviceUrl/"
Write-Info "  GET  $serviceUrl/health"
Write-Info "  POST $serviceUrl/predict"
Write-Info "  POST $serviceUrl/predict/batch"
Write-Info "  GET  $serviceUrl/model/info"
Write-Host ""
Write-Info "Cloud Console:"
Write-Info "  https://console.cloud.google.com/run/detail/$Region/$ServiceName?project=$ProjectId"
Write-Host ""
Write-Info "Logs:"
Write-Info "  .\scripts\view-cloudrun-logs.ps1"
Write-Host ""
Write-Info "Metrics:"
Write-Info "  .\scripts\check-cloudrun-metrics.ps1"
Write-Host ""
Write-Success "Deployment script finished successfully!"
