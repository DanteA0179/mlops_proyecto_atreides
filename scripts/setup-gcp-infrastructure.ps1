# GCP Infrastructure Setup Script
# Energy Optimization API - Cloud Run Deployment
# This script sets up the necessary GCP infrastructure for deploying the API

param(
    [Parameter(Mandatory=$false)]
    [string]$ProjectId = $env:GCP_PROJECT_ID,
    
    [Parameter(Mandatory=$false)]
    [string]$Region = "us-central1",
    
    [Parameter(Mandatory=$false)]
    [string]$RepositoryName = "energy-api-repo"
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

# Main script
Write-Info "=========================================="
Write-Info "GCP Infrastructure Setup for Cloud Run"
Write-Info "=========================================="
Write-Host ""

# Check if gcloud is installed
Write-Info "Checking gcloud CLI installation..."
try {
    $gcloudVersion = gcloud version 2>&1 | Select-String "Google Cloud SDK"
    Write-Success "gcloud CLI is installed: $gcloudVersion"
} catch {
    Write-Error "gcloud CLI is not installed or not in PATH"
    Write-Info "Please install from: https://cloud.google.com/sdk/docs/install"
    exit 1
}

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
}

# Validate Project ID
if (-not $ProjectId) {
    Write-Error "Project ID is required. Set GCP_PROJECT_ID in .env or pass as parameter."
    exit 1
}

Write-Info "Project ID: $ProjectId"
Write-Info "Region: $Region"
Write-Info "Repository Name: $RepositoryName"
Write-Host ""

# Authenticate and set project
Write-Info "Setting active project..."
try {
    gcloud config set project $ProjectId 2>&1 | Out-Null
    Write-Success "Active project set to: $ProjectId"
} catch {
    Write-Error "Failed to set project. Please authenticate with: gcloud auth login"
    exit 1
}

# Enable required APIs
Write-Info "Enabling required GCP APIs..."
$apis = @(
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "secretmanager.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com"
)

foreach ($api in $apis) {
    Write-Info "Enabling $api..."
    try {
        gcloud services enable $api --project=$ProjectId 2>&1 | Out-Null
        Write-Success "Enabled: $api"
    } catch {
        Write-Warning "Failed to enable $api (may already be enabled)"
    }
}

Write-Host ""

# Check if Artifact Registry repository exists
Write-Info "Checking Artifact Registry repository..."
$repoExists = gcloud artifacts repositories describe $RepositoryName `
    --location=$Region `
    --project=$ProjectId 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Warning "Repository '$RepositoryName' already exists in $Region"
} else {
    Write-Info "Creating Artifact Registry repository..."
    try {
        gcloud artifacts repositories create $RepositoryName `
            --repository-format=docker `
            --location=$Region `
            --description="Energy Optimization API Docker images" `
            --project=$ProjectId
        
        Write-Success "Artifact Registry repository created successfully"
    } catch {
        Write-Error "Failed to create Artifact Registry repository"
        exit 1
    }
}

Write-Host ""

# Configure Docker authentication
Write-Info "Configuring Docker authentication for Artifact Registry..."
try {
    gcloud auth configure-docker "$Region-docker.pkg.dev" --quiet
    Write-Success "Docker authentication configured"
} catch {
    Write-Warning "Failed to configure Docker auth (may already be configured)"
}

Write-Host ""

# Display service account info
Write-Info "Checking service account permissions..."
$serviceAccount = gcloud config get-value account 2>&1
Write-Info "Current account: $serviceAccount"

# Check IAM permissions
Write-Info "Verifying IAM roles..."
$roles = gcloud projects get-iam-policy $ProjectId `
    --flatten="bindings[].members" `
    --filter="bindings.members:$serviceAccount" `
    --format="value(bindings.role)" 2>&1

if ($roles -match "roles/owner" -or $roles -match "roles/editor" -or $roles -match "roles/run.admin") {
    Write-Success "Service account has sufficient permissions"
} else {
    Write-Warning "Service account may not have sufficient permissions"
    Write-Info "Required roles: Cloud Run Admin, Artifact Registry Admin, Cloud Build Editor"
}

Write-Host ""

# Check quotas (skip this check as it can be slow)
Write-Info "Skipping quota check (can be verified in Cloud Console)..."
Write-Host ""

# Display summary
Write-Success "=========================================="
Write-Success "Infrastructure Setup Complete!"
Write-Success "=========================================="
Write-Host ""
Write-Info "Next steps:"
Write-Info "1. Ensure your model file exists: models/ensembles/ensemble_lightgbm_v1.pkl"
Write-Info "2. Run deployment script: .\scripts\deploy-to-cloudrun.ps1"
Write-Info "3. Monitor deployment in Cloud Console:"
Write-Info "   https://console.cloud.google.com/run?project=$ProjectId"
Write-Host ""
Write-Info "Artifact Registry URL:"
Write-Info "$Region-docker.pkg.dev/$ProjectId/$RepositoryName"
Write-Host ""
