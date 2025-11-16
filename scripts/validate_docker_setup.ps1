# Validation script for Docker setup

$ErrorActionPreference = "Stop"

Write-Host "🔍 Validating Docker Setup for US-022" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

$allChecks = $true

# Check 1: Docker files exist
Write-Host "📁 Checking Docker files..." -ForegroundColor Yellow
$dockerFiles = @(
    "Dockerfile.api",
    ".dockerignore",
    "docker-compose.yml"
)

foreach ($file in $dockerFiles) {
    if (Test-Path $file) {
        Write-Host "  ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $file NOT FOUND" -ForegroundColor Red
        $allChecks = $false
    }
}

# Check 2: ONNX models exist
Write-Host ""
Write-Host "🤖 Checking ONNX models..." -ForegroundColor Yellow
$modelPaths = @(
    "models/onnx/lightgbm.onnx",
    "models/onnx/xgboost.onnx",
    "models/onnx/catboost.onnx",
    "models/onnx/lightgbm_ensemble"
)

foreach ($path in $modelPaths) {
    if (Test-Path $path) {
        Write-Host "  ✅ $path" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  $path NOT FOUND (optional)" -ForegroundColor Yellow
    }
}

# Check 3: Preprocessing artifacts
Write-Host ""
Write-Host "🔧 Checking preprocessing artifacts..." -ForegroundColor Yellow
if (Test-Path "models/preprocessing") {
    Write-Host "  ✅ models/preprocessing/" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  models/preprocessing/ NOT FOUND (optional)" -ForegroundColor Yellow
}

# Check 4: Requirements file
Write-Host ""
Write-Host "📦 Checking requirements..." -ForegroundColor Yellow
if (Test-Path "requirements-api.txt") {
    Write-Host "  ✅ requirements-api.txt" -ForegroundColor Green
    $reqCount = (Get-Content "requirements-api.txt" | Where-Object { $_ -notmatch '^#' -and $_ -ne '' }).Count
    Write-Host "     Dependencies: $reqCount" -ForegroundColor Cyan
} else {
    Write-Host "  ❌ requirements-api.txt NOT FOUND" -ForegroundColor Red
    $allChecks = $false
}

# Check 5: Source code
Write-Host ""
Write-Host "💻 Checking source code..." -ForegroundColor Yellow
$srcPaths = @(
    "src/api/main.py",
    "src/api/services/onnx_service.py",
    "src/api/routes/predict_onnx.py"
)

foreach ($path in $srcPaths) {
    if (Test-Path $path) {
        Write-Host "  ✅ $path" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $path NOT FOUND" -ForegroundColor Red
        $allChecks = $false
    }
}

# Check 6: Documentation
Write-Host ""
Write-Host "📚 Checking documentation..." -ForegroundColor Yellow
$docPaths = @(
    "docker/README.md",
    "docker/QUICKSTART.md",
    "docs/deployment/docker-deployment.md",
    "docs/us-resolved/us-022.md"
)

foreach ($path in $docPaths) {
    if (Test-Path $path) {
        Write-Host "  ✅ $path" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $path NOT FOUND" -ForegroundColor Red
        $allChecks = $false
    }
}

# Check 7: Scripts
Write-Host ""
Write-Host "🔨 Checking build scripts..." -ForegroundColor Yellow
$scriptPaths = @(
    "scripts/docker_build.sh",
    "scripts/docker_build.ps1"
)

foreach ($path in $scriptPaths) {
    if (Test-Path $path) {
        Write-Host "  ✅ $path" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $path NOT FOUND" -ForegroundColor Red
        $allChecks = $false
    }
}

# Check 8: CI/CD workflows
Write-Host ""
Write-Host "🚀 Checking CI/CD workflows..." -ForegroundColor Yellow
$workflowPaths = @(
    ".github/workflows/docker-build.yml",
    ".github/workflows/deploy-cloudrun.yml"
)

foreach ($path in $workflowPaths) {
    if (Test-Path $path) {
        Write-Host "  ✅ $path" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $path NOT FOUND" -ForegroundColor Red
        $allChecks = $false
    }
}

# Check 9: Docker daemon
Write-Host ""
Write-Host "🐳 Checking Docker daemon..." -ForegroundColor Yellow
try {
    $dockerVersion = docker version --format '{{.Server.Version}}' 2>$null
    if ($dockerVersion) {
        Write-Host "  ✅ Docker daemon running (version: $dockerVersion)" -ForegroundColor Green
    } else {
        Write-Host "  ❌ Docker daemon not running" -ForegroundColor Red
        $allChecks = $false
    }
} catch {
    Write-Host "  ❌ Docker not installed or not running" -ForegroundColor Red
    $allChecks = $false
}

# Check 10: Docker Compose
Write-Host ""
Write-Host "🐙 Checking Docker Compose..." -ForegroundColor Yellow
try {
    $composeVersion = docker-compose version --short 2>$null
    if ($composeVersion) {
        Write-Host "  ✅ Docker Compose installed (version: $composeVersion)" -ForegroundColor Green
    } else {
        Write-Host "  ❌ Docker Compose not installed" -ForegroundColor Red
        $allChecks = $false
    }
} catch {
    Write-Host "  ❌ Docker Compose not installed" -ForegroundColor Red
    $allChecks = $false
}

# Summary
Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
if ($allChecks) {
    Write-Host "✅ All checks passed! Ready to build." -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Build: docker-compose build api" -ForegroundColor White
    Write-Host "  2. Run: docker-compose up api" -ForegroundColor White
    Write-Host "  3. Test: curl http://localhost:8000/health" -ForegroundColor White
    Write-Host ""
    Write-Host "Or use automated script:" -ForegroundColor Cyan
    Write-Host "  .\scripts\docker_build.ps1" -ForegroundColor White
    exit 0
} else {
    Write-Host "❌ Some checks failed. Please fix the issues above." -ForegroundColor Red
    exit 1
}
