# Docker build script with validation and optimization (PowerShell)

$ErrorActionPreference = "Stop"

Write-Host "🐳 Building Energy Optimization API Docker Image" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Variables
$IMAGE_NAME = "energy-optimization-api"
$IMAGE_TAG = if ($args.Count -gt 0) { $args[0] } else { "latest" }
$FULL_IMAGE = "${IMAGE_NAME}:${IMAGE_TAG}"

# Build with BuildKit for optimization
Write-Host "📦 Building image: $FULL_IMAGE" -ForegroundColor Yellow
$env:DOCKER_BUILDKIT = 1
docker build `
  --file Dockerfile.api `
  --tag $FULL_IMAGE `
  --build-arg BUILDKIT_INLINE_CACHE=1 `
  --progress=plain `
  .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed" -ForegroundColor Red
    exit 1
}

# Get image size
$IMAGE_SIZE = docker images $FULL_IMAGE --format "{{.Size}}"
Write-Host "✅ Image built successfully: $IMAGE_SIZE" -ForegroundColor Green

# Validate image size
$SIZE_VALUE = [regex]::Match($IMAGE_SIZE, '\d+\.?\d*').Value
$SIZE_UNIT = $IMAGE_SIZE -replace '\d+\.?\d*', '' -replace '\s', ''

if ($SIZE_UNIT -eq "GB") {
    $SIZE_MB = [double]$SIZE_VALUE * 1024
} else {
    $SIZE_MB = [double]$SIZE_VALUE
}

if ($SIZE_MB -gt 1500) {
    Write-Host "⚠️  WARNING: Image size ($IMAGE_SIZE) exceeds 1.5GB target" -ForegroundColor Yellow
} else {
    Write-Host "✅ Image size OK: $IMAGE_SIZE < 1.5GB" -ForegroundColor Green
}

# Test container startup
Write-Host "🧪 Testing container startup..." -ForegroundColor Yellow
$CONTAINER_ID = docker run -d -p 8001:8000 $FULL_IMAGE
Write-Host "Container ID: $CONTAINER_ID" -ForegroundColor Cyan
Start-Sleep -Seconds 15

# Test health endpoint
Write-Host "🏥 Testing health endpoint..." -ForegroundColor Yellow
$MAX_RETRIES = 5
$RETRY_COUNT = 0
$HEALTH_OK = $false

while ($RETRY_COUNT -lt $MAX_RETRIES) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8001/health" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ Health check passed" -ForegroundColor Green
            $HEALTH_OK = $true
            break
        }
    } catch {
        $RETRY_COUNT++
        if ($RETRY_COUNT -lt $MAX_RETRIES) {
            Write-Host "⏳ Health check failed, retrying ($RETRY_COUNT/$MAX_RETRIES)..." -ForegroundColor Yellow
            Start-Sleep -Seconds 5
        }
    }
}

if (-not $HEALTH_OK) {
    Write-Host "❌ Health check failed after $MAX_RETRIES attempts" -ForegroundColor Red
    docker logs $CONTAINER_ID
    docker stop $CONTAINER_ID
    docker rm $CONTAINER_ID
    exit 1
}

# Test prediction endpoint
Write-Host "🔮 Testing prediction endpoint..." -ForegroundColor Yellow
$body = @{
    lagging_reactive_power = 23.45
    leading_reactive_power = 12.30
    co2 = 0.05
    lagging_power_factor = 0.85
    leading_power_factor = 0.92
    nsm = 36000
    day_of_week = 1
    load_type = "Medium"
} | ConvertTo-Json

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8001/predict" `
        -Method POST `
        -ContentType "application/json" `
        -Body $body `
        -UseBasicParsing

    $responseContent = $response.Content
    if ($responseContent -match "predicted_usage_kwh") {
        Write-Host "✅ Prediction endpoint working" -ForegroundColor Green
        Write-Host "Response: $responseContent" -ForegroundColor Cyan
    } else {
        throw "Response does not contain predicted_usage_kwh"
    }
} catch {
    Write-Host "❌ Prediction endpoint failed" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    docker logs $CONTAINER_ID
    docker stop $CONTAINER_ID
    docker rm $CONTAINER_ID
    exit 1
}

# Cleanup
Write-Host "🧹 Cleaning up test container..." -ForegroundColor Yellow
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID

Write-Host ""
Write-Host "🎉 All tests passed!" -ForegroundColor Green
Write-Host "📊 Image Details:" -ForegroundColor Cyan
docker images $FULL_IMAGE --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

Write-Host ""
Write-Host "🚀 Next steps:" -ForegroundColor Cyan
Write-Host "  - Run locally: docker run -p 8000:8000 $FULL_IMAGE"
Write-Host "  - Run with compose: docker-compose up api"
Write-Host "  - Push to registry: docker tag $FULL_IMAGE <registry>/$FULL_IMAGE"
