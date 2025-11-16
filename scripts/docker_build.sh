#!/bin/bash
# Docker build script with validation and optimization

set -e

echo "🐳 Building Energy Optimization API Docker Image"
echo "================================================"

# Variables
IMAGE_NAME="energy-optimization-api"
IMAGE_TAG="${1:-latest}"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

# Build with BuildKit for optimization
echo "📦 Building image: ${FULL_IMAGE}"
DOCKER_BUILDKIT=1 docker build \
  --file Dockerfile.api \
  --tag "${FULL_IMAGE}" \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --progress=plain \
  .

# Get image size
IMAGE_SIZE=$(docker images "${FULL_IMAGE}" --format "{{.Size}}")
echo "✅ Image built successfully: ${IMAGE_SIZE}"

# Validate image size (convert to MB for comparison)
SIZE_VALUE=$(echo "${IMAGE_SIZE}" | sed 's/[^0-9.]//g')
SIZE_UNIT=$(echo "${IMAGE_SIZE}" | sed 's/[0-9.]//g' | tr -d ' ')

if [ "$SIZE_UNIT" = "GB" ]; then
    SIZE_MB=$(echo "$SIZE_VALUE * 1024" | bc)
else
    SIZE_MB=$SIZE_VALUE
fi

if (( $(echo "$SIZE_MB > 1500" | bc -l) )); then
    echo "⚠️  WARNING: Image size (${IMAGE_SIZE}) exceeds 1.5GB target"
else
    echo "✅ Image size OK: ${IMAGE_SIZE} < 1.5GB"
fi

# Test container startup
echo "🧪 Testing container startup..."
CONTAINER_ID=$(docker run -d -p 8001:8000 "${FULL_IMAGE}")
echo "Container ID: ${CONTAINER_ID}"
sleep 15

# Test health endpoint
echo "🏥 Testing health endpoint..."
MAX_RETRIES=5
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f http://localhost:8001/health > /dev/null 2>&1; then
        echo "✅ Health check passed"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo "⏳ Health check failed, retrying ($RETRY_COUNT/$MAX_RETRIES)..."
            sleep 5
        else
            echo "❌ Health check failed after $MAX_RETRIES attempts"
            docker logs "${CONTAINER_ID}"
            docker stop "${CONTAINER_ID}"
            docker rm "${CONTAINER_ID}"
            exit 1
        fi
    fi
done

# Test prediction endpoint
echo "🔮 Testing prediction endpoint..."
RESPONSE=$(curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "lagging_reactive_power": 23.45,
    "leading_reactive_power": 12.30,
    "co2": 0.05,
    "lagging_power_factor": 0.85,
    "leading_power_factor": 0.92,
    "nsm": 36000,
    "day_of_week": 1,
    "load_type": "Medium"
  }')

if echo "$RESPONSE" | grep -q "predicted_usage_kwh"; then
    echo "✅ Prediction endpoint working"
    echo "Response: $RESPONSE"
else
    echo "❌ Prediction endpoint failed"
    echo "Response: $RESPONSE"
    docker logs "${CONTAINER_ID}"
    docker stop "${CONTAINER_ID}"
    docker rm "${CONTAINER_ID}"
    exit 1
fi

# Cleanup
echo "🧹 Cleaning up test container..."
docker stop "${CONTAINER_ID}"
docker rm "${CONTAINER_ID}"

echo ""
echo "🎉 All tests passed!"
echo "📊 Image Details:"
docker images "${FULL_IMAGE}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

echo ""
echo "🚀 Next steps:"
echo "  - Run locally: docker run -p 8000:8000 ${FULL_IMAGE}"
echo "  - Run with compose: docker-compose up api"
echo "  - Push to registry: docker tag ${FULL_IMAGE} <registry>/${FULL_IMAGE}"
