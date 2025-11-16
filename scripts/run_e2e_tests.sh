#!/bin/bash
# Script to run end-to-end tests for Energy Optimization Copilot

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Energy Optimization Copilot E2E Tests${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Must run from project root${NC}"
    exit 1
fi

# Parse arguments
RUN_API_TESTS=false
RUN_PIPELINE_TESTS=false
START_API=false
COVERAGE=false

show_help() {
    echo "Usage: ./scripts/run_e2e_tests.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --api           Run API e2e tests (requires API to be running)"
    echo "  --pipeline      Run pipeline e2e tests"
    echo "  --all           Run all e2e tests"
    echo "  --start-api     Start API server before running tests"
    echo "  --coverage      Generate coverage report"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./scripts/run_e2e_tests.sh --pipeline"
    echo "  ./scripts/run_e2e_tests.sh --api --start-api"
    echo "  ./scripts/run_e2e_tests.sh --all --coverage"
}

if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --api)
            RUN_API_TESTS=true
            shift
            ;;
        --pipeline)
            RUN_PIPELINE_TESTS=true
            shift
            ;;
        --all)
            RUN_API_TESTS=true
            RUN_PIPELINE_TESTS=true
            shift
            ;;
        --start-api)
            START_API=true
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Function to check if API is running
check_api() {
    echo -e "${YELLOW}Checking if API is running...${NC}"
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API is running${NC}"
        return 0
    else
        echo -e "${RED}✗ API is not running${NC}"
        return 1
    fi
}

# Function to start API
start_api() {
    echo -e "${YELLOW}Starting API server...${NC}"
    poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
    API_PID=$!
    
    # Wait for API to be ready
    echo -e "${YELLOW}Waiting for API to be ready...${NC}"
    max_attempts=30
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "${GREEN}✓ API is ready${NC}"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
    done
    
    echo -e "${RED}✗ API failed to start${NC}"
    kill $API_PID 2>/dev/null || true
    return 1
}

# Function to cleanup
cleanup() {
    if [ ! -z "$API_PID" ]; then
        echo -e "${YELLOW}Stopping API server...${NC}"
        kill $API_PID 2>/dev/null || true
        wait $API_PID 2>/dev/null || true
        echo -e "${GREEN}✓ API stopped${NC}"
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Build pytest command
PYTEST_CMD="poetry run pytest tests/e2e/"
PYTEST_ARGS="-v"

if [ "$COVERAGE" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS --cov=src --cov-report=html --cov-report=term"
fi

# Run tests
EXIT_CODE=0

if [ "$RUN_PIPELINE_TESTS" = true ]; then
    echo -e "${GREEN}Running pipeline e2e tests...${NC}"
    echo ""
    $PYTEST_CMD/test_pipeline_e2e.py $PYTEST_ARGS || EXIT_CODE=$?
    echo ""
fi

if [ "$RUN_API_TESTS" = true ]; then
    # Check if API is running or start it
    if ! check_api; then
        if [ "$START_API" = true ]; then
            start_api || {
                echo -e "${RED}Failed to start API${NC}"
                exit 1
            }
        else
            echo -e "${RED}API is not running. Use --start-api to start it automatically${NC}"
            exit 1
        fi
    fi
    
    echo -e "${GREEN}Running API e2e tests...${NC}"
    echo ""
    $PYTEST_CMD/test_api_e2e.py $PYTEST_ARGS || EXIT_CODE=$?
    echo ""
fi

# Summary
echo -e "${GREEN}========================================${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
fi
echo -e "${GREEN}========================================${NC}"

if [ "$COVERAGE" = true ]; then
    echo -e "${YELLOW}Coverage report generated in htmlcov/index.html${NC}"
fi

exit $EXIT_CODE
