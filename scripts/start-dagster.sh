#!/usr/bin/env bash
#
# start-dagster.sh - Start Dagster development server
#
# Usage:
#   ./start-dagster.sh [PORT] [HOST]
#
# Arguments:
#   PORT    Port for Dagster UI (default: 3000)
#   HOST    Host to bind to (default: 127.0.0.1)
#
# Examples:
#   ./start-dagster.sh
#   ./start-dagster.sh 3001
#   ./start-dagster.sh 3000 0.0.0.0

set -e

# Default values
PORT=${1:-3000}
HOST=${2:-127.0.0.1}

# Get project root (one level up from scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Set DAGSTER_HOME
export DAGSTER_HOME="$PROJECT_ROOT/dagster_home"

# Print banner
echo "====================================="
echo "  Energy Optimization Copilot"
echo "  Dagster Development Server"
echo "====================================="
echo ""
echo "Project Root:  $PROJECT_ROOT"
echo "DAGSTER_HOME:  $DAGSTER_HOME"
echo "UI Address:    http://$HOST:$PORT"
echo ""
echo "Available Jobs:"
echo "  - complete_training_job      (XGBoost, LightGBM, CatBoost, Ensembles)"
echo "  - chronos_zeroshot_job       (Chronos-2 zero-shot)"
echo "  - chronos_finetuned_job      (Chronos-2 fine-tuned)"
echo "  - chronos_covariates_job     (Chronos-2 with covariates)"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Dagster dev server
poetry run dagster dev -m src.dagster_pipeline.definitions -h "$HOST" -p "$PORT"
