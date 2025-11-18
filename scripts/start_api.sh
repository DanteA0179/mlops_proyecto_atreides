#!/bin/bash

# ============================================================================
# Script para iniciar la API FastAPI
# Energy Optimization AI
# ============================================================================

echo "🚀 Iniciando API de Energy Optimization..."
echo ""

# Activar entorno Poetry (si estás usando Poetry)
# poetry shell

# Opción 1: Usando uvicorn directamente (recomendado para desarrollo)
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Opción 2: Ejecutar main.py directamente
# python src/api/main.py
