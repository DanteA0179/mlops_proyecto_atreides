#!/bin/bash

echo "🚀 Iniciando Stack Completo..."

# API
echo "📡 Iniciando API..."
poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 &
API_PID=$!

sleep 5

# Frontend
echo "🎨 Iniciando Frontend..."
cd frontend
poetry run streamlit run app.py &
FRONTEND_PID=$!

echo ""
echo "✅ Stack iniciado:"
echo "   - API: http://localhost:8000"
echo "   - Docs: http://localhost:8000/docs"
echo "   - Frontend: http://localhost:8501"
echo ""
echo "Presiona Ctrl+C para detener"

cleanup() {
    echo "🛑 Deteniendo servicios..."
    kill $API_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup INT
wait
