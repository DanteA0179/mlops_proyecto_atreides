# Frontend - Sistema de Optimización Energética

Interfaz web con Streamlit para predicción de consumo energético.

## 🚀 Inicio Rápido

### Opción 1: Script automático
```bash
# Desde la raíz del proyecto
./scripts/start_frontend.sh
```

### Opción 2: Manual
```bash
# Terminal 1 - API
poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend
cd frontend
streamlit run app.py
```

## 📍 URLs

- Frontend: http://localhost:8501
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 📁 Estructura
```
frontend/
├── app.py                    # Página principal
├── pages/
│   ├── 01_🤖_Predicción.py  # Predicción en tiempo real
│   └── 02_📈_Análisis.py    # Análisis de modelos
├── utils/
│   └── api_client.py         # Cliente HTTP
├── .streamlit/
│   └── config.toml           # Configuración
└── requirements.txt          # Dependencias
```

## 🔧 Troubleshooting

### Error: No se puede conectar a la API
```bash
# Verificar que la API está corriendo
curl http://localhost:8000/health

# Reiniciar API
poetry run uvicorn src.api.main:app --reload
```
