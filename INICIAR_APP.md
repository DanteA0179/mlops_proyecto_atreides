# 🚀 Guía para Iniciar API + Streamlit

## 📋 Requisitos Previos

1. **Poetry instalado** (para gestión de dependencias)
2. **Python 3.11+**
3. **Modelo entrenado** (en `models/` o MLflow)

---

## 🎯 Opción 1: Inicio Rápido (Recomendado)

### Terminal 1: Iniciar API

```bash
# Desde la raíz del proyecto
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 2: Iniciar Streamlit

```bash
# Desde la raíz del proyecto
python -m streamlit run src/streamlit_app/app.py
```

### Acceder

- **Streamlit UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health

---

## 🔧 Opción 2: Con Scripts

### Terminal 1: API

```bash
chmod +x scripts/start_api.sh
./scripts/start_api.sh
```

### Terminal 2: Streamlit

```bash
python -m streamlit run src/streamlit_app/app.py
```

---

## 🐳 Opción 3: Con Docker Compose

```bash
# Construir e iniciar ambos servicios
docker-compose -f docker-compose.streamlit.yml up --build

# En segundo plano
docker-compose -f docker-compose.streamlit.yml up -d --build

# Ver logs
docker-compose -f docker-compose.streamlit.yml logs -f

# Detener
docker-compose -f docker-compose.streamlit.yml down
```

**Acceso:**
- Streamlit: http://localhost:8501
- API: http://localhost:8000

---

## 🔍 Verificar que todo funciona

### 1. Verificar API

```bash
# Health check
curl http://localhost:8000/health

# Root endpoint
curl http://localhost:8000/
```

### 2. Verificar Streamlit

1. Abre http://localhost:8501
2. Deberías ver el mensaje "✅ API Online" en el sidebar
3. Navega a "🔮 Predicción Simple"
4. Llena el formulario y haz una predicción
5. Navega a "🤖 Copiloto Conversacional"
6. Envía un mensaje

---

## ⚠️ Troubleshooting

### Problema: API no inicia

**Error: "ModuleNotFoundError"**

```bash
# Instalar dependencias
poetry install

# O con pip
pip install -r requirements.txt
```

**Error: "Model not found"**

La API necesita un modelo entrenado. Opciones:

1. **Entrenar un modelo:**
```bash
python src/models/train_xgboost.py
```

2. **O usar modelo mock** (editar `src/api/utils/config.py`):
```python
MODEL_TYPE = "mock"  # En lugar de "xgboost"
```

### Problema: Streamlit no conecta con API

**Síntoma:** "❌ API Offline" en el sidebar

**Solución:**

1. Verifica que la API esté corriendo:
```bash
curl http://localhost:8000/health
```

2. Verifica la URL en Streamlit:
```python
# En src/streamlit_app/pages/__init__.py
API_BASE_URL = "http://localhost:8000"  # Debe ser correcta
```

3. Reinicia Streamlit:
```bash
# Ctrl+C para detener
# Luego reinicia
python -m streamlit run src/streamlit_app/app.py
```

### Problema: Puerto ya en uso

**Error: "Address already in use"**

**Para API (puerto 8000):**
```bash
# macOS/Linux
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Para Streamlit (puerto 8501):**
```bash
# macOS/Linux
lsof -ti:8501 | xargs kill -9

# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Problema: CORS errors

Si ves errores de CORS en el navegador, verifica que la API tiene CORS habilitado:

```python
# En src/api/main.py (ya debería estar)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📝 Endpoints Disponibles

### API Endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Información de la API |
| `/health` | GET | Health check |
| `/predict` | POST | Predicción simple |
| `/predict/batch` | POST | Predicción por lotes |
| `/chat` | POST | Copiloto conversacional |
| `/model/info` | GET | Información del modelo |
| `/model/metrics` | GET | Métricas del modelo |
| `/docs` | GET | Documentación Swagger |

### Streamlit Pages

| Página | Ruta | Descripción |
|--------|------|-------------|
| Home | `/` | Página de inicio |
| Predicción Simple | `/` | Formulario de predicción |
| Copiloto | `/` | Chat conversacional |

---

## 🎨 Variables de Entorno (Opcional)

Crear archivo `.env` en la raíz:

```bash
# API
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# MLflow (opcional)
MLFLOW_TRACKING_URI=http://localhost:5000

# Modelo
MODEL_TYPE=xgboost
MODEL_VERSION=latest

# Streamlit
API_URL=http://localhost:8000
```

---

## 🔄 Flujo de Trabajo Completo

```
┌─────────────────────────────────────────┐
│   1. Iniciar API (Terminal 1)           │
│   python -m uvicorn src.api.main:app... │
└─────────────────┬───────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────┐
│   2. Verificar API                      │
│   curl http://localhost:8000/health     │
└─────────────────┬───────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────┐
│   3. Iniciar Streamlit (Terminal 2)     │
│   streamlit run src/streamlit_app/app.py│
└─────────────────┬───────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────┐
│   4. Acceder a http://localhost:8501    │
│   - Ver "✅ API Online" en sidebar      │
│   - Probar predicciones                 │
│   - Probar chatbot                      │
└─────────────────────────────────────────┘
```

---

## 📊 Monitoreo

### Ver Logs de API

```bash
# Los logs se muestran en la terminal donde iniciaste la API
# O puedes redirigir a archivo:
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload 2>&1 | tee api.log
```

### Ver Logs de Streamlit

```bash
# Los logs aparecen en la terminal de Streamlit
# También se pueden ver en la UI (hamburger menu > Settings > Logs)
```

---

## 🎉 ¡Listo!

Si todo funciona correctamente, deberías ver:

- ✅ API corriendo en http://localhost:8000
- ✅ Streamlit corriendo en http://localhost:8501
- ✅ "✅ API Online" en el sidebar de Streamlit
- ✅ Predicciones funcionando
- ✅ Chatbot respondiendo

---

## 📚 Documentación Adicional

- **API Docs**: http://localhost:8000/docs (Swagger UI interactivo)
- **API Redoc**: http://localhost:8000/redoc (Documentación alternativa)
- **Streamlit Docs**: https://docs.streamlit.io/

---

## 🆘 Ayuda

Si tienes problemas:

1. Revisa los logs en ambas terminales
2. Verifica que todas las dependencias estén instaladas
3. Asegúrate de que los puertos 8000 y 8501 estén libres
4. Revisa este archivo para troubleshooting

**¡Disfruta tu aplicación de Optimización Energética con IA!** 🚀⚡
