#!/bin/bash

echo "🚀 Configurando Frontend para MLOps Proyecto Atreides"
echo "=================================================="
echo ""

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar que estamos en el directorio correcto
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: No se encuentra pyproject.toml"
    echo "   Ejecuta este script desde la raíz del proyecto mlops_proyecto_atreides"
    exit 1
fi

echo -e "${GREEN}✓${NC} Directorio correcto detectado"
echo ""

# 1. Crear estructura de directorios
echo -e "${BLUE}📁 Creando estructura de directorios...${NC}"
mkdir -p frontend/pages
mkdir -p frontend/utils
mkdir -p frontend/.streamlit
echo -e "${GREEN}✓${NC} Directorios creados"
echo ""

# 2. Crear requirements.txt
echo -e "${BLUE}📦 Creando frontend/requirements.txt...${NC}"
cat > frontend/requirements.txt << 'EOF'
streamlit==1.29.0
requests==2.31.0
plotly==5.18.0
pandas==2.1.4
polars==0.19.0
EOF
echo -e "${GREEN}✓${NC} requirements.txt creado"
echo ""

# 3. Crear config.toml
echo -e "${BLUE}🎨 Creando configuración de Streamlit...${NC}"
cat > frontend/.streamlit/config.toml << 'EOF'
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"

[server]
port = 8501
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200

[browser]
gatherUsageStats = false
EOF
echo -e "${GREEN}✓${NC} config.toml creado"
echo ""

# 4. Crear __init__.py en utils
echo -e "${BLUE}🔧 Creando utils/__init__.py...${NC}"
cat > frontend/utils/__init__.py << 'EOF'
"""Utils package"""
EOF
echo -e "${GREEN}✓${NC} __init__.py creado"
echo ""

# 5. Crear api_client.py
echo -e "${BLUE}🔌 Creando api_client.py...${NC}"
cat > frontend/utils/api_client.py << 'EOF'
"""Cliente HTTP para la API de FastAPI"""
import requests
from typing import Dict, Any, Optional, List
import streamlit as st


class APIClient:
    """Cliente para consumir la API de predicción"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.timeout = 30
        
    def health_check(self) -> bool:
        """Verifica que la API esté disponible"""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def get_models(self) -> Optional[List[str]]:
        """Obtiene lista de modelos disponibles"""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            st.error(f"Error obteniendo modelos: {e}")
            return None
    
    def get_metrics(self, model: Optional[str] = None) -> Optional[Dict]:
        """Obtiene métricas del modelo"""
        try:
            params = {"model": model} if model else {}
            response = requests.get(
                f"{self.base_url}/metrics",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error obteniendo métricas: {e}")
            return None
    
    def predict(self, features: Dict[str, Any]) -> Optional[Dict]:
        """Realiza predicción"""
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json=features,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error en predicción: {e}")
            return None
    
    def simulate(self, base_features: Dict[str, Any], variations: List[Dict[str, Any]], model: Optional[str] = None) -> Optional[Dict]:
        """Ejecuta simulaciones what-if"""
        try:
            payload = {
                "base_features": base_features,
                "variations": variations,
                "model": model
            }
            response = requests.post(
                f"{self.base_url}/simulate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error en simulación: {e}")
            return None
EOF
echo -e "${GREEN}✓${NC} api_client.py creado"
echo ""

# 6. Crear app.py principal
echo -e "${BLUE}🏠 Creando app.py principal...${NC}"
cat > frontend/app.py << 'EOF'
"""Aplicación principal de Streamlit - Sistema de Optimización Energética"""
import streamlit as st
import sys
from pathlib import Path

# Agregar el directorio utils al path
sys.path.insert(0, str(Path(__file__).parent))

from utils.api_client import APIClient

st.set_page_config(
    page_title="Optimización Energética - Atreides",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #FF4B4B;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.2rem;
    text-align: center;
    color: #FAFAFA;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# Inicializar cliente API
if 'api_client' not in st.session_state:
    api_urls = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://api:8000",
    ]
    
    st.session_state.api_client = None
    for url in api_urls:
        client = APIClient(base_url=url)
        if client.health_check():
            st.session_state.api_client = client
            st.session_state.api_url = url
            break

# Header
st.markdown('<div class="main-header">🏭 Sistema de Optimización Energética</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Copiloto Inteligente con IA - Industria Siderúrgica</div>', unsafe_allow_html=True)

st.markdown("---")

# Verificar conexión con API
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.session_state.get('api_client') and st.session_state.api_client.health_check():
        st.success(f"✅ API Conectada: {st.session_state.api_url}")
    else:
        st.error("❌ No se puede conectar a la API")
        st.info("""
        **Para iniciar la API, ejecuta en otra terminal:**
```bash
        poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```
        """)
        st.stop()

# Información del proyecto
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🎯 Objetivo")
    st.write("""
    - Predecir consumo con **RMSE < 0.205**
    - Análisis conversacional de drivers
    - Simulaciones what-if
    """)

with col2:
    st.markdown("### 🤖 Modelos Disponibles")
    models = st.session_state.api_client.get_models()
    if models:
        for model in models[:5]:
            st.write(f"• {model}")
    else:
        st.write("• XGBoost\n• LightGBM\n• Chronos-2")

with col3:
    st.markdown("### 📊 Métricas")
    metrics = st.session_state.api_client.get_metrics()
    if metrics:
        st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
        st.metric("R²", f"{metrics.get('r2', 0):.4f}")
    else:
        st.info("Cargando métricas...")

# Navegación
st.markdown("---")
st.markdown("### 📑 Páginas Disponibles")

col1, col2 = st.columns(2)

with col1:
    if st.button("🤖 Predicción", use_container_width=True):
        st.switch_page("pages/01_🤖_Predicción.py")

with col2:
    if st.button("📈 Análisis", use_container_width=True):
        st.switch_page("pages/02_📈_Análisis.py")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>Proyecto MLOps - Equipo Atreides 2025</p>
</div>
""", unsafe_allow_html=True)
EOF
echo -e "${GREEN}✓${NC} app.py creado"
echo ""

# 7. Crear página de predicción
echo -e "${BLUE}🤖 Creando página de Predicción...${NC}"
cat > 'frontend/pages/01_🤖_Predicción.py' << 'EOF'
"""Página de predicción en tiempo real"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Predicción", page_icon="🤖", layout="wide")

if 'api_client' not in st.session_state or st.session_state.api_client is None:
    st.error("⚠️ No hay conexión con la API. Ve a la página principal.")
    if st.button("⬅️ Volver a Home"):
        st.switch_page("app.py")
    st.stop()

api_client = st.session_state.api_client

st.title("🤖 Predicción de Consumo Energético")
st.markdown("Ingresa los parámetros operacionales para predecir el consumo.")

st.markdown("---")

with st.form("prediction_form"):
    st.subheader("Parámetros de Entrada")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚡ Variables Eléctricas")
        
        lagging_pf = st.slider(
            "Lagging Current Power Factor",
            0.0, 1.0, 0.85, 0.01,
            help="Factor de potencia de corriente atrasada"
        )
        
        lagging_rp = st.number_input(
            "Lagging Reactive Power (kVarh)",
            min_value=0.0, value=50.0, step=1.0
        )
        
        leading_pf = st.slider(
            "Leading Current Power Factor",
            0.0, 1.0, 0.90, 0.01
        )
        
        leading_rp = st.number_input(
            "Leading Reactive Power (kVarh)",
            min_value=0.0, value=30.0, step=1.0
        )
    
    with col2:
        st.markdown("#### 🏭 Variables Operacionales")
        
        co2 = st.number_input(
            "CO2 (tCO2)",
            min_value=0.0, value=0.05, step=0.01,
            help="Variable más importante según MI"
        )
        
        nsm = st.number_input(
            "NSM (Segundos desde medianoche)",
            min_value=0, max_value=86400, value=43200, step=900
        )
        
        load_type = st.selectbox(
            "Tipo de Carga",
            ["Light_Load", "Medium_Load", "Maximum_Load"],
            index=1
        )
        
        day = st.selectbox(
            "Día de la Semana",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
    
    submit = st.form_submit_button("🔮 Predecir Consumo", use_container_width=True)

if submit:
    with st.spinner("🔄 Realizando predicción..."):
        payload = {
            "Lagging_Current_Power_Factor": lagging_pf,
            "Lagging_Current_Reactive.Power_kVarh": lagging_rp,
            "Leading_Current_Power_Factor": leading_pf,
            "Leading_Current_Reactive_Power_kVarh": leading_rp,
            "CO2(tCO2)": co2,
            "NSM": nsm,
            "Load_Type": load_type,
            "Day_of_week": day
        }
        
        result = api_client.predict(payload)
        
        if result:
            st.success("✅ Predicción completada")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Consumo Predicho", f"{result['prediction']:.2f} kWh")
            
            with col2:
                if 'confidence_interval' in result:
                    ci = result['confidence_interval']
                    st.metric("Límite Inferior", f"{ci['lower']:.2f} kWh")
            
            with col3:
                if 'confidence_interval' in result:
                    ci = result['confidence_interval']
                    st.metric("Límite Superior", f"{ci['upper']:.2f} kWh")
            
            st.markdown("---")
            
            prediction_value = result['prediction']
            if prediction_value < 50:
                st.success("🟢 **Consumo BAJO** - Operación eficiente")
            elif prediction_value < 100:
                st.warning("🟡 **Consumo MEDIO** - Dentro de rango normal")
            else:
                st.error("🔴 **Consumo ALTO** - Considerar optimización")

with st.expander("ℹ️ Información sobre variables"):
    st.markdown("""
    ### Top 5 Features por Mutual Information:
    
    1. **CO2 (tCO2)** - MI: 1.214 (más importante)
    2. **Lagging Current Power Factor** - MI: 1.204
    3. **Lagging Reactive Power** - MI: 0.823
    4. **NSM** - MI: 0.450
    5. **Leading Power Factor** - MI: 0.413
    """)
EOF
echo -e "${GREEN}✓${NC} Página de Predicción creada"
echo ""

# 8. Crear página de análisis
echo -e "${BLUE}📈 Creando página de Análisis...${NC}"
cat > 'frontend/pages/02_📈_Análisis.py' << 'EOF'
"""Página de análisis de modelos"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Análisis", page_icon="��", layout="wide")

if 'api_client' not in st.session_state or st.session_state.api_client is None:
    st.error("⚠️ No hay conexión con la API")
    if st.button("⬅️ Volver a Home"):
        st.switch_page("app.py")
    st.stop()

api_client = st.session_state.api_client

st.title("📈 Análisis de Modelos")
st.markdown("Evaluación y comparación de modelos de Machine Learning")

st.markdown("---")

# Métricas
st.subheader("📊 Métricas de Rendimiento")

models = api_client.get_models()
selected_model = st.selectbox("Seleccionar Modelo", models or ["xgboost"])

metrics = api_client.get_metrics(selected_model)

if metrics:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rmse = metrics.get('rmse', 0)
        delta = -21.5 if rmse < 0.205 else None
        st.metric("RMSE", f"{rmse:.4f}", delta=f"{delta}%" if delta else None, delta_color="inverse")
    
    with col2:
        st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
    
    with col3:
        st.metric("R² Score", f"{metrics.get('r2', 0):.4f}")
    
    with col4:
        st.metric("CV (%)", f"{metrics.get('cv', 0):.4f}")
    
    # Comparación con benchmark
    st.markdown("---")
    st.subheader("🎯 Comparación con Benchmark CUBIST")
    
    benchmark_data = {
        "Métrica": ["RMSE", "MAE", "CV (%)"],
        "Benchmark (CUBIST)": [0.2410, 0.0547, 0.8770],
        "Meta": [0.205, 0.046, 0.75],
        "Modelo Actual": [
            metrics.get('rmse', 0),
            metrics.get('mae', 0),
            metrics.get('cv', 0)
        ]
    }
    
    df_benchmark = pd.DataFrame(benchmark_data)
    
    df_benchmark['Mejora (%)'] = (
        (df_benchmark['Benchmark (CUBIST)'] - df_benchmark['Modelo Actual']) / 
        df_benchmark['Benchmark (CUBIST)'] * 100
    ).round(2)
    
    st.dataframe(df_benchmark, use_container_width=True)
    
    if rmse < 0.205:
        st.success("✅ **Meta RMSE alcanzada!** El modelo supera el benchmark CUBIST")
    else:
        improvement_needed = ((rmse - 0.205) / rmse) * 100
        st.warning(f"⚠️ Se necesita una mejora del {improvement_needed:.1f}% para alcanzar la meta")

else:
    st.info("No hay métricas disponibles para este modelo")

# Feature Importance
st.markdown("---")
st.subheader("�� Top Features por Mutual Information")

feature_data = {
    "Feature": [
        "CO2(tCO2)",
        "Lagging_Current_Power_Factor",
        "Lagging_Current_Reactive.Power_kVarh",
        "NSM",
        "Leading_Current_Power_Factor"
    ],
    "MI Score": [1.214, 1.204, 0.823, 0.450, 0.413]
}

df_features = pd.DataFrame(feature_data)

col1, col2 = st.columns([2, 1])

with col1:
    st.dataframe(df_features, use_container_width=True)

with col2:
    st.info("""
    **Insights:**
    
    • CO2 es el predictor más fuerte
    • Variables eléctricas dominan top 5
    • NSM captura patrones temporales
    """)
EOF
echo -e "${GREEN}✓${NC} Página de Análisis creada"
echo ""

# 9. Crear scripts de inicio
echo -e "${BLUE}🚀 Creando scripts de inicio...${NC}"

# Script bash
cat > scripts/start_frontend.sh << 'EOF'
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
EOF

chmod +x scripts/start_frontend.sh

# Script PowerShell
cat > scripts/start_frontend.ps1 << 'EOF'
Write-Host "🚀 Iniciando Stack..." -ForegroundColor Green

Write-Host "📡 Iniciando API..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "poetry run uvicorn src.api.main:app --reload"

Start-Sleep -Seconds 5

Write-Host "🎨 Iniciando Frontend..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; poetry run streamlit run app.py"

Write-Host ""
Write-Host "✅ URLs:" -ForegroundColor Green
Write-Host "   - API: http://localhost:8000" -ForegroundColor Yellow
Write-Host "   - Frontend: http://localhost:8501" -ForegroundColor Yellow
EOF

echo -e "${GREEN}✓${NC} Scripts de inicio creados"
echo ""

# 10. Crear README del frontend
echo -e "${BLUE}📝 Creando README...${NC}"
cat > frontend/README.md << 'EOF'
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
EOF
echo -e "${GREEN}✓${NC} README creado"
echo ""

# Resumen final
echo ""
echo "=========================================="
echo -e "${GREEN}✅ Setup completado exitosamente!${NC}"
echo "=========================================="
echo ""
echo "Estructura creada:"
echo "  frontend/"
echo "  ├── app.py"
echo "  ├── requirements.txt"
echo "  ├── README.md"
echo "  ├── pages/"
echo "  │   ├── 01_🤖_Predicción.py"
echo "  │   └── 02_📈_Análisis.py"
echo "  ├── utils/"
echo "  │   ├── __init__.py"
echo "  │   └── api_client.py"
echo "  └── .streamlit/"
echo "      └── config.toml"
echo ""
echo -e "${YELLOW}📋 Próximos pasos:${NC}"
echo ""
echo "1. Instalar dependencias del frontend:"
echo -e "   ${BLUE}cd frontend && pip install -r requirements.txt${NC}"
echo ""
echo "2. Iniciar el stack completo:"
echo -e "   ${BLUE}./scripts/start_frontend.sh${NC}"
echo ""
echo "3. Abrir en el navegador:"
echo "   - Frontend: http://localhost:8501"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo -e "${GREEN}¡Listo para usar! 🎉${NC}"
echo ""

