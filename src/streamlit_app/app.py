"""
Aplicación principal de Streamlit para Energy Optimization AI.

Este módulo implementa la interfaz web del sistema de optimización energética
con IA para la industria siderúrgica.

Autor: Equipo Atreides
Fecha: 2025
"""

import streamlit as st
from pathlib import Path
import sys

# Agregar el directorio raíz al path para imports
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

# Configuración de la página
st.set_page_config(
    page_title="Energy Optimization AI - Steel Industry",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/DanteA0179/mlops_proyecto_atreides',
        'Report a bug': 'https://github.com/DanteA0179/mlops_proyecto_atreides/issues',
        'About': """
        # Energy Optimization AI
        Sistema de Optimización Energética con IA para la Industria Siderúrgica.
        
        **Versión:** 1.0.0  
        **Equipo:** Atreides  
        **Proyecto:** MLOps 2025
        """
    }
)

# CSS personalizado para mejor diseño
st.markdown("""
    <style>
    /* Estilos generales */
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Tarjetas de métricas */
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Botones */
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #155a8a;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 5px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        font-weight: bold;
    }
    
    /* Formularios */
    .stForm {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        background-color: #fafafa;
    }
    
    /* Mejoras de responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .stButton>button {
            font-size: 0.9rem;
        }
    }
    
    /* Animaciones sutiles */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .element-container {
        animation: fadeIn 0.5s ease-in;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar para navegación
st.sidebar.title("🧭 Navegación")
st.sidebar.markdown("---")

# Radio buttons para selección de página
page = st.sidebar.radio(
    "Selecciona una página:",
    ["🏠 Home", "🔮 Predicción Simple", "🤖 Copiloto Conversacional"],
    help="Navega entre las diferentes funcionalidades del sistema"
)

# Información del proyecto en sidebar
st.sidebar.markdown("---")
st.sidebar.info("""
    ### 📊 Proyecto Atreides
    Sistema de Optimización Energética con IA
    
    **Versión:** 1.0.0  
    **Dataset:** Steel Industry Energy  
    **Registros:** 35,040 mediciones  
    **Frecuencia:** 15 minutos  
    **Periodo:** 2018
""")

# Sección expandible con información técnica
with st.sidebar.expander("🔧 Stack Tecnológico"):
    st.markdown("""
    **Frontend:**
    - 🎨 Streamlit 1.28+
    - 📊 Plotly 5.17+
    
    **Backend:**
    - ⚡ FastAPI
    - 🐍 Python 3.11
    
    **Machine Learning:**
    - 🌲 XGBoost
    - 🔮 Chronos-2 (Foundation Model)
    - 🤖 LightGBM, CatBoost
    
    **Orquestación:**
    - 🔄 Dagster
    - 📦 DVC (Data Versioning)
    
    **Data Storage:**
    - 🦆 DuckDB
    - ☁️ Google Cloud Storage
    
    **Monitoring:**
    - 📈 MLflow
    - 👁️ Evidently AI
    """)

# Sección expandible con métricas del modelo
with st.sidebar.expander("📊 Métricas del Sistema"):
    st.markdown("""
    **Rendimiento del Modelo:**
    - 🎯 RMSE: < 0.205 kWh
    - 📉 MAE: < 0.046 kWh
    - 📊 R²: > 0.92
    
    **Mejora vs Benchmark:**
    - ✅ 15% mejor que CUBIST
    - ⚡ Latencia: < 500ms p95
    
    **Cobertura de Tests:**
    - 🧪 Unit Tests: > 80%
    - 🔬 Integration: > 70%
    """)

# Estado de la API
with st.sidebar.expander("🔌 Estado del Sistema"):
    import requests
    from datetime import datetime
    
    try:
        # Intentar conectar con la API
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ API Online")
            st.caption(f"Última verificación: {datetime.now().strftime('%H:%M:%S')}")
        else:
            st.warning("⚠️ API respondiendo con errores")
    except requests.exceptions.ConnectionError:
        st.error("❌ API Offline")
        st.caption("Inicia la API con:")
        st.code("poetry run uvicorn src.api.main:app --reload", language="bash")
    except Exception as e:
        st.warning(f"⚠️ Error verificando API: {str(e)[:50]}...")

# Enlaces útiles
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 Enlaces Útiles")
st.sidebar.markdown("""
- [📖 Documentación](https://github.com/DanteA0179/mlops_proyecto_atreides)
- [🐛 Reportar Bug](https://github.com/DanteA0179/mlops_proyecto_atreides/issues)
- [📊 API Docs](http://localhost:8000/docs)
- [🔬 MLflow UI](http://localhost:5000)
- [⚙️ Dagster UI](http://localhost:3000)
""")

# Routing de páginas con manejo de errores
try:
    if page == "🏠 Home":
        from src.streamlit_app.pages import home
        home.render()
    elif page == "🔮 Predicción Simple":
        from src.streamlit_app.pages import prediction
        prediction.render()
    elif page == "🤖 Copiloto Conversacional":
        from src.streamlit_app.pages import chatbot
        chatbot.render()
except ImportError as e:
    st.error(f"""
    ❌ **Error al cargar la página**
    
    No se pudo importar el módulo: `{e.name if hasattr(e, 'name') else 'desconocido'}`
    
    **Posibles soluciones:**
    1. Verifica que todos los archivos de páginas existan en `src/streamlit_app/pages/`
    2. Asegúrate de que los archivos `__init__.py` estén presentes
    3. Ejecuta: `poetry install` para instalar dependencias
    
    **Error completo:**
    ```
    {str(e)}
    ```
    """)
    
    # Mostrar información de debug
    with st.expander("🔍 Información de Debug"):
        st.write("**Python Path:**")
        st.code("\n".join(sys.path))
        
        st.write("**Directorio actual:**")
        st.code(str(Path(__file__).parent))
        
except Exception as e:
    st.error(f"""
    ❌ **Error inesperado**
    
    Ocurrió un error al renderizar la página.
    
    **Error:**
    ```
    {str(e)}
    ```
    
    Por favor, reporta este error en GitHub Issues.
    """)
    
    import traceback
    with st.expander("🔍 Traceback completo"):
        st.code(traceback.format_exc())

# Footer en sidebar
st.sidebar.markdown("---")
st.sidebar.caption("🔧 Desarrollado por **Equipo Atreides**")
st.sidebar.caption("📅 MLOps 2025 - Proyecto Final")
st.sidebar.caption("⚡ Optimización Energética Industrial")

# Mensaje de bienvenida inicial (solo en primera carga)
if 'first_load' not in st.session_state:
    st.session_state.first_load = False
    st.balloons()
