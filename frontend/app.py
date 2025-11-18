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
