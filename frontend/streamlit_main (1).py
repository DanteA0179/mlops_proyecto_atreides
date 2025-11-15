"""
Aplicación Streamlit Multi-página para Sistema de Optimización Energética
Proyecto MLOps - Atreides
"""

import streamlit as st
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Energy Optimizer - Atreides",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/DanteA0179/mlops_proyecto_atreides',
        'Report a bug': 'https://github.com/DanteA0179/mlops_proyecto_atreides/issues',
        'About': "Sistema de Optimización Energética con IA para la Industria Siderúrgica"
    }
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-card {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar con información del proyecto
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/factory.png", width=80)
    st.title("⚡ Energy Optimizer")
    st.markdown("---")
    
    st.markdown("### 📊 Proyecto MLOps")
    st.markdown("""
    **Sistema de Optimización Energética**  
    Industria Siderúrgica
    """)
    
    st.markdown("---")
    
    # Información del equipo
    st.markdown("### 👥 Equipo Atreides")
    st.markdown("""
    - **Data Engineer**: Juan
    - **Data Scientist**: Erick
    - **ML Engineer**: Julian
    - **Software Engineer**: Dante
    - **MLOps/SRE**: Arthur
    """)
    
    st.markdown("---")
    
    # Métricas del modelo
    st.markdown("### 🎯 Métricas del Modelo")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("RMSE", "< 0.205", "-15%")
    with col2:
        st.metric("MAE", "< 0.046", "-16%")
    
    st.markdown("---")
    
    # Links útiles
    st.markdown("### 🔗 Enlaces")
    st.markdown("""
    - [📘 GitHub](https://github.com/DanteA0179/mlops_proyecto_atreides)
    - [📊 MLflow](http://localhost:5000)
    - [🔄 Dagster](http://localhost:3000)
    - [🚀 API Docs](http://localhost:8000/docs)
    """)

# Contenido principal
st.title("⚡ Sistema de Optimización Energética con IA")
st.markdown("### Copiloto Inteligente para la Industria Siderúrgica")

# Tabs de navegación
tab1, tab2, tab3 = st.tabs(["🏠 Inicio", "🔮 Predicción", "💬 Copiloto IA"])

with tab1:
    st.markdown("## Bienvenido al Sistema de Optimización Energética")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Objetivo del Proyecto
        
        Desarrollar un sistema MLOps completo que:
        
        - ✅ **Predice consumo energético** con RMSE < 0.205 (15% mejor que benchmark CUBIST)
        - ✅ **Explica drivers de consumo** mediante análisis conversacional
        - ✅ **Optimiza operaciones industriales** a través de simulaciones "what-if"
        
        ---
        
        ### 📊 Dataset
        
        **Fuente**: [UCI ML Repository - Steel Industry Energy Consumption](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption)
        
        - **Registros**: 35,040 mediciones (año 2018)
        - **Frecuencia**: 15 minutos
        - **Variable objetivo**: `Usage_kWh` (consumo energético)
        
        ---
        
        ### 🤖 Modelos Implementados
        
        #### Modelos Tradicionales
        - XGBoost, LightGBM, CatBoost
        - Random Forest, Gradient Boosting
        
        #### Foundation Models
        - **Chronos-2** (Amazon): Zero-shot + Fine-tuning
        - Soporte para 9 covariables temporales
        
        ---
        
        ### 🛠️ Stack Tecnológico
        
        **Data & ML**
        - Polars, Pandas, NumPy, DuckDB
        - Scikit-learn, XGBoost, LightGBM
        - PyTorch, Transformers
        
        **MLOps**
        - DVC (versionado de datos)
        - MLflow (tracking de experimentos)
        - Dagster (orquestación)
        - Evidently (monitoreo)
        
        **Backend & Deployment**
        - FastAPI, Docker
        - Google Cloud Run
        - Streamlit
        
        **LLM & AI**
        - Ollama (inferencia local)
        - Llama 3.2 (3B)
        - LangChain
        """)
    
    with col2:
        # Tarjetas de características
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 🎨 Características")
        st.markdown("""
        **Features Temporales**
        - 7 features engineered
        - Codificación cíclica
        - Patrones temporales
        
        **Validación**
        - Test coverage > 70%
        - Pre-commit hooks
        - CI/CD automatizado
        
        **Monitoreo**
        - Data drift detection
        - Model performance
        - API health checks
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Métricas destacadas
        st.markdown("### 📈 Benchmark vs Meta")
        
        metrics_data = {
            "Métrica": ["RMSE", "MAE", "CV (%)"],
            "Benchmark": [0.2410, 0.0547, 0.8770],
            "Meta": [0.2050, 0.0460, 0.7500],
            "Mejora": ["15%", "16%", "14%"]
        }
        
        import pandas as pd
        st.dataframe(
            pd.DataFrame(metrics_data),
            hide_index=True,
            use_container_width=True
        )

with tab2:
    st.info("👉 Por favor, navega a la página **'🔮 Predicción'** en la barra lateral para realizar predicciones.")
    st.markdown("""
    ### ¿Qué puedes hacer en la página de Predicción?
    
    - Ingresar parámetros operacionales de la planta
    - Obtener predicciones de consumo energético en tiempo real
    - Visualizar resultados con gráficos interactivos
    - Comparar escenarios "what-if"
    """)

with tab3:
    st.info("👉 Por favor, navega a la página **'💬 Copiloto IA'** en la barra lateral para chatear con el asistente.")
    st.markdown("""
    ### ¿Qué puedes hacer con el Copiloto IA?
    
    - Hacer preguntas sobre consumo energético
    - Obtener explicaciones de las predicciones
    - Analizar patrones de consumo
    - Recibir recomendaciones de optimización
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>Proyecto MLOps - Equipo Atreides</strong></p>
    <p>Sistema de Optimización Energética con IA | Última actualización: Noviembre 2025</p>
</div>
""", unsafe_allow_html=True)
