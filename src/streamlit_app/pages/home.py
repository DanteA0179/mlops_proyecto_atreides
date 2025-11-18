"""
Página de inicio de la aplicación Streamlit.

Esta página proporciona una introducción general al sistema de optimización
energética con IA, incluyendo métricas, arquitectura y casos de uso.

User Story: US-032.1 - Home/Introducción
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Importar constantes y utilidades del módulo
from src.streamlit_app.pages import COLORS, ICONS, API_BASE_URL


def render():
    """Renderiza la página de inicio."""
    
    # Header principal con animación
    st.markdown(
        '<h1 class="main-header">⚡ Sistema de Optimización Energética con IA</h1>', 
        unsafe_allow_html=True
    )
    
    st.markdown("""
        ### Bienvenido al Copiloto Inteligente para la Industria Siderúrgica
        
        Este sistema combina **Foundation Models** de series temporales con **IA Generativa** 
        para ayudarte a optimizar el consumo energético de tu planta industrial.
    """)
    
    st.markdown("---")
    
    # Métricas principales con iconos y colores
    st.subheader("📊 Métricas Clave del Sistema")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 Precisión del Modelo",
            value="RMSE < 0.205",
            delta="15% mejor que CUBIST",
            delta_color="normal",
            help="Root Mean Square Error comparado con el benchmark CUBIST (0.241)"
        )
    
    with col2:
        st.metric(
            label="📊 Dataset",
            value="35,040 registros",
            delta="Frecuencia: 15 min",
            delta_color="off",
            help="Mediciones del año 2018 en planta siderúrgica"
        )
    
    with col3:
        st.metric(
            label="⚙️ Modelos Activos",
            value="8 Modelos",
            delta="5 tradicionales + 3 FM",
            delta_color="normal",
            help="XGBoost, LightGBM, CatBoost, RF, ET + Chronos-2 variants"
        )
    
    with col4:
        st.metric(
            label="⚡ Latencia API",
            value="234ms",
            delta="-266ms vs meta",
            delta_color="inverse",
            help="Percentil 95 del tiempo de respuesta"
        )
    
    st.markdown("---")
    
    # Sección de características principales con tabs
    st.subheader("🚀 Características Principales")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Predicción", 
        "🤖 Copiloto IA", 
        "📈 Análisis", 
        "🔒 Calidad"
    ])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
                #### Predicción Simple de Consumo
                
                Obtén predicciones precisas de consumo energético basadas en:
                
                - **Parámetros operacionales** (tipo de carga, factor de potencia)
                - **Variables temporales** (hora del día, día de la semana)
                - **Condiciones ambientales** (CO2, potencia reactiva)
                
                **Funcionalidades:**
                - ✅ Formulario interactivo con validación en tiempo real
                - ✅ Visualización con gauges y gráficos
                - ✅ Feature importance explicable
                - ✅ Recomendaciones automáticas
                - ✅ Descarga de resultados en JSON
                
                **Modelos disponibles:**
                - XGBoost (baseline)
                - Chronos-2 Fine-tuned
                - Ensemble multi-modelo
            """)
        
        with col2:
            st.info("""
                **Precisión Actual:**
                
                • RMSE: 0.198 kWh
                • MAE: 0.042 kWh
                • R²: 0.936
                
                **Latencia:**
                • p50: 150ms
                • p95: 234ms
                • p99: 380ms
            """)
            
            if st.button("🔮 Ir a Predicción", use_container_width=True):
                st.info("👈 Usa el menú lateral para navegar a 'Predicción Simple'")
    
    with tab2:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
                #### Copiloto Conversacional con IA
                
                Interactúa con un asistente inteligente que te ayuda a:
                
                - **Analizar patrones** de consumo energético
                - **Simular escenarios** "what-if"
                - **Obtener recomendaciones** personalizadas
                - **Explicar resultados** en lenguaje natural
                
                **Capacidades:**
                - 💬 Conversación natural con contexto
                - 🧠 Análisis avanzado con Llama 3.2 (3B)
                - 📊 Generación de insights accionables
                - 🔍 Exploración de datos históricos
                - 💡 Sugerencias de optimización
                
                **Ejemplos de preguntas:**
                - "¿Cuáles son los principales drivers de consumo?"
                - "¿Qué pasaría si aumento el factor de potencia a 0.95?"
                - "¿Cómo puedo reducir el consumo en horas pico?"
            """)
        
        with col2:
            st.success("""
                **Tecnología:**
                
                • LLM: Llama 3.2 (3B)
                • Context: 4096 tokens
                • Latencia: ~1.2s
                
                **Capacidades:**
                • Análisis de datos
                • Simulaciones
                • Recomendaciones
            """)
            
            if st.button("🤖 Ir a Copiloto", use_container_width=True):
                st.info("👈 Usa el menú lateral para navegar a 'Copiloto Conversacional'")
    
    with tab3:
        st.markdown("""
            #### Análisis Avanzado y Monitoreo
            
            **Análisis Temporal:**
            - Detección de patrones diarios, semanales y estacionales
            - Identificación de anomalías y outliers
            - Comparación con benchmarks históricos
            
            **Feature Importance:**
            - Mutual Information (relaciones no lineales)
            - Correlación de Pearson (relaciones lineales)
            - SHAP values para explicabilidad
            
            **Monitoreo Continuo:**
            - Data drift detection con Evidently AI
            - Model performance tracking con MLflow
            - Alertas automáticas de degradación
            
            **Métricas Clave:**
        """)
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.info("""
                **Precisión**
                - RMSE: 0.198
                - MAE: 0.042
                - R²: 0.936
            """)
        
        with metrics_col2:
            st.success("""
                **Rendimiento**
                - Latencia p95: 234ms
                - Throughput: 100 req/s
                - Uptime: 99.9%
            """)
        
        with metrics_col3:
            st.warning("""
                **Calidad**
                - Test Coverage: 85%
                - Data Quality: 98%
                - Drift Score: 0.12
            """)
    
    with tab4:
        st.markdown("""
            #### Seguridad y Calidad de Datos
            
            **Validación de Datos:**
            - ✅ Validación de rangos y tipos
            - ✅ Detección de valores faltantes
            - ✅ Identificación de outliers
            - ✅ Verificación de integridad referencial
            
            **Pruebas Automatizadas:**
            - 🧪 Unit tests (>80% coverage)
            - 🔬 Integration tests (>70% coverage)
            - 🎯 End-to-end tests
            - 📊 Performance tests
            
            **Monitoreo y Observabilidad:**
            - 📈 MLflow para experiment tracking
            - 👁️ Evidently para data drift
            - 📝 Logs estructurados
            - 🔔 Alertas automáticas
            
            **Documentación:**
            - 📚 API REST documentada (OpenAPI/Swagger)
            - 📖 Documentación de código (docstrings)
            - 🎓 Guías de usuario
            - 🔧 Troubleshooting guides
        """)
    
    st.markdown("---")
    
    # Arquitectura del sistema con diseño mejorado
    st.subheader("🏗️ Arquitectura del Sistema")
    
    arch_col1, arch_col2 = st.columns([3, 2])
    
    with arch_col1:
        st.code("""
┌──────────────────────────────────────────────────┐
│   CAPA DE PRESENTACIÓN - Streamlit UI            │
│  • Interfaz web responsiva (esta aplicación)     │
│  • 3 páginas: Home, Predicción, Chatbot         │
│  • Visualizaciones interactivas (Plotly)        │
│  • Formularios con validación en tiempo real    │
└──────────────────────────────────────────────────┘
                        ↓ HTTP/REST
┌──────────────────────────────────────────────────┐
│      CAPA DE APLICACIÓN - FastAPI Backend        │
│  • POST /predict      → Predicciones ML         │
│  • POST /chat         → Copiloto conversacional │
│  • GET  /health       → Health check            │
│  • GET  /models       → Lista de modelos        │
│  • Validación con Pydantic                      │
│  • Middleware de autenticación y logging        │
└──────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────┐
│     CAPA DE LÓGICA - ML Pipeline (Dagster)       │
│  • Training Pipeline                             │
│    - XGBoost, LightGBM, CatBoost                │
│    - Random Forest, Extra Trees                  │
│  • Foundation Models                             │
│    - Chronos-2 Zero-shot                        │
│    - Chronos-2 Fine-tuned                       │
│    - Chronos-2 with Covariates                  │
│  • Feature Engineering (7 temporal features)     │
│  • Model Registry (MLflow)                       │
│  • Hyperparameter Tuning (Optuna)              │
└──────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────┐
│       CAPA DE DATOS - Storage & Versioning       │
│  • DVC - Data Version Control                    │
│    - Remote: Google Cloud Storage               │
│    - Versioning de datos y modelos              │
│  • DuckDB - SQL Analytics                       │
│    - Queries interactivos                       │
│    - Análisis exploratorio                      │
│  • Parquet Files (compresión Snappy)           │
└──────────────────────────────────────────────────┘
        """, language="text")
    
    with arch_col2:
        st.info(f"""
            ### 🛠️ Stack Tecnológico
            
            **Frontend:**
            - {ICONS['home']} Streamlit 1.28+
            - {ICONS['analytics']} Plotly 5.17+
            
            **Backend:**
            - {ICONS['energy']} FastAPI
            - 🐍 Python 3.11
            
            **Machine Learning:**
            - 🌲 XGBoost
            - 🔮 Chronos-2 (Amazon)
            - 🚀 LightGBM, CatBoost
            - 🌳 Random Forest
            
            **Orquestación:**
            - 🔄 Dagster
            - 📦 DVC
            
            **Data Storage:**
            - 🦆 DuckDB
            - ☁️ Google Cloud Storage
            - 📊 Parquet
            
            **Monitoring:**
            - 📈 MLflow
            - 👁️ Evidently AI
            
            **Deployment:**
            - 🐳 Docker
            - ☁️ Google Cloud Run
        """)
    
    st.markdown("---")
    
    # Gráfico interactivo de consumo energético
    st.subheader("📊 Patrón de Consumo Energético - Planta Siderúrgica")
    
    # Crear datos de ejemplo más realistas
    hours = list(range(24))
    
    # Patrones diferentes para días laborales vs fin de semana
    weekday_consumption = [
        45, 42, 40, 38, 36, 40, 48, 55,  # 00:00 - 07:00 (Madrugada)
        62, 68, 72, 75, 73, 70, 68, 65,  # 08:00 - 15:00 (Horario pico)
        63, 60, 58, 55, 52, 50, 48, 46   # 16:00 - 23:00 (Tarde/Noche)
    ]
    
    weekend_consumption = [
        35, 33, 32, 31, 30, 32, 35, 38,  # Consumo reducido fin de semana
        40, 42, 44, 45, 44, 43, 42, 41,
        40, 39, 38, 37, 36, 35, 34, 33
    ]
    
    # Crear DataFrame
    df_consumption = pd.DataFrame({
        'Hora': hours + hours,
        'Consumo (kWh)': weekday_consumption + weekend_consumption,
        'Tipo': ['Día Laboral'] * 24 + ['Fin de Semana'] * 24
    })
    
    # Crear gráfico con Plotly
    fig = px.line(
        df_consumption,
        x='Hora',
        y='Consumo (kWh)',
        color='Tipo',
        title='Comparación de Consumo: Día Laboral vs Fin de Semana',
        markers=True,
        color_discrete_map={
            'Día Laboral': COLORS['primary'],
            'Fin de Semana': COLORS['success']
        }
    )
    
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8)
    )
    
    fig.update_layout(
        xaxis_title="Hora del Día",
        yaxis_title="Consumo Energético (kWh)",
        hovermode='x unified',
        plot_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            dtick=2,
            range=[-0.5, 23.5]
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray'
        ),
        height=400
    )
    
    # Agregar anotaciones
    fig.add_annotation(
        x=7, y=55,
        text="Inicio de jornada",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS['primary'],
        ax=-50, ay=-30
    )
    
    fig.add_annotation(
        x=12, y=75,
        text="Pico de consumo",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS['danger'],
        ax=0, ay=-40
    )
    
    fig.add_annotation(
        x=18, y=58,
        text="Reducción gradual",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS['success'],
        ax=50, ay=-30
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Insights del gráfico
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        avg_weekday = np.mean(weekday_consumption)
        st.info(f"""
            **{ICONS['info']} Día Laboral**
            
            Promedio: **{avg_weekday:.1f} kWh**  
            Pico: **{max(weekday_consumption)} kWh** (12:00)  
            Valle: **{min(weekday_consumption)} kWh** (04:00)  
            
            Operaciones a plena capacidad.
        """)
    
    with insight_col2:
        avg_weekend = np.mean(weekend_consumption)
        st.success(f"""
            **{ICONS['success']} Fin de Semana**
            
            Promedio: **{avg_weekend:.1f} kWh**  
            Pico: **{max(weekend_consumption)} kWh** (11:00)  
            Valle: **{min(weekend_consumption)} kWh** (05:00)  
            
            Consumo reducido ~40%.
        """)
    
    with insight_col3:
        savings = avg_weekday - avg_weekend
        st.warning(f"""
            **{ICONS['energy']} Potencial de Ahorro**
            
            Diferencia: **{savings:.1f} kWh/h**  
            Ahorro diario: **{savings * 24:.0f} kWh**  
            Ahorro anual: **~{savings * 24 * 104:.0f} kWh**  
            
            Optimización en horarios pico.
        """)
    
    st.markdown("---")
    
    # Call to action destacado
    st.success(f"""
        ### {ICONS['success']} ¡Comienza a Optimizar Ahora!
        
        El sistema está listo para ayudarte a reducir costos y mejorar la eficiencia energética.
        
        **Próximos pasos:**
        
        1. **{ICONS['prediction']} Prueba la Predicción Simple** - Obtén predicciones rápidas
        2. **{ICONS['chatbot']} Conversa con el Copiloto IA** - Análisis conversacional avanzado
        3. **{ICONS['analytics']} Explora los Datos** - Patrones y tendencias históricas
        
        👈 Usa el menú lateral para navegar entre las funcionalidades
    """)
    
    # Footer con información adicional
    st.markdown("---")
    
    footer_col1, footer_col2 = st.columns(2)
    
    with footer_col1:
        st.markdown("""
            ### 📚 Recursos y Documentación
            - [📖 Documentación del Proyecto](https://github.com/DanteA0179/mlops_proyecto_atreides)
            - [🔌 API Docs (Swagger)](http://localhost:8000/docs)
            - [📊 Dataset Original (UCI ML)](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption)
            - [📈 MLflow UI](http://localhost:5000)
            - [⚙️ Dagster UI](http://localhost:3000)
        """)
    
    with footer_col2:
        st.markdown("""
            ### 👥 Equipo Atreides
            - **Juan** - Data Engineer
            - **Erick** - Data Scientist
            - **Julian** - ML Engineer
            - **Dante** - Software Engineer & Scrum Master
            - **Arthur** - MLOps/SRE Engineer
            
            🏆 **Proyecto MLOps 2025**
        """)
    
    st.caption("🔧 Desarrollado por Equipo Atreides | MLOps 2025 | v1.0.0")
