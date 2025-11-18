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
