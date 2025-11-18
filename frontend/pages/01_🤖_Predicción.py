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
