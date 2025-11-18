"""
Página de predicción simple de consumo energético.

Esta página proporciona un formulario interactivo para realizar predicciones
de consumo energético con validación en tiempo real y visualizaciones.

User Story: US-032.2 - Predicción Simple
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, time
import json

# Importar constantes del módulo
from src.streamlit_app.pages import API_BASE_URL, COLORS, ICONS, MESSAGES, DEFAULT_TIMEOUT


def validate_inputs(data):
    """Valida los inputs del formulario."""
    errors = []
    
    if data['Usage_kWh'] < 0:
        errors.append("⚠️ El consumo no puede ser negativo")
    
    if data['Lagging_Current_Reactive.Power_kVarh'] < 0:
        errors.append("⚠️ La potencia reactiva retrasada no puede ser negativa")
    
    if data['Leading_Current_Reactive_Power_kVarh'] < 0:
        errors.append("⚠️ La potencia reactiva adelantada no puede ser negativa")
    
    if not 0 <= data['Lagging_Current_Power_Factor'] <= 1:
        errors.append("⚠️ El factor de potencia retrasada debe estar entre 0 y 1")
    
    if not 0 <= data['Leading_Current_Power_Factor'] <= 1:
        errors.append("⚠️ El factor de potencia adelantada debe estar entre 0 y 1")
    
    if data['CO2(tCO2)'] < 0:
        errors.append("⚠️ Las emisiones de CO2 no pueden ser negativas")
    
    return errors


def create_gauge_chart(value, title, max_value=100):
    """Crea un gráfico de gauge para visualizar métricas."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': max_value * 0.7},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1},
            'bar': {'color': COLORS['primary']},
            'steps': [
                {'range': [0, max_value * 0.5], 'color': 'lightgreen'},
                {'range': [max_value * 0.5, max_value * 0.75], 'color': 'yellow'},
                {'range': [max_value * 0.75, max_value], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def render():
    """Renderiza la página de predicción simple."""
    
    st.title("🔮 Predicción de Consumo Energético")
    
    st.markdown("""
        Ingresa los parámetros operacionales de tu planta para obtener una predicción 
        del consumo energético esperado.
    """)
    
    st.markdown("---")
    
    # Formulario de entrada
    with st.form("prediction_form"):
        st.subheader("📝 Parámetros de Entrada")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fecha y hora
            fecha = st.date_input(
                "📅 Fecha",
                value=datetime.now(),
                help="Selecciona la fecha de la operación"
            )
            
            hora = st.time_input(
                "🕐 Hora",
                value=time(12, 0),
                help="Hora de la medición"
            )
            
            # Tipo de carga
            load_type = st.selectbox(
                "⚙️ Tipo de Carga",
                ["Light_Load", "Medium_Load", "Maximum_Load"],
                help="Selecciona el tipo de carga operacional"
            )
            
            # Día de la semana
            day_of_week = st.selectbox(
                "📆 Día de la Semana",
                ["Monday", "Tuesday", "Wednesday", "Thursday", 
                 "Friday", "Saturday", "Sunday"],
                index=fecha.weekday()
            )
            
            # Consumo actual
            usage_kwh = st.number_input(
                "⚡ Consumo Actual (kWh)",
                min_value=0.0,
                value=50.0,
                step=1.0,
                help="Consumo energético actual en kWh"
            )
        
        with col2:
            # Potencia reactiva retrasada
            reactive_power = st.number_input(
                "🔋 Potencia Reactiva Retrasada (kVarh)",
                min_value=0.0,
                value=25.0,
                step=1.0,
                help="Potencia reactiva con corriente retrasada"
            )
            
            # Factor de potencia retrasada
            lagging_pf = st.slider(
                "📊 Factor de Potencia Retrasada",
                min_value=0.0,
                max_value=1.0,
                value=0.85,
                step=0.01,
                help="Factor de potencia con corriente retrasada"
            )
            
            # Factor de potencia adelantada
            leading_pf = st.slider(
                "📈 Factor de Potencia Adelantada",
                min_value=0.0,
                max_value=1.0,
                value=0.90,
                step=0.01,
                help="Factor de potencia con corriente adelantada"
            )
            
            # Potencia reactiva adelantada
            leading_reactive = st.number_input(
                "⚡ Potencia Reactiva Adelantada (kVarh)",
                min_value=0.0,
                value=15.0,
                step=1.0,
                help="Potencia reactiva con corriente adelantada"
            )
            
            # CO2
            co2 = st.number_input(
                "🌍 Emisiones CO2 (tCO2)",
                min_value=0.0,
                value=0.05,
                step=0.01,
                help="Emisiones de CO2 en toneladas"
            )
        
        # NSM (calculado desde hora)
        nsm = hora.hour * 3600 + hora.minute * 60 + hora.second
        
        # WeekStatus (calculado desde día)
        week_status = "Weekend" if day_of_week in ["Saturday", "Sunday"] else "Weekday"
        
        # Botón de predicción
        submitted = st.form_submit_button(
            "🚀 Realizar Predicción",
            use_container_width=True
        )
    
    if submitted:
        # Preparar datos para enviar a la API
        input_data = {
            "date": fecha.isoformat(),
            "Usage_kWh": usage_kwh,
            "Lagging_Current_Reactive.Power_kVarh": reactive_power,
            "Leading_Current_Reactive_Power_kVarh": leading_reactive,
            "CO2(tCO2)": co2,
            "Lagging_Current_Power_Factor": lagging_pf,
            "Leading_Current_Power_Factor": leading_pf,
            "NSM": nsm,
            "WeekStatus": week_status,
            "Day_of_week": day_of_week,
            "Load_Type": load_type
        }
        
        # Validar inputs
        validation_errors = validate_inputs(input_data)
        
        if validation_errors:
            for error in validation_errors:
                st.error(error)
        else:
            # Mostrar spinner mientras se hace la predicción
            with st.spinner("🔄 Realizando predicción..."):
                try:
                    # Llamar a la API
                    response = requests.post(
                        f"{API_BASE_URL}/predict",
                        json=input_data,
                        timeout=DEFAULT_TIMEOUT
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.success("✅ Predicción completada exitosamente")
                        
                        st.markdown("---")
                        st.subheader("📊 Resultados de la Predicción")
                        
                        # Métricas principales
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric(
                                "Consumo Predicho",
                                f"{result.get('predicted_usage', 0):.2f} kWh",
                                delta=f"{result.get('predicted_usage', 0) - usage_kwh:.2f} kWh"
                            )
                        
                        with metric_col2:
                            confidence = result.get('confidence', 0.85) * 100
                            st.metric(
                                "Confianza",
                                f"{confidence:.1f}%"
                            )
                        
                        with metric_col3:
                            st.metric(
                                "Modelo Utilizado",
                                result.get('model_name', 'XGBoost')
                            )
                        
                        # Visualizaciones
                        st.markdown("---")
                        viz_col1, viz_col2 = st.columns(2)
                        
                        with viz_col1:
                            # Gauge de consumo
                            gauge_fig = create_gauge_chart(
                                result.get('predicted_usage', 0),
                                "Consumo Predicho (kWh)",
                                max_value=100
                            )
                            st.plotly_chart(gauge_fig, use_container_width=True)
                        
                        with viz_col2:
                            # Gauge de eficiencia
                            efficiency = (1 - abs(result.get('predicted_usage', 0) - usage_kwh) / usage_kwh) * 100
                            efficiency_fig = create_gauge_chart(
                                efficiency,
                                "Eficiencia Operacional (%)",
                                max_value=100
                            )
                            st.plotly_chart(efficiency_fig, use_container_width=True)
                        
                        # Feature importance si está disponible
                        if 'feature_importance' in result:
                            st.markdown("---")
                            st.subheader("🎯 Factores Más Influyentes")
                            
                            importance_df = pd.DataFrame(
                                result['feature_importance'].items(),
                                columns=['Feature', 'Importancia']
                            ).sort_values('Importancia', ascending=False).head(5)
                            
                            fig = go.Figure(go.Bar(
                                x=importance_df['Importancia'],
                                y=importance_df['Feature'],
                                orientation='h',
                                marker=dict(color=COLORS['primary'])
                            ))
                            
                            fig.update_layout(
                                title="Top 5 Features Más Importantes",
                                xaxis_title="Importancia",
                                yaxis_title="Feature",
                                height=300
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Recomendaciones
                        st.markdown("---")
                        st.subheader("💡 Recomendaciones")
                        
                        if result.get('predicted_usage', 0) > usage_kwh * 1.1:
                            st.warning("""
                                ⚠️ Se predice un **aumento significativo** en el consumo energético.
                                
                                **Sugerencias:**
                                - Revisar la carga operacional
                                - Verificar el factor de potencia
                                - Considerar optimización de procesos
                            """)
                        elif result.get('predicted_usage', 0) < usage_kwh * 0.9:
                            st.info("""
                                ℹ️ Se predice una **disminución** en el consumo energético.
                                
                                **Observaciones:**
                                - Operación dentro de parámetros eficientes
                                - Continuar con prácticas actuales
                            """)
                        else:
                            st.success("""
                                ✅ Consumo predicho está **dentro del rango esperado**.
                                
                                **Status:** Operación normal
                            """)
                        
                        # Botón para descargar resultados
                        st.markdown("---")
                        result_json = json.dumps(result, indent=2)
                        st.download_button(
                            label="📥 Descargar Resultados (JSON)",
                            data=result_json,
                            file_name=f"prediction_{fecha}_{hora.hour}h.json",
                            mime="application/json"
                        )
                    
                    else:
                        st.error(f"❌ Error en la API: {response.status_code}")
                        st.json(response.json())
                
                except requests.exceptions.ConnectionError:
                    st.error(MESSAGES['api_offline'])
                
                except Exception as e:
                    st.error(f"❌ Error inesperado: {str(e)}")
