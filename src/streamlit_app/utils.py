"""
Utilidades para la aplicación Streamlit.

Este módulo contiene funciones helper compartidas por todas las páginas
de la aplicación Streamlit.

Funciones principales:
    - check_api_health: Verifica estado de la API
    - format_metric_value: Formatea valores de métricas
    - load_sample_data: Carga datos de muestra
    - calculate_*: Funciones de cálculo (costos, eficiencia, CO2)
    - create_*: Funciones para crear visualizaciones
    - validate_*: Funciones de validación

Autor: Equipo Atreides
Fecha: 2025
"""

import streamlit as st
from typing import Dict, Any, Optional, Tuple
import plotly.graph_objects as go
import requests
import os


def check_api_health(api_url: str, timeout: int = 5) -> Tuple[bool, Optional[Dict]]:
    """
    Verifica si la API está disponible y devuelve información de salud.
    
    Args:
        api_url: URL base de la API
        timeout: Tiempo de espera en segundos
        
    Returns:
        Tupla (is_healthy: bool, health_data: dict o None)
        
    Example:
        >>> is_healthy, data = check_api_health("http://localhost:8000")
        >>> if is_healthy:
        ...     print("API está online")
    """
    try:
        response = requests.get(f"{api_url}/health", timeout=timeout)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except requests.exceptions.RequestException:
        return False, None


def format_metric_value(value: float, metric_type: str = "energy") -> str:
    """
    Formatea valores de métricas para mostrar.
    
    Args:
        value: Valor a formatear
        metric_type: Tipo de métrica ('energy', 'percentage', 'co2', 'power_factor', 'currency')
        
    Returns:
        String formateado
        
    Example:
        >>> format_metric_value(50.5, "energy")
        '50.50 kWh'
        >>> format_metric_value(0.85, "power_factor")
        '0.850'
    """
    if metric_type == "energy":
        return f"{value:.2f} kWh"
    elif metric_type == "percentage":
        return f"{value:.1f}%"
    elif metric_type == "co2":
        return f"{value:.4f} tCO2"
    elif metric_type == "power_factor":
        return f"{value:.3f}"
    elif metric_type == "currency":
        return f"${value:,.2f}"
    else:
        return f"{value:.2f}"


def create_status_badge(status: str, message: str = "") -> None:
    """
    Crea un badge de estado visual.
    
    Args:
        status: Estado ('success', 'warning', 'error', 'info')
        message: Mensaje a mostrar
        
    Example:
        >>> create_status_badge('success', 'Operación completada')
    """
    status_config = {
        'success': {'icon': '✅', 'color': 'green'},
        'warning': {'icon': '⚠️', 'color': 'orange'},
        'error': {'icon': '❌', 'color': 'red'},
        'info': {'icon': 'ℹ️', 'color': 'blue'}
    }
    
    config = status_config.get(status, status_config['info'])
    
    st.markdown(
        f"""<div style="padding: 10px; border-left: 4px solid {config['color']}; background-color: rgba(128,128,128,0.1); margin: 10px 0;">
        {config['icon']} {message}
        </div>""",
        unsafe_allow_html=True
    )


@st.cache_data(ttl=300)  # Cache por 5 minutos
def load_sample_data():
    """
    Carga datos de muestra para demostraciones.
    
    Returns:
        DataFrame con datos de ejemplo
        
    Example:
        >>> df = load_sample_data()
        >>> print(df.columns)
        Index(['hour', 'consumption', 'is_peak'], dtype='object')
    """
    import pandas as pd
    import numpy as np
    
    # Generar datos de muestra realistas
    hours = range(24)
    
    # Patrón de consumo típico de planta industrial
    base_consumption = 40
    peak_hours = [8, 9, 10, 11, 12, 13, 14, 15]  # Horario pico
    
    consumption = []
    for hour in hours:
        if hour in peak_hours:
            # Mayor consumo en horario pico
            value = base_consumption + np.random.normal(25, 5)
        elif 6 <= hour <= 18:
            # Consumo medio durante día
            value = base_consumption + np.random.normal(15, 3)
        else:
            # Consumo bajo en noche
            value = base_consumption + np.random.normal(-5, 2)
        
        consumption.append(max(value, 20))  # Mínimo de 20 kWh
    
    return pd.DataFrame({
        'hour': hours,
        'consumption': consumption,
        'is_peak': [h in peak_hours for h in hours]
    })


def calculate_energy_cost(
    consumption_kwh: float,
    rate_per_kwh: float = 0.15,
    peak_multiplier: float = 1.5,
    is_peak_hour: bool = False
) -> Dict[str, float]:
    """
    Calcula el costo de energía basado en consumo y tarifas.
    
    Args:
        consumption_kwh: Consumo en kWh
        rate_per_kwh: Tarifa base por kWh
        peak_multiplier: Multiplicador para horas pico
        is_peak_hour: Si es hora pico
        
    Returns:
        Diccionario con desglose de costos
        
    Example:
        >>> cost = calculate_energy_cost(100, 0.15, 1.5, True)
        >>> print(f"Costo total: ${cost['total_cost']:.2f}")
    """
    effective_rate = rate_per_kwh * (peak_multiplier if is_peak_hour else 1.0)
    base_cost = consumption_kwh * rate_per_kwh
    total_cost = consumption_kwh * effective_rate
    surcharge = total_cost - base_cost
    
    return {
        'base_cost': base_cost,
        'surcharge': surcharge,
        'total_cost': total_cost,
        'effective_rate': effective_rate,
        'savings_potential': surcharge if is_peak_hour else 0
    }


def calculate_power_factor_efficiency(
    lagging_pf: float,
    leading_pf: float
) -> Dict[str, Any]:
    """
    Calcula métricas de eficiencia del factor de potencia.
    
    Args:
        lagging_pf: Factor de potencia retrasada
        leading_pf: Factor de potencia adelantada
        
    Returns:
        Diccionario con análisis de eficiencia
        
    Example:
        >>> efficiency = calculate_power_factor_efficiency(0.85, 0.90)
        >>> print(efficiency['efficiency_class'])
        'Bueno'
    """
    avg_pf = (lagging_pf + leading_pf) / 2
    
    # Clasificación de eficiencia
    if avg_pf >= 0.95:
        efficiency_class = "Excelente"
        efficiency_score = 100
        color = "green"
    elif avg_pf >= 0.90:
        efficiency_class = "Muy Bueno"
        efficiency_score = 90
        color = "lightgreen"
    elif avg_pf >= 0.85:
        efficiency_class = "Bueno"
        efficiency_score = 75
        color = "yellow"
    elif avg_pf >= 0.75:
        efficiency_class = "Aceptable"
        efficiency_score = 60
        color = "orange"
    else:
        efficiency_class = "Mejorable"
        efficiency_score = 40
        color = "red"
    
    # Calcular potencial de mejora
    improvement_potential = (0.95 - avg_pf) * 100  # Objetivo: 0.95
    
    return {
        'average_pf': avg_pf,
        'efficiency_class': efficiency_class,
        'efficiency_score': efficiency_score,
        'color': color,
        'improvement_potential': max(0, improvement_potential),
        'lagging_pf': lagging_pf,
        'leading_pf': leading_pf
    }


def calculate_co2_impact(
    consumption_kwh: float,
    co2_total: float
) -> Dict[str, Any]:
    """
    Calcula métricas de impacto ambiental.
    
    Args:
        consumption_kwh: Consumo en kWh
        co2_total: Emisiones totales de CO2 en tCO2
        
    Returns:
        Diccionario con análisis de impacto
        
    Example:
        >>> impact = calculate_co2_impact(100, 0.05)
        >>> print(f"Clase: {impact['impact_class']}")
    """
    co2_per_kwh = co2_total / max(consumption_kwh, 0.001)
    
    # Benchmarks de industria (valores aproximados)
    excellent_threshold = 0.0008
    good_threshold = 0.0015
    acceptable_threshold = 0.0025
    
    if co2_per_kwh <= excellent_threshold:
        impact_class = "Bajo Impacto"
        impact_score = 100
        color = "green"
    elif co2_per_kwh <= good_threshold:
        impact_class = "Impacto Moderado Bajo"
        impact_score = 80
        color = "lightgreen"
    elif co2_per_kwh <= acceptable_threshold:
        impact_class = "Impacto Moderado"
        impact_score = 60
        color = "yellow"
    else:
        impact_class = "Alto Impacto"
        impact_score = 40
        color = "red"
    
    # Equivalencias para contexto
    trees_equivalent = co2_total * 50  # Aprox. árboles necesarios para compensar
    car_km_equivalent = co2_total * 4500  # Aprox. km recorridos en auto
    
    return {
        'co2_per_kwh': co2_per_kwh,
        'co2_total': co2_total,
        'impact_class': impact_class,
        'impact_score': impact_score,
        'color': color,
        'trees_equivalent': trees_equivalent,
        'car_km_equivalent': car_km_equivalent
    }


def create_comparison_chart(
    actual: float,
    predicted: float,
    title: str = "Comparación",
    unit: str = "kWh"
) -> go.Figure:
    """
    Crea un gráfico de comparación entre valor actual y predicho.
    
    Args:
        actual: Valor actual
        predicted: Valor predicho
        title: Título del gráfico
        unit: Unidad de medida
        
    Returns:
        Figura de Plotly
        
    Example:
        >>> fig = create_comparison_chart(50, 55, "Consumo", "kWh")
        >>> st.plotly_chart(fig)
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Actual', 'Predicho'],
        y=[actual, predicted],
        text=[f'{actual:.2f} {unit}', f'{predicted:.2f} {unit}'],
        textposition='auto',
        marker=dict(
            color=['lightblue', 'lightcoral'],
            line=dict(color='darkblue', width=2)
        )
    ))
    
    fig.update_layout(
        title=title,
        yaxis_title=f"Valor ({unit})",
        showlegend=False,
        height=400,
        template='plotly_white'
    )
    
    return fig


def format_time_ago(timestamp: str) -> str:
    """
    Formatea un timestamp como "hace X tiempo".
    
    Args:
        timestamp: Timestamp en formato ISO
        
    Returns:
        String formateado
        
    Example:
        >>> format_time_ago("2025-01-15T10:00:00")
        'hace 2 horas'
    """
    from datetime import datetime
    
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        now = datetime.now(dt.tzinfo)
        diff = now - dt
        
        seconds = diff.total_seconds()
        
        if seconds < 60:
            return "hace unos segundos"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"hace {minutes} minuto{'s' if minutes != 1 else ''}"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"hace {hours} hora{'s' if hours != 1 else ''}"
        else:
            days = int(seconds / 86400)
            return f"hace {days} día{'s' if days != 1 else ''}"
    except:
        return "desconocido"


def validate_api_response(response: requests.Response) -> Tuple[bool, str, Optional[Dict]]:
    """
    Valida una respuesta de la API y extrae información útil.
    
    Args:
        response: Respuesta de requests
        
    Returns:
        Tupla (is_valid: bool, message: str, data: dict o None)
        
    Example:
        >>> response = requests.get("http://localhost:8000/health")
        >>> is_valid, msg, data = validate_api_response(response)
        >>> if is_valid:
        ...     print(f"Éxito: {msg}")
    """
    if response.status_code == 200:
        try:
            data = response.json()
            return True, "✅ Respuesta exitosa", data
        except:
            return False, "❌ Error al parsear respuesta JSON", None
    
    elif response.status_code == 422:
        return False, "❌ Error de validación en los datos enviados", None
    
    elif response.status_code == 500:
        return False, "❌ Error interno del servidor", None
    
    elif response.status_code == 503:
        return False, "❌ Servicio no disponible temporalmente", None
    
    else:
        return False, f"❌ Error desconocido (código {response.status_code})", None


def get_api_url() -> str:
    """
    Obtiene la URL de la API desde variables de entorno o default.
    
    Returns:
        URL de la API
        
    Example:
        >>> api_url = get_api_url()
        >>> print(api_url)
        'http://localhost:8000'
    """
    return os.getenv("API_URL", "http://localhost:8000")


def display_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    help_text: Optional[str] = None,
    icon: str = "📊"
) -> None:
    """
    Muestra una tarjeta de métrica personalizada.
    
    Args:
        label: Etiqueta de la métrica
        value: Valor principal
        delta: Valor de cambio (opcional)
        help_text: Texto de ayuda (opcional)
        icon: Emoji o icono (opcional)
        
    Example:
        >>> display_metric_card("Consumo", "50.5 kWh", "+5.2 kWh", "Comparado con ayer", "⚡")
    """
    delta_html = f'<div style="font-size: 14px; color: gray;">{delta}</div>' if delta else ''
    help_html = f'<div style="font-size: 12px; color: gray; margin-top: 5px;">{help_text}</div>' if help_text else ''
    
    st.markdown(
        f"""
        <div style="
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 10px 0;
        ">
            <div style="font-size: 14px; color: #666; margin-bottom: 5px;">
                {icon} {label}
            </div>
            <div style="font-size: 24px; font-weight: bold; color: #1f77b4;">
                {value}
            </div>
            {delta_html}
            {help_html}
        </div>
        """,
        unsafe_allow_html=True
    )


@st.cache_data(ttl=60)
def get_system_stats() -> Dict[str, Any]:
    """
    Obtiene estadísticas del sistema (cacheadas).
    
    Returns:
        Diccionario con estadísticas
        
    Example:
        >>> stats = get_system_stats()
        >>> print(stats['timestamp'])
    """
    from datetime import datetime
    
    return {
        'timestamp': datetime.now().isoformat(),
        'uptime': 'N/A',
        'requests_count': 0,
        'avg_response_time': 0,
    }
