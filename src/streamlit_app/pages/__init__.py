"""
Módulo de páginas de la aplicación Streamlit.

Este paquete contiene todas las páginas de la interfaz web del sistema
de optimización energética con IA.

Páginas disponibles:
    - home: Página de inicio con información general del sistema
    - prediction: Página de predicción simple con formulario interactivo
    - chatbot: Copiloto conversacional con IA generativa

Estructura:
    pages/
    ├── __init__.py         # Este archivo
    ├── home.py            # Página de inicio (US-032.1)
    ├── prediction.py      # Predicción simple (US-032.2)
    └── chatbot.py         # Copiloto conversacional (US-032.3)

Uso:
    from src.streamlit_app.pages import home, prediction, chatbot
    
    # Renderizar página de inicio
    home.render()
    
    # Renderizar página de predicción
    prediction.render()
    
    # Renderizar copiloto conversacional
    chatbot.render()

Autor: Equipo Atreides
Proyecto: MLOps - Optimización Energética Industrial
Fecha: 2025
Versión: 1.0.0
"""

# Importar módulos de páginas para facilitar el acceso
try:
    from . import home
    from . import prediction
    from . import chatbot
    
    __all__ = ['home', 'prediction', 'chatbot']
    
except ImportError as e:
    # Si hay error en la importación, registrarlo pero no fallar
    import warnings
    warnings.warn(
        f"No se pudieron importar todas las páginas: {e}. "
        "Algunas funcionalidades pueden no estar disponibles.",
        ImportWarning
    )
    __all__ = []

# Metadata del módulo
__version__ = '1.0.0'
__author__ = 'Equipo Atreides'
__email__ = 'equipo.atreides@mlops.com'
__status__ = 'Production'

# Información de las páginas disponibles
PAGES_INFO = {
    'home': {
        'title': '🏠 Home',
        'description': 'Página de inicio con información general del sistema',
        'module': 'home',
        'function': 'render',
        'requirements': [],
        'user_story': 'US-032.1'
    },
    'prediction': {
        'title': '🔮 Predicción Simple',
        'description': 'Formulario de predicción de consumo energético',
        'module': 'prediction',
        'function': 'render',
        'requirements': ['requests', 'plotly'],
        'user_story': 'US-032.2'
    },
    'chatbot': {
        'title': '🤖 Copiloto Conversacional',
        'description': 'Asistente conversacional con IA para análisis energético',
        'module': 'chatbot',
        'function': 'render',
        'requirements': ['requests'],
        'user_story': 'US-032.3'
    }
}

def get_available_pages():
    """
    Retorna una lista de páginas disponibles en el sistema.
    
    Returns:
        list: Lista de nombres de páginas disponibles
        
    Example:
        >>> from src.streamlit_app.pages import get_available_pages
        >>> pages = get_available_pages()
        >>> print(pages)
        ['home', 'prediction', 'chatbot']
    """
    return list(PAGES_INFO.keys())

def get_page_info(page_name: str):
    """
    Obtiene información detallada de una página específica.
    
    Args:
        page_name (str): Nombre de la página ('home', 'prediction', 'chatbot')
        
    Returns:
        dict: Diccionario con información de la página o None si no existe
        
    Example:
        >>> from src.streamlit_app.pages import get_page_info
        >>> info = get_page_info('prediction')
        >>> print(info['title'])
        '🔮 Predicción Simple'
    """
    return PAGES_INFO.get(page_name)

def validate_page_requirements(page_name: str):
    """
    Valida que todos los requisitos de una página estén instalados.
    
    Args:
        page_name (str): Nombre de la página a validar
        
    Returns:
        tuple: (bool, list) - (todos_instalados, lista_de_faltantes)
        
    Example:
        >>> from src.streamlit_app.pages import validate_page_requirements
        >>> is_valid, missing = validate_page_requirements('prediction')
        >>> if not is_valid:
        ...     print(f"Faltan dependencias: {missing}")
    """
    import importlib.util
    
    page_info = PAGES_INFO.get(page_name)
    if not page_info:
        return False, [f"Página '{page_name}' no existe"]
    
    requirements = page_info.get('requirements', [])
    missing = []
    
    for req in requirements:
        spec = importlib.util.find_spec(req)
        if spec is None:
            missing.append(req)
    
    return len(missing) == 0, missing

# Función helper para verificar salud del módulo
def check_module_health():
    """
    Verifica el estado de salud del módulo de páginas.
    
    Returns:
        dict: Estado de salud con información de cada página
        
    Example:
        >>> from src.streamlit_app.pages import check_module_health
        >>> health = check_module_health()
        >>> print(health['status'])
        'healthy'
    """
    health_status = {
        'status': 'healthy',
        'pages': {},
        'total_pages': len(PAGES_INFO),
        'available_pages': 0,
        'unavailable_pages': []
    }
    
    for page_name, page_info in PAGES_INFO.items():
        is_valid, missing = validate_page_requirements(page_name)
        
        page_status = {
            'available': is_valid,
            'missing_requirements': missing,
            'user_story': page_info['user_story']
        }
        
        health_status['pages'][page_name] = page_status
        
        if is_valid:
            health_status['available_pages'] += 1
        else:
            health_status['unavailable_pages'].append(page_name)
    
    # Determinar estado general
    if health_status['unavailable_pages']:
        if health_status['available_pages'] == 0:
            health_status['status'] = 'critical'
        else:
            health_status['status'] = 'degraded'
    
    return health_status

# Constantes útiles para las páginas
API_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3

# Configuración de colores para consistencia visual
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#2c3e50',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

# Configuración de iconos
ICONS = {
    'home': '🏠',
    'prediction': '🔮',
    'chatbot': '🤖',
    'energy': '⚡',
    'analytics': '📊',
    'settings': '⚙️',
    'help': '❓',
    'success': '✅',
    'error': '❌',
    'warning': '⚠️',
    'info': 'ℹ️'
}

# Mensajes comunes
MESSAGES = {
    'api_offline': """
    ❌ **API No Disponible**
    
    La API del backend no está respondiendo. Por favor:
    1. Verifica que la API esté corriendo en `http://localhost:8000`
    2. Ejecuta: `poetry run uvicorn src.api.main:app --reload`
    3. Revisa los logs del servidor
    """,
    
    'loading': "🔄 Cargando datos...",
    'processing': "⚙️ Procesando solicitud...",
    'success': "✅ Operación completada exitosamente",
    'error_generic': "❌ Ocurrió un error inesperado. Por favor, intenta nuevamente.",
}
