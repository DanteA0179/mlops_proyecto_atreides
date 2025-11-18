"""
Módulo de aplicación Streamlit para Energy Optimization AI.

Este paquete contiene la interfaz web del sistema de optimización energética
con IA para la industria siderúrgica.

Estructura:
    streamlit_app/
    ├── __init__.py         # Este archivo
    ├── app.py             # Aplicación principal
    ├── utils.py           # Funciones utilitarias
    └── pages/             # Páginas de la aplicación
        ├── __init__.py
        ├── home.py
        ├── prediction.py
        └── chatbot.py

Uso:
    Para ejecutar la aplicación:
    
    ```bash
    streamlit run src/streamlit_app/app.py
    ```
    
    O con Poetry:
    
    ```bash
    poetry run streamlit run src/streamlit_app/app.py
    ```

User Stories Implementadas:
    - US-032: Streamlit UI - Página de predicción
        - US-032.1: Home/Introducción
        - US-032.2: Predicción Simple
        - US-032.3: Copiloto Conversacional

Autor: Equipo Atreides
Proyecto: MLOps - Optimización Energética Industrial
Fecha: 2025
"""

# Metadata del módulo
__version__ = "1.0.0"
__author__ = "Equipo Atreides"
__email__ = "equipo.atreides@mlops.com"
__status__ = "Production"
__license__ = "MIT"

# Importar componentes principales
try:
    from . import pages
    from . import utils
    
    __all__ = ['pages', 'utils']
    
except ImportError as e:
    import warnings
    warnings.warn(
        f"No se pudieron importar todos los módulos de streamlit_app: {e}",
        ImportWarning
    )
    __all__ = []

# Configuración de la aplicación
APP_CONFIG = {
    'title': 'Energy Optimization AI - Steel Industry',
    'icon': '⚡',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'version': __version__,
    'pages': ['Home', 'Predicción Simple', 'Copiloto Conversacional']
}

# Información del proyecto
PROJECT_INFO = {
    'name': 'Sistema de Optimización Energética con IA',
    'description': 'Copiloto inteligente para la industria siderúrgica',
    'dataset': 'Steel Industry Energy Consumption',
    'records': 35040,
    'frequency': '15 minutos',
    'period': '2018',
    'models': ['XGBoost', 'LightGBM', 'CatBoost', 'Random Forest', 
               'Extra Trees', 'Chronos-2 Zero-shot', 'Chronos-2 Fine-tuned', 
               'Chronos-2 with Covariates'],
    'metrics': {
        'rmse_target': 0.205,
        'rmse_benchmark': 0.241,
        'improvement': 0.15
    }
}

# URLs y endpoints
URLS = {
    'github': 'https://github.com/DanteA0179/mlops_proyecto_atreides',
    'documentation': 'https://github.com/DanteA0179/mlops_proyecto_atreides/blob/main/README.md',
    'api_docs': 'http://localhost:8000/docs',
    'mlflow': 'http://localhost:5000',
    'dagster': 'http://localhost:3000',
    'dataset': 'https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption'
}

# Equipo
TEAM = {
    'Juan': 'Data Engineer',
    'Erick': 'Data Scientist',
    'Julian': 'ML Engineer',
    'Dante': 'Software Engineer & Scrum Master',
    'Arthur': 'MLOps/SRE Engineer'
}


def get_version():
    """
    Retorna la versión actual de la aplicación.
    
    Returns:
        str: Versión de la aplicación
        
    Example:
        >>> from src.streamlit_app import get_version
        >>> print(get_version())
        '1.0.0'
    """
    return __version__


def get_app_config():
    """
    Retorna la configuración de la aplicación.
    
    Returns:
        dict: Diccionario con la configuración
        
    Example:
        >>> from src.streamlit_app import get_app_config
        >>> config = get_app_config()
        >>> print(config['title'])
        'Energy Optimization AI - Steel Industry'
    """
    return APP_CONFIG.copy()


def get_project_info():
    """
    Retorna información del proyecto.
    
    Returns:
        dict: Diccionario con información del proyecto
        
    Example:
        >>> from src.streamlit_app import get_project_info
        >>> info = get_project_info()
        >>> print(info['name'])
        'Sistema de Optimización Energética con IA'
    """
    return PROJECT_INFO.copy()


def get_team_info():
    """
    Retorna información del equipo.
    
    Returns:
        dict: Diccionario con miembros del equipo y roles
        
    Example:
        >>> from src.streamlit_app import get_team_info
        >>> team = get_team_info()
        >>> print(team['Dante'])
        'Software Engineer & Scrum Master'
    """
    return TEAM.copy()


def get_urls():
    """
    Retorna URLs importantes del proyecto.
    
    Returns:
        dict: Diccionario con URLs
        
    Example:
        >>> from src.streamlit_app import get_urls
        >>> urls = get_urls()
        >>> print(urls['github'])
        'https://github.com/DanteA0179/mlops_proyecto_atreides'
    """
    return URLS.copy()
