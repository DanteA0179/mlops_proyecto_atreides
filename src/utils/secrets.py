"""
Módulo para gestión de secretos y configuración desde Google Secret Manager.

Este módulo proporciona utilidades para:
- Cargar configuración desde Secret Manager
- Fallback a variables de entorno locales
- Cache de secretos para optimizar llamadas a la API
"""

import os
from functools import lru_cache

try:
    from google.api_core import exceptions
    from google.cloud import secretmanager

    SECRETS_AVAILABLE = True
except ImportError:
    SECRETS_AVAILABLE = False


class SecretsManager:
    """Gestiona el acceso a secretos con fallback a variables de entorno."""

    def __init__(self, project_id: str | None = None):
        """
        Inicializa el gestor de secretos.

        Args:
            project_id: ID del proyecto de GCP. Si no se proporciona,
                       se intenta obtener de la variable de entorno GCP_PROJECT_ID
        """
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.client = None

        if SECRETS_AVAILABLE and self.project_id:
            try:
                self.client = secretmanager.SecretManagerServiceClient()
            except Exception as e:
                print(f"⚠️  No se pudo inicializar Secret Manager: {e}")
                print("   Usando variables de entorno locales")

    @lru_cache(maxsize=128)
    def get_secret(
        self, secret_id: str, default: str | None = None, version: str = "latest"
    ) -> str | None:
        """
        Obtiene un secreto de Secret Manager o variable de entorno.

        Args:
            secret_id: ID del secreto
            default: Valor por defecto si el secreto no existe
            version: Versión del secreto (default: "latest")

        Returns:
            Valor del secreto o None si no existe
        """
        # Primero intentar con variable de entorno
        env_value = os.getenv(secret_id)
        if env_value:
            return env_value

        # Si hay cliente de Secret Manager, intentar desde allí
        if self.client and self.project_id:
            try:
                name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
                response = self.client.access_secret_version(request={"name": name})
                return response.payload.data.decode("UTF-8")
            except exceptions.NotFound:
                pass
            except Exception as e:
                print(f"⚠️  Error obteniendo secreto '{secret_id}': {e}")

        return default

    def get_required_secret(self, secret_id: str, version: str = "latest") -> str:
        """
        Obtiene un secreto requerido, lanza excepción si no existe.

        Args:
            secret_id: ID del secreto
            version: Versión del secreto

        Returns:
            Valor del secreto

        Raises:
            ValueError: Si el secreto no existe
        """
        value = self.get_secret(secret_id, version=version)
        if not value:
            raise ValueError(
                f"Secreto requerido '{secret_id}' no encontrado. "
                f"Configurar en Secret Manager o variable de entorno."
            )
        return value

    def load_secrets_to_env(self, secret_ids: list[str]) -> dict[str, str]:
        """
        Carga múltiples secretos como variables de entorno.

        Args:
            secret_ids: Lista de IDs de secretos a cargar

        Returns:
            Diccionario con secretos cargados
        """
        loaded = {}
        for secret_id in secret_ids:
            value = self.get_secret(secret_id)
            if value:
                os.environ[secret_id] = value
                loaded[secret_id] = value
        return loaded


# Instancia global del gestor de secretos
_secrets_manager: SecretsManager | None = None


def get_secrets_manager() -> SecretsManager:
    """
    Obtiene la instancia global del gestor de secretos.

    Returns:
        Instancia de SecretsManager
    """
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


def get_secret(secret_id: str, default: str | None = None) -> str | None:
    """
    Función helper para obtener un secreto.

    Args:
        secret_id: ID del secreto
        default: Valor por defecto

    Returns:
        Valor del secreto
    """
    return get_secrets_manager().get_secret(secret_id, default)


def get_required_secret(secret_id: str) -> str:
    """
    Función helper para obtener un secreto requerido.

    Args:
        secret_id: ID del secreto

    Returns:
        Valor del secreto

    Raises:
        ValueError: Si el secreto no existe
    """
    return get_secrets_manager().get_required_secret(secret_id)


# Constantes para IDs de secretos comunes
class SecretKeys:
    """IDs de secretos estándar del proyecto."""

    # GCP Configuration
    GCP_PROJECT_ID = "GCP_PROJECT_ID"
    GCP_REGION = "GCP_REGION"
    GCS_BUCKET_NAME = "GCS_BUCKET_NAME"

    # DVC Configuration
    DVC_REMOTE_URL = "DVC_REMOTE_URL"

    # MLflow Configuration
    MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"
    MLFLOW_EXPERIMENT_NAME = "MLFLOW_EXPERIMENT_NAME"

    # API Keys
    OPENAI_API_KEY = "OPENAI_API_KEY"
    HUGGINGFACE_TOKEN = "HUGGINGFACE_TOKEN"

    # Database
    DATABASE_URL = "DATABASE_URL"


def load_project_secrets() -> dict[str, str]:
    """
    Carga todos los secretos del proyecto.

    Returns:
        Diccionario con secretos cargados
    """
    manager = get_secrets_manager()

    secret_ids = [
        SecretKeys.GCP_PROJECT_ID,
        SecretKeys.GCP_REGION,
        SecretKeys.GCS_BUCKET_NAME,
        SecretKeys.DVC_REMOTE_URL,
        SecretKeys.MLFLOW_TRACKING_URI,
        SecretKeys.MLFLOW_EXPERIMENT_NAME,
        SecretKeys.OPENAI_API_KEY,
        SecretKeys.HUGGINGFACE_TOKEN,
        SecretKeys.DATABASE_URL,
    ]

    return manager.load_secrets_to_env(secret_ids)
