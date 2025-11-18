#!/usr/bin/env python3
"""
Script para configurar y gestionar secretos en Google Secret Manager.

Este script permite:
1. Crear secretos en Google Secret Manager
2. Acceder a secretos desde variables de entorno
3. Configurar credenciales para DVC, MLflow, y otros servicios

Uso:
    # Crear secretos
    python scripts/setup_secrets.py create --project-id PROJECT_ID

    # Cargar secretos como variables de entorno
    python scripts/setup_secrets.py load --project-id PROJECT_ID

    # Listar secretos existentes
    python scripts/setup_secrets.py list --project-id PROJECT_ID
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from google.api_core import exceptions
    from google.cloud import secretmanager
except ImportError:
    print("Error: google-cloud-secret-manager no está instalado")
    print("Instalar con: poetry add google-cloud-secret-manager")
    sys.exit(1)


class SecretManagerHandler:
    """Gestiona secretos en Google Secret Manager."""

    def __init__(self, project_id: str):
        """
        Inicializa el cliente de Secret Manager.

        Args:
            project_id: ID del proyecto de GCP
        """
        self.project_id = project_id
        self.client = secretmanager.SecretManagerServiceClient()
        self.parent = f"projects/{project_id}"

    def create_secret(self, secret_id: str, secret_value: str) -> str:
        """
        Crea un secreto en Secret Manager.

        Args:
            secret_id: Identificador único del secreto
            secret_value: Valor del secreto

        Returns:
            Nombre completo del secreto creado
        """
        secret_path = f"{self.parent}/secrets/{secret_id}"

        try:
            # Intentar crear el secreto
            secret = self.client.create_secret(
                request={
                    "parent": self.parent,
                    "secret_id": secret_id,
                    "secret": {
                        "replication": {"automatic": {}},
                    },
                }
            )
            print(f"Secreto creado: {secret.name}")
        except exceptions.AlreadyExists:
            print(f"Secreto ya existe: {secret_path}")
            secret_path = secret_path

        # Agregar versión del secreto
        version = self.client.add_secret_version(
            request={
                "parent": secret_path,
                "payload": {"data": secret_value.encode("UTF-8")},
            }
        )
        print(f"Versión agregada: {version.name}")
        return secret_path

    def get_secret(self, secret_id: str, version: str = "latest") -> str:
        """
        Obtiene el valor de un secreto.

        Args:
            secret_id: Identificador del secreto
            version: Versión del secreto (default: "latest")

        Returns:
            Valor del secreto
        """
        name = f"{self.parent}/secrets/{secret_id}/versions/{version}"

        try:
            response = self.client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except exceptions.NotFound:
            print(f"Secreto no encontrado: {secret_id}")
            return ""

    def list_secrets(self) -> list:
        """
        Lista todos los secretos del proyecto.

        Returns:
            Lista de nombres de secretos
        """
        secrets = []
        for secret in self.client.list_secrets(request={"parent": self.parent}):
            secret_name = secret.name.split("/")[-1]
            secrets.append(secret_name)
            print(f"  - {secret_name}")
        return secrets

    def delete_secret(self, secret_id: str) -> None:
        """
        Elimina un secreto.

        Args:
            secret_id: Identificador del secreto
        """
        name = f"{self.parent}/secrets/{secret_id}"
        try:
            self.client.delete_secret(request={"name": name})
            print(f"Secreto eliminado: {secret_id}")
        except exceptions.NotFound:
            print(f"Secreto no encontrado: {secret_id}")


def get_default_secrets() -> dict[str, str]:
    """
    Define los secretos por defecto para el proyecto.

    Returns:
        Diccionario con secretos y sus valores de ejemplo
    """
    return {
        # GCP Configuration
        "GCP_PROJECT_ID": os.getenv("GCP_PROJECT_ID", ""),
        "GCP_REGION": os.getenv("GCP_REGION", "us-central1"),
        "GCS_BUCKET_NAME": os.getenv("GCS_BUCKET_NAME", ""),
        # DVC Configuration
        "DVC_REMOTE_URL": os.getenv("DVC_REMOTE_URL", ""),
        # MLflow Configuration
        "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        "MLFLOW_EXPERIMENT_NAME": os.getenv("MLFLOW_EXPERIMENT_NAME", "energy-optimization"),
        # API Keys (si son necesarias)
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "HUGGINGFACE_TOKEN": os.getenv("HUGGINGFACE_TOKEN", ""),
        # Database (si aplica)
        "DATABASE_URL": os.getenv("DATABASE_URL", ""),
    }


def create_secrets_command(args):
    """Comando para crear secretos en Secret Manager."""
    handler = SecretManagerHandler(args.project_id)

    secrets = get_default_secrets()

    # Filtrar secretos vacíos
    secrets = {k: v for k, v in secrets.items() if v}

    if not secrets:
        print("ADVERTENCIA: No hay secretos definidos en variables de entorno")
        print("Definir variables en .env o como variables de entorno del sistema")
        return

    print(f"\nCreando {len(secrets)} secretos en Secret Manager...")
    for secret_id, secret_value in secrets.items():
        handler.create_secret(secret_id, secret_value)

    print(f"\nSecretos creados exitosamente en proyecto: {args.project_id}")


def load_secrets_command(args):
    """Comando para cargar secretos como variables de entorno."""
    handler = SecretManagerHandler(args.project_id)

    secrets = get_default_secrets()

    print("\nCargando secretos desde Secret Manager...")

    env_vars = {}
    for secret_id in secrets.keys():
        value = handler.get_secret(secret_id)
        if value:
            env_vars[secret_id] = value
            # Exportar como variable de entorno
            os.environ[secret_id] = value
            print(f"  {secret_id}")

    # Opcionalmente guardar en archivo temporal (NO USAR EN PRODUCCIÓN)
    if args.export_file:
        env_file = Path(args.export_file)
        with env_file.open("w") as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        print(f"\nVariables exportadas a: {env_file}")
        print("ADVERTENCIA: Este archivo contiene secretos. NO commitear a git.")


def list_secrets_command(args):
    """Comando para listar secretos."""
    handler = SecretManagerHandler(args.project_id)

    print(f"\nSecretos en proyecto '{args.project_id}':")
    secrets = handler.list_secrets()

    if not secrets:
        print("  (ninguno)")

    print(f"\nTotal: {len(secrets)} secretos")


def delete_secret_command(args):
    """Comando para eliminar un secreto."""
    handler = SecretManagerHandler(args.project_id)

    if not args.secret_id:
        print("Error: --secret-id es requerido")
        return

    confirm = input(f"¿Eliminar secreto '{args.secret_id}'? (yes/no): ")
    if confirm.lower() == "yes":
        handler.delete_secret(args.secret_id)
    else:
        print("Operación cancelada")


def setup_dvc_with_secrets(project_id: str, bucket_name: str):
    """
    Configura DVC con GCS usando credenciales de Secret Manager.

    Args:
        project_id: ID del proyecto de GCP
        bucket_name: Nombre del bucket de GCS
    """
    SecretManagerHandler(project_id)

    # Cargar credenciales
    print("\nConfigurando DVC con Google Cloud Storage...")

    # Configurar remote de DVC
    remote_url = f"gs://{bucket_name}/dvc-storage"

    os.system(f"dvc remote add -d --force gcs-remote {remote_url}")
    os.system(f"dvc remote modify gcs-remote projectname {project_id}")

    print(f"DVC configurado con bucket: {bucket_name}")
    print(f"   Remote URL: {remote_url}")


def main():
    """Función principal."""
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env.local")

    parser = argparse.ArgumentParser(description="Gestión de secretos en Google Secret Manager")

    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")

    # Comando: create
    create_parser = subparsers.add_parser("create", help="Crear secretos")
    create_parser.add_argument("--project-id", required=True, help="ID del proyecto de GCP")

    # Comando: load
    load_parser = subparsers.add_parser("load", help="Cargar secretos")
    load_parser.add_argument("--project-id", required=True, help="ID del proyecto de GCP")
    load_parser.add_argument(
        "--export-file", help="Archivo para exportar variables (opcional, NO RECOMENDADO)"
    )

    # Comando: list
    list_parser = subparsers.add_parser("list", help="Listar secretos")
    list_parser.add_argument("--project-id", required=True, help="ID del proyecto de GCP")

    # Comando: delete
    delete_parser = subparsers.add_parser("delete", help="Eliminar secreto")
    delete_parser.add_argument("--project-id", required=True, help="ID del proyecto de GCP")
    delete_parser.add_argument("--secret-id", required=True, help="ID del secreto a eliminar")

    # Comando: setup-dvc
    dvc_parser = subparsers.add_parser("setup-dvc", help="Configurar DVC con GCS")
    dvc_parser.add_argument("--project-id", required=True, help="ID del proyecto de GCP")
    dvc_parser.add_argument("--bucket-name", required=True, help="Nombre del bucket de GCS")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Ejecutar comando
    commands = {
        "create": create_secrets_command,
        "load": load_secrets_command,
        "list": list_secrets_command,
        "delete": delete_secret_command,
    }

    if args.command == "setup-dvc":
        setup_dvc_with_secrets(args.project_id, args.bucket_name)
    else:
        commands[args.command](args)


if __name__ == "__main__":
    main()
