# Configuración de Google Secret Manager

Guía para configurar y usar Google Secret Manager para gestionar credenciales del proyecto de forma segura.

## 🎯 Objetivo

Evitar almacenar credenciales en archivos `.env` locales y permitir que todo el equipo acceda a las mismas configuraciones de forma segura a través de Google Secret Manager.

## 📋 Prerrequisitos

1. **Cuenta de GCP activa** con permisos de Secret Manager
2. **Google Cloud SDK** instalado y configurado
3. **Autenticación configurada**:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```
4. **Proyecto de GCP creado**:
   ```bash
   export GCP_PROJECT_ID="tu-proyecto-id"
   gcloud config set project $GCP_PROJECT_ID
   ```

## 🔧 Instalación

### 1. Habilitar Secret Manager API

```bash
gcloud services enable secretmanager.googleapis.com
```

### 2. Instalar dependencias de Python

```bash
poetry add google-cloud-secret-manager
```

## 🚀 Configuración Inicial

### Paso 1: Configurar variables de entorno locales (temporal)

Crea un archivo `.env.local` **SOLO para configuración inicial** (este archivo NO debe commitearse):

```bash
# .env.local - SOLO PARA CONFIGURACIÓN INICIAL
GCP_PROJECT_ID=mlops-atreides-2025
GCP_REGION=us-central1
GCS_BUCKET_NAME=mlops-atreides-dvc

DVC_REMOTE_URL=gs://mlops-atreides-dvc/dvc-storage

MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=energy-optimization

OPENAI_API_KEY=sk-...
HUGGINGFACE_TOKEN=hf_...
```

### Paso 2: Crear secretos en Secret Manager

```bash
# Cargar variables desde .env.local
source .env.local

# Ejecutar script de setup
python scripts/setup_secrets.py create --project-id $GCP_PROJECT_ID
```

Esto creará los siguientes secretos:
- ✅ `GCP_PROJECT_ID`
- ✅ `GCP_REGION`
- ✅ `GCS_BUCKET_NAME`
- ✅ `DVC_REMOTE_URL`
- ✅ `MLFLOW_TRACKING_URI`
- ✅ `MLFLOW_EXPERIMENT_NAME`
- ✅ `OPENAI_API_KEY` (si aplica)
- ✅ `HUGGINGFACE_TOKEN` (si aplica)

### Paso 3: Eliminar archivo temporal

```bash
# IMPORTANTE: Eliminar .env.local después de crear los secretos
rm .env.local
```

## 📖 Uso del Sistema de Secretos

### Opción 1: Usar el script de Python

#### Listar secretos existentes

```bash
python scripts/setup_secrets.py list --project-id $GCP_PROJECT_ID
```

#### Cargar secretos como variables de entorno

```bash
python scripts/setup_secrets.py load --project-id $GCP_PROJECT_ID
```

#### Configurar DVC con Secret Manager

```bash
python scripts/setup_secrets.py setup-dvc \
  --project-id $GCP_PROJECT_ID \
  --bucket-name mlops-atreides-dvc
```

### Opción 2: Usar el módulo de Python en tu código

```python
from src.utils.secrets import get_secret, get_required_secret, SecretKeys

# Obtener secreto con fallback
project_id = get_secret(SecretKeys.GCP_PROJECT_ID, default="default-project")

# Obtener secreto requerido (lanza excepción si no existe)
bucket_name = get_required_secret(SecretKeys.GCS_BUCKET_NAME)

# Usar directamente
mlflow_uri = get_secret("MLFLOW_TRACKING_URI")
```

### Opción 3: Usar gcloud CLI directamente

```bash
# Crear secreto
echo -n "mi-valor-secreto" | gcloud secrets create MI_SECRETO --data-file=-

# Leer secreto
gcloud secrets versions access latest --secret="MI_SECRETO"

# Listar secretos
gcloud secrets list

# Eliminar secreto
gcloud secrets delete MI_SECRETO
```

## 🔒 Permisos y Control de Acceso

### Otorgar permisos a miembros del equipo

```bash
# Otorgar acceso de lectura a un usuario
gcloud secrets add-iam-policy-binding GCP_PROJECT_ID \
  --member="user:juan@example.com" \
  --role="roles/secretmanager.secretAccessor"

# Otorgar acceso a una service account
gcloud secrets add-iam-policy-binding GCP_PROJECT_ID \
  --member="serviceAccount:dvc-service@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### Roles disponibles

| Rol | Permisos | Uso |
|-----|----------|-----|
| `roles/secretmanager.admin` | Crear, leer, modificar, eliminar secretos | Administrador del proyecto |
| `roles/secretmanager.secretAccessor` | Leer secretos | Desarrolladores del equipo |
| `roles/secretmanager.secretVersionManager` | Crear versiones de secretos | CI/CD pipelines |

## 🔄 Workflow del Equipo

### Para nuevos miembros

1. **Autenticarse en GCP**:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   gcloud config set project mlops-atreides-2025
   ```

2. **Cargar secretos**:
   ```bash
   python scripts/setup_secrets.py load --project-id mlops-atreides-2025
   ```

3. **Verificar configuración**:
   ```bash
   echo $GCP_PROJECT_ID
   echo $GCS_BUCKET_NAME
   ```

### Para CI/CD (GitHub Actions)

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SERVICE_ACCOUNT_KEY }}

      - name: Load secrets from Secret Manager
        run: |
          python scripts/setup_secrets.py load --project-id mlops-atreides-2025

      - name: Run tests
        run: poetry run pytest
```

## 🛡️ Seguridad y Mejores Prácticas

### ✅ Hacer

- ✅ Usar Secret Manager para **todas** las credenciales
- ✅ Rotar secretos periódicamente (cada 90 días recomendado)
- ✅ Usar versiones de secretos para rollback
- ✅ Auditar accesos con Cloud Audit Logs
- ✅ Otorgar permisos mínimos necesarios

### ❌ No Hacer

- ❌ **NUNCA** commitear archivos `.env` con credenciales
- ❌ **NUNCA** compartir credenciales por Slack/email
- ❌ **NUNCA** incluir credenciales en logs
- ❌ **NUNCA** usar credenciales hardcodeadas en el código

## 🔍 Troubleshooting

### Error: Permission Denied

```bash
# Verificar permisos
gcloud projects get-iam-policy $GCP_PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:user:TU_EMAIL"

# Solicitar acceso al admin del proyecto
```

### Error: Secret not found

```bash
# Listar secretos disponibles
gcloud secrets list

# Crear secreto manualmente
echo -n "valor" | gcloud secrets create NOMBRE_SECRETO --data-file=-
```

### Error: API not enabled

```bash
# Habilitar Secret Manager API
gcloud services enable secretmanager.googleapis.com
```

## 📊 Monitoreo y Auditoría

### Ver historial de accesos

```bash
# Logs de acceso a secretos
gcloud logging read "resource.type=secretmanager.googleapis.com/Secret" \
  --limit 50 \
  --format json
```

### Ver versiones de un secreto

```bash
gcloud secrets versions list GCP_PROJECT_ID
```

### Crear nueva versión de un secreto

```bash
echo -n "nuevo-valor" | gcloud secrets versions add GCP_PROJECT_ID --data-file=-
```

## 💰 Costos

Secret Manager tiene un tier gratuito generoso:

| Recurso | Tier Gratuito | Costo Adicional |
|---------|---------------|-----------------|
| Versiones de secretos activas | Primeras 6 gratis | $0.06 por versión/mes |
| Llamadas a la API | Primeras 10,000 gratis | $0.03 per 10,000 llamadas |

**Estimación para este proyecto**: ~$0.00/mes (dentro del tier gratuito)

## 📚 Referencias

- [Secret Manager Documentation](https://cloud.google.com/secret-manager/docs)
- [Secret Manager Pricing](https://cloud.google.com/secret-manager/pricing)
- [Best Practices](https://cloud.google.com/secret-manager/docs/best-practices)
- [Python Client Library](https://cloud.google.com/python/docs/reference/secretmanager/latest)

## 🔗 Integración con DVC

### Configuración Inicial (Una sola vez por máquina)

DVC necesita credenciales de GCP para acceder al bucket. Cada miembro del equipo debe ejecutar:

```bash
# 1. Autenticarse con GCP
gcloud auth application-default login

# 2. Configurar DVC localmente (sin commitear)
# En Windows PowerShell:
dvc remote modify --local gcs-remote credentialpath "$env:APPDATA\gcloud\application_default_credentials.json"

# En Linux/Mac:
dvc remote modify --local gcs-remote credentialpath "$HOME/.config/gcloud/application_default_credentials.json"
```

**Nota importante**: El archivo `.dvc/config.local` contiene rutas específicas de tu máquina y **NO debe commitearse** (ya está en `.gitignore`).

### Uso Normal

```bash
# Verificar configuración
dvc remote list

# Ver estado del cache remoto
dvc status --cloud

# Push de datos
dvc add data/raw/steel_energy_original.csv
dvc push

# Pull de datos
dvc pull
```

### Troubleshooting

Si recibes el error `Anonymous caller does not have storage.objects.list access`:

1. Verifica que estés autenticado:
   ```bash
   gcloud auth application-default print-access-token
   ```

2. Verifica que la configuración local exista:
   ```bash
   cat .dvc/config.local  # Linux/Mac
   type .dvc\config.local  # Windows
   ```

3. Si no existe, ejecuta el paso 2 de la configuración inicial nuevamente.

---

**Contacto**: Para problemas de acceso, contactar al Scrum Master (Dante) o al MLOps Engineer (Arthur).
