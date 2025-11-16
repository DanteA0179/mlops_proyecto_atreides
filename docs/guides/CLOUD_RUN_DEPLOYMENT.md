# Guía de Deployment en Google Cloud Run

**Energy Optimization API - Cloud Deployment Guide**

---

## 📋 Tabla de Contenidos

1. [Prerequisitos](#prerequisitos)
2. [Setup Inicial](#setup-inicial)
3. [Deployment](#deployment)
4. [Configuración FinOps](#configuración-finops)
5. [Validación y Testing](#validación-y-testing)
6. [Monitoreo y Logs](#monitoreo-y-logs)
7. [Troubleshooting](#troubleshooting)
8. [Rollback](#rollback)
9. [Costos Estimados](#costos-estimados)

---

## Prerequisitos

### Software Requerido

1. **Google Cloud SDK (gcloud CLI)**
   - Descargar: https://cloud.google.com/sdk/docs/install
   - Versión mínima: 400.0.0
   - Verificar instalación:
     ```powershell
     gcloud version
     ```

2. **PowerShell 7+**
   - Windows 11 viene con PowerShell 5.1 por defecto
   - Recomendado: PowerShell 7+ (https://github.com/PowerShell/PowerShell)
   - Verificar versión:
     ```powershell
     $PSVersionTable.PSVersion
     ```

3. **Acceso a GCP Project**
   - Project ID: `mlops-equipo-37` (o tu project ID)
   - Permisos requeridos:
     - Cloud Run Admin
     - Artifact Registry Admin
     - Cloud Build Editor
     - Service Account User

### Archivos Requeridos

Antes de deployar, asegúrate de tener:

```
✅ cloudbuild.yaml         # Pipeline declarativo de CI/CD
✅ Dockerfile.api          # Dockerfile de US-022 (adaptado para Cloud Run)
✅ .gcloudignore          # Optimización de contexto de build
✅ src/api/main.py
✅ models/ensembles/ensemble_lightgbm_v1.pkl
✅ .env (con GCP_PROJECT_ID configurado)
```

---

## Setup Inicial

### 1. Autenticación con GCP

```powershell
# Login con tu cuenta de Google
gcloud auth login

# Configurar application default credentials
gcloud auth application-default login
```

### 2. Configurar Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto (copia `.env.example`):

```bash
# GCP Configuration
GCP_PROJECT_ID=mlops-equipo-37
GCP_REGION=us-central1
GCP_ARTIFACT_REGISTRY=us-central1-docker.pkg.dev
GCP_REPOSITORY=energy-api-repo

# Cloud Run Configuration
CLOUD_RUN_SERVICE_NAME=energy-optimization-api
CLOUD_RUN_MIN_INSTANCES=0
CLOUD_RUN_MAX_INSTANCES=2
CLOUD_RUN_CONCURRENCY=80
CLOUD_RUN_CPU=1
CLOUD_RUN_MEMORY=2Gi
CLOUD_RUN_TIMEOUT=300

# Model Configuration
MODEL_PATH=/app/models/ensembles/ensemble_lightgbm_v1.pkl
MODEL_TYPE=stacking_ensemble
```

**⚠️ IMPORTANTE**: El archivo `.env` NO debe commitearse a Git.

### 3. Ejecutar Setup de Infraestructura

Este script se ejecuta **una sola vez** para configurar GCP:

```powershell
.\scripts\setup-gcp-infrastructure.ps1
```

Este script:
- ✅ Habilita las APIs necesarias (Cloud Run, Artifact Registry, Cloud Build)
- ✅ Crea el Artifact Registry repository
- ✅ Configura la autenticación de Docker
- ✅ Valida permisos y cuotas

**Output esperado**:
```
==========================================
Infrastructure Setup Complete!
==========================================

Next steps:
1. Ensure your model file exists: models/ensembles/ensemble_lightgbm_v1.pkl
2. Run deployment script: .\scripts\deploy-to-cloudrun.ps1
3. Monitor deployment in Cloud Console:
   https://console.cloud.google.com/run?project=mlops-equipo-37
```

---

## Deployment

### Deployment Completo (Recomendado)

Este script ejecuta todo el proceso de deployment automáticamente:

```powershell
# Deployment con valores por defecto
.\scripts\deploy-to-cloudrun.ps1

# Deployment con tag específico
.\scripts\deploy-to-cloudrun.ps1 -Tag v1.2.0

# Deployment sin cache (rebuild completo)
.\scripts\deploy-to-cloudrun.ps1 -NoCache

# Dry run (simular sin deployar)
.\scripts\deploy-to-cloudrun.ps1 -DryRun
```

### Proceso de Deployment

El script `deploy-to-cloudrun.ps1` orquesta un pipeline declarativo definido en `cloudbuild.yaml`.

#### Fase 1: Validaciones
- Verifica gcloud CLI
- Valida autenticación
- Revisa archivos requeridos

#### Fase 2: Ejecución de Cloud Build Pipeline
El script invoca a Cloud Build, que ejecuta los pasos definidos en `cloudbuild.yaml`:
```powershell
# El script ejecuta internamente:
gcloud builds submit `
  --config cloudbuild.yaml `
  --substitutions _TAG=latest `
  --project mlops-equipo-37
```
El pipeline de `cloudbuild.yaml` realiza las siguientes acciones:
1.  **Build**: Construye la imagen Docker usando `Dockerfile.api` (de US-022).
2.  **Push**: Sube la imagen a Artifact Registry.
3.  **Deploy**: Despliega la nueva imagen en Cloud Run con la configuración FinOps especificada.

**Tiempo estimado**: 5-7 minutos

#### Fase 3: Validación Automática
- El script de deployment espera a que el pipeline finalice.
- Realiza un Health check (`GET /health`).
- Ejecuta un Test de predicción (`POST /predict`).
- Verifica la latencia del servicio.

**Output esperado**:
```
==========================================
Deployment Complete!
==========================================

Service Details:
  Name: energy-optimization-api
  Region: us-central1
  URL: https://energy-optimization-api-HASH-uc.a.run.app
  Image: us-central1-docker.pkg.dev/mlops-equipo-37/energy-api-repo/energy-optimization-api:latest

Configuration:
  Min Instances: 0 (scale-to-zero)
  Max Instances: 2
  Concurrency: 80
  CPU: 1 vCPU
  Memory: 2 GiB
  Timeout: 300s
```

---

## Configuración FinOps

La configuración está optimizada para minimizar costos en desarrollo:

### Scale-to-Zero

```yaml
Min Instances: 0
```

**Beneficios**:
- ✅ **Ahorro del 90%** en costos de idle time
- ✅ Costo $0 cuando no hay tráfico
- ✅ Ideal para desarrollo y demos

**Trade-offs**:
- ⚠️ Cold start de 2-5 segundos después de 15 minutos sin tráfico
- ⚠️ Primera request puede ser lenta

### Max Instances

```yaml
Max Instances: 2
```

**Beneficios**:
- ✅ Previene costos descontrolados
- ✅ Suficiente para tráfico de desarrollo (~100 req/min)
- ✅ Protección contra uso accidental

### Concurrency

```yaml
Concurrency: 80
```

- Cada instancia puede manejar 80 requests simultáneas
- Total máximo: 160 requests simultáneas (2 instancias × 80)

### Recursos

```yaml
CPU: 1 vCPU
Memory: 2 GiB
Timeout: 300s (5 minutos)
```

**Justificación**:
- 1 vCPU suficiente para inference con LightGBM
- 2 GiB necesario para cargar modelo + procesar requests
- 5 minutos timeout para requests complejos

---

## Validación y Testing

### 1. Health Check

```powershell
# Obtener URL del servicio
$SERVICE_URL = gcloud run services describe energy-optimization-api `
  --region us-central1 `
  --project mlops-equipo-37 `
  --format "value(status.url)"

# Test health endpoint
curl $SERVICE_URL/health
```

**Respuesta esperada**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "stacking_ensemble",
  "timestamp": "2025-11-15T10:30:00Z"
}
```

### 2. Test de Predicción

```powershell
# Test con datos de ejemplo
curl -X POST "$SERVICE_URL/predict" `
  -H "Content-Type: application/json" `
  -d '{
    "lagging_current_reactive_power_kvarh": 10.5,
    "leading_current_reactive_power_kvarh": 5.2,
    "co2_tco2": 0.03,
    "lagging_current_power_factor": 0.85,
    "leading_current_power_factor": 0.90,
    "nsm": 36000,
    "week_status": "Weekday",
    "day_of_week": "Monday",
    "load_type": "Medium_Load"
  }'
```

**Respuesta esperada**:
```json
{
  "predicted_usage_kwh": 45.23,
  "model_version": "v1",
  "confidence": 0.95,
  "timestamp": "2025-11-15T10:30:00Z"
}
```

### 3. Test de Latencia (Automated)

```powershell
# Ejecutar test de latencia con 100 requests
.\scripts\test-cloudrun-latency.ps1

# Test con más requests
.\scripts\test-cloudrun-latency.ps1 -RequestCount 500
```

**Métricas a validar**:
- ✅ P95 latency < 1 segundo
- ✅ Success rate > 99%
- ✅ Avg latency < 500ms

---

## Monitoreo y Logs

### Ver Logs en Tiempo Real

```powershell
# Ver últimos 100 logs
.\scripts\view-cloudrun-logs.ps1

# Seguir logs en tiempo real
.\scripts\view-cloudrun-logs.ps1 -Follow

# Filtrar solo errores
.\scripts\view-cloudrun-logs.ps1 -Filter "severity=ERROR"

# Filtrar logs de predicción
.\scripts\view-cloudrun-logs.ps1 -Filter "textPayload:predict"
```

### Ver Métricas

```powershell
# Métricas de la última hora
.\scripts\check-cloudrun-metrics.ps1

# Métricas de las últimas 24 horas
.\scripts\check-cloudrun-metrics.ps1 -Hours 24
```

**Métricas disponibles**:
- Request count
- Request latency (P50, P75, P90, P95, P99)
- Container instance count
- CPU utilization
- Memory utilization

### Cloud Console

**Logs**:
```
https://console.cloud.google.com/run/detail/us-central1/energy-optimization-api/logs?project=mlops-equipo-37
```

**Metrics**:
```
https://console.cloud.google.com/run/detail/us-central1/energy-optimization-api/metrics?project=mlops-equipo-37
```

---

## Troubleshooting

### Problema: Build Falla en Cloud Build

**Síntomas**:
```
ERROR: build step failed
```

**Soluciones**:

1. **Verificar Dockerfile.api**:
   ```powershell
   # Validar sintaxis localmente
   docker build -f Dockerfile.api -t test:latest .
   ```

2. **Verificar .gcloudignore**:
   - Asegurar que archivos necesarios NO estén ignorados
   - `models/ensembles/ensemble_lightgbm_v1.pkl` debe incluirse

3. **Revisar logs de Cloud Build**:
   ```
   https://console.cloud.google.com/cloud-build/builds?project=mlops-equipo-37
   ```

### Problema: Deployment Falla

**Síntomas**:
```
ERROR: gcloud run deploy failed
```

**Soluciones**:

1. **Verificar permisos**:
   ```powershell
   gcloud projects get-iam-policy mlops-equipo-37 `
     --flatten="bindings[].members" `
     --filter="bindings.members:$(gcloud config get-value account)"
   ```

2. **Verificar cuotas**:
   ```
   https://console.cloud.google.com/iam-admin/quotas?project=mlops-equipo-37
   ```

3. **Revisar imagen en Artifact Registry**:
   ```powershell
   gcloud artifacts docker images list us-central1-docker.pkg.dev/mlops-equipo-37/energy-api-repo
   ```

### Problema: Health Check Falla

**Síntomas**:
```
Health check failed: Connection timeout
```

**Soluciones**:

1. **Verificar que el servicio esté running**:
   ```powershell
   gcloud run services describe energy-optimization-api `
     --region us-central1 `
     --project mlops-equipo-37
   ```

2. **Revisar logs de la aplicación**:
   ```powershell
   .\scripts\view-cloudrun-logs.ps1 -Limit 50
   ```

3. **Verificar modelo está cargado**:
   - Revisar que `MODEL_PATH` esté correctamente configurado
   - Verificar que el archivo del modelo existe en la imagen

### Problema: Alta Latencia (P95 > 1s)

**Síntomas**:
```
P95 latency: 1.5 seconds (exceeds 1 second requirement)
```

**Soluciones**:

1. **Aumentar recursos**:
   ```powershell
   # Editar deployment con más CPU/memoria
   gcloud run services update energy-optimization-api `
     --cpu 2 `
     --memory 4Gi `
     --region us-central1
   ```

2. **Reducir cold starts**:
   ```powershell
   # Cambiar min instances a 1
   gcloud run services update energy-optimization-api `
     --min-instances 1 `
     --region us-central1
   ```
   
   **⚠️ Nota**: Esto incrementará costos (~$10-15/mes)

3. **Optimizar modelo**:
   - Usar modelo más ligero
   - Implementar caching de predicciones

### Problema: Costos Inesperados

**Síntomas**:
```
GCP billing alert: $20 USD spent this week
```

**Soluciones**:

1. **Verificar configuración scale-to-zero**:
   ```powershell
   gcloud run services describe energy-optimization-api `
     --region us-central1 `
     --format "value(spec.template.spec.containers[0].resources.limits)"
   ```

2. **Revisar instancias activas**:
   ```powershell
   .\scripts\check-cloudrun-metrics.ps1 -Hours 24
   ```

3. **Verificar tráfico inesperado**:
   - Revisar logs para requests sospechosos
   - Considerar agregar autenticación si hay abuso

---

## Rollback

Si un deployment introduce problemas, puedes hacer rollback rápidamente:

### Rollback Automático (a versión anterior)

```powershell
# Rollback a la versión inmediatamente anterior
.\scripts\rollback-cloudrun.ps1
```

### Rollback a Revisión Específica

```powershell
# Listar todas las revisiones
.\scripts\rollback-cloudrun.ps1 -ListRevisions

# Rollback a revisión específica
.\scripts\rollback-cloudrun.ps1 -RevisionName energy-optimization-api-00005-abc
```

### Rollback Manual

```powershell
# Ver revisiones disponibles
gcloud run revisions list `
  --service energy-optimization-api `
  --region us-central1 `
  --project mlops-equipo-37

# Cambiar tráfico a revisión anterior
gcloud run services update-traffic energy-optimization-api `
  --to-revisions REVISION_NAME=100 `
  --region us-central1 `
  --project mlops-equipo-37
```

---

## Costos Estimados

### Configuración Actual (Development)

Con la configuración FinOps optimizada:

| Recurso | Configuración | Costo Mensual Est. |
|---------|---------------|-------------------|
| Cloud Run (CPU) | 1 vCPU, scale-to-zero | $1-3 USD |
| Cloud Run (Memory) | 2 GiB, scale-to-zero | $0.50-1.50 USD |
| Cloud Run (Requests) | ~10,000/mes | $0.40 USD |
| Artifact Registry | 1 imagen (~500MB) | $0.10 USD |
| Cloud Build | ~20 builds/mes | $1.00 USD |
| Cloud Logging | Logs básicos | $0.50 USD |
| **TOTAL** | | **$3-7 USD/mes** |

### Configuración Producción (min-instances=1)

Si cambias a `min-instances=1` para eliminar cold starts:

| Recurso | Configuración | Costo Mensual Est. |
|---------|---------------|-------------------|
| Cloud Run (CPU) | 1 vCPU, always-on | $12-15 USD |
| Cloud Run (Memory) | 2 GiB, always-on | $8-10 USD |
| Cloud Run (Requests) | ~100,000/mes | $4 USD |
| **TOTAL** | | **$24-30 USD/mes** |

### Tips para Reducir Costos

1. **Usar scale-to-zero en desarrollo**
   - Solo paga cuando hay tráfico
   - Acepta 2-5s de cold start

2. **Limpiar revisiones antiguas**
   ```powershell
   # Listar revisiones
   gcloud run revisions list --service energy-optimization-api --region us-central1
   
   # Eliminar revisión específica
   gcloud run revisions delete REVISION_NAME --region us-central1
   ```

3. **Limpiar imágenes antiguas en Artifact Registry**
   ```powershell
   gcloud artifacts docker images list us-central1-docker.pkg.dev/mlops-equipo-37/energy-api-repo
   
   # Eliminar imagen específica
   gcloud artifacts docker images delete IMAGE_PATH --delete-tags
   ```

4. **Monitorear costos diariamente**
   ```
   https://console.cloud.google.com/billing?project=mlops-equipo-37
   ```

---

## Comandos Rápidos de Referencia

### Deployment
```powershell
# Setup inicial (una vez)
.\scripts\setup-gcp-infrastructure.ps1

# Deploy completo
.\scripts\deploy-to-cloudrun.ps1

# Deploy sin cache
.\scripts\deploy-to-cloudrun.ps1 -NoCache
```

### Monitoreo
```powershell
# Ver logs
.\scripts\view-cloudrun-logs.ps1

# Ver métricas
.\scripts\check-cloudrun-metrics.ps1

# Test de latencia
.\scripts\test-cloudrun-latency.ps1
```

### Rollback
```powershell
# Rollback automático
.\scripts\rollback-cloudrun.ps1

# Listar revisiones
.\scripts\rollback-cloudrun.ps1 -ListRevisions
```

### Información del Servicio
```powershell
# Describir servicio
gcloud run services describe energy-optimization-api --region us-central1

# Obtener URL
gcloud run services describe energy-optimization-api --region us-central1 --format "value(status.url)"

# Listar revisiones
gcloud run revisions list --service energy-optimization-api --region us-central1
```

---

## Próximos Pasos

### Sprint 3 (Opcional)

- [ ] GitHub Actions para CI/CD automático
- [ ] Terraform para Infrastructure as Code
- [ ] Secret Manager para variables sensibles
- [ ] Cloud Armor para protección DDoS
- [ ] Multi-region deployment para alta disponibilidad

### Producción

- [ ] Autenticación con Identity Platform
- [ ] Rate limiting con Cloud Endpoints
- [ ] Monitoring avanzado con Prometheus/Grafana
- [ ] Cloud Load Balancer para balanceo
- [ ] Cloud CDN para caching

---

## Recursos Adicionales

### Documentación GCP

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Artifact Registry Documentation](https://cloud.google.com/artifact-registry/docs)
- [Cloud Build Documentation](https://cloud.google.com/build/docs)
- [Cloud Run Pricing](https://cloud.google.com/run/pricing)

### Documentación Interna

- [US-025 Planning](../us-planning/us-025.md)
- [US-020 FastAPI Endpoints](../us-resolved/us-020.md)
- [AGENTS.md](../../AGENTS.md)
- [API README](../../src/api/README.md)

### Soporte

- **GCP Support**: https://cloud.google.com/support
- **Project Issues**: https://github.com/DanteA0179/mlops_proyecto_atreides/issues
- **Team Contact**: Ver README.md

---

**Versión**: 1.0  
**Última actualización**: 15 de Noviembre, 2025  
**Mantenido por**: Arthur (MLOps/SRE Engineer)
