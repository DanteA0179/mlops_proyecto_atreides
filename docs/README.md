# Documentación del Proyecto

Índice de documentación técnica y guías de configuración.

## 🚀 Guías de Inicio Rápido

### Para Nuevos Miembros del Equipo

1. **[Configuración de DVC](DVC_SETUP.md)** ⭐
   - Setup de credenciales de Google Cloud Storage
   - Scripts automáticos para Windows/Linux/Mac
   - Troubleshooting común
   - **Tiempo estimado:** 5 minutos

2. **[Google Secret Manager Setup](SECRET_MANAGER_SETUP.md)**
   - Gestión segura de credenciales del proyecto
   - Configuración de secretos compartidos
   - Integración con DVC y MLflow
   - **Tiempo estimado:** 10 minutos

## 📋 Orden Recomendado de Configuración

Para nuevos miembros, seguir este orden:

```bash
# 1. Clonar repositorio
git clone <repo-url>
cd energy-optimization-copilot

# 2. Instalar dependencias
poetry install
poetry shell

# 3. Autenticarse en GCP
gcloud auth login
gcloud auth application-default login
gcloud config set project mlops-equipo-37

# 4. Configurar DVC (elegir una opción)
# Opción A: Script automático
.\scripts\setup_dvc_credentials.ps1  # Windows
./scripts/setup_dvc_credentials.sh   # Linux/Mac

# Opción B: Manual
dvc remote modify --local gcs-remote credentialpath "$env:APPDATA\gcloud\application_default_credentials.json"

# 5. Descargar datos
dvc pull

# 6. Verificar instalación
pytest
```

## 📚 Documentación Adicional

### Arquitectura y Diseño
- [ML Canvas](ml_canvas.md) *(pendiente)*
- [Architecture Decision Records](adr/) *(pendiente)*

### Contexto del Proyecto
- [Plan de Proyecto](../context/PlaneacionProyecto.md)

### API
- [Swagger UI](http://localhost:8000/docs) - Documentación interactiva de la API
- [ReDoc](http://localhost:8000/redoc) - Documentación alternativa

## 🔧 Configuración Avanzada

### DVC
- **Archivo de configuración global:** `.dvc/config` (commiteado)
- **Archivo de configuración local:** `.dvc/config.local` (NO commiteado)
- **Remote storage:** Google Cloud Storage (`gs://energy-opt-dvc-remote/dvc-storage`)

### Secret Manager
- **Proyecto GCP:** `mlops-equipo-37`
- **Secretos disponibles:** Ver [SECRET_MANAGER_SETUP.md](SECRET_MANAGER_SETUP.md#paso-2-crear-secretos-en-secret-manager)

## 🐛 Troubleshooting

### Problemas Comunes

#### Error: "Anonymous caller does not have storage.objects.list access"
**Solución:** Ver [DVC_SETUP.md - Troubleshooting](DVC_SETUP.md#error-anonymous-caller-does-not-have-storageobjectslist-access)

#### Error: "Permission denied" en GCS
**Solución:** Contactar al admin del proyecto para obtener permisos de `Storage Object Admin`

#### DVC no encuentra credenciales
**Solución:** Ejecutar `gcloud auth application-default login` y reconfigurar DVC

## 📞 Contacto

Para problemas de acceso o configuración:
- **Scrum Master:** Dante
- **MLOps Engineer:** Arthur
- **Canal de WhatsApp:**

## 🔗 Enlaces Útiles

- [DVC Documentation](https://dvc.org/doc)
- [Google Cloud Storage](https://cloud.google.com/storage/docs)
- [Google Secret Manager](https://cloud.google.com/secret-manager/docs)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Última actualización:** Octubre 2025
