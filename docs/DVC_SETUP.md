# Configuración de DVC con Google Cloud Storage

Guía rápida para configurar DVC en tu máquina local.

## �  Dos Métodos de Autenticación

### Método 1: Cuenta Personal (Recomendado para desarrollo)
- ✅ Rápido y simple
- ✅ Usa tu cuenta de Google
- ✅ Ideal para desarrollo local
- ❌ No funciona para CI/CD

### Método 2: Service Account (Recomendado para producción/CI/CD)
- ✅ Funciona en CI/CD
- ✅ Permisos granulares
- ✅ Compartible con el equipo
- ⚠️ Requiere manejo seguro de credenciales

---

## 🚀 Setup Rápido - Método 1: Cuenta Personal (5 minutos)

### Opción 1: Script Automático (Recomendado)

**Windows:**
```powershell
.\scripts\setup_dvc_credentials.ps1
```

**Linux/Mac:**
```bash
chmod +x scripts/setup_dvc_credentials.sh
./scripts/setup_dvc_credentials.sh
```

### Opción 2: Manual

1. **Autenticarse con GCP:**
   ```bash
   gcloud auth application-default login
   ```

2. **Configurar DVC localmente:**
   
   **Windows PowerShell:**
   ```powershell
   dvc remote modify --local gcs-remote credentialpath "$env:APPDATA\gcloud\application_default_credentials.json"
   ```
   
   **Linux/Mac:**
   ```bash
   dvc remote modify --local gcs-remote credentialpath "$HOME/.config/gcloud/application_default_credentials.json"
   ```

3. **Verificar:**
   ```bash
   dvc status --cloud
   ```

## 📝 ¿Por qué `--local`?

El flag `--local` guarda la configuración en `.dvc/config.local`, que:
- ✅ **NO se commitea** a Git (está en `.gitignore`)
- ✅ Es específico de tu máquina
- ✅ Evita exponer rutas personales en el repositorio
- ✅ Permite que cada miembro del equipo tenga su propia configuración

## 🔒 Seguridad

- ❌ **NUNCA** commitees `.dvc/config.local`
- ❌ **NUNCA** agregues rutas absolutas a `.dvc/config`
- ✅ Usa `--local` para configuración específica de tu máquina
- ✅ Las credenciales se manejan automáticamente por `gcloud`

## 🛠️ Comandos Útiles

```bash
# Ver configuración global (commiteada)
cat .dvc/config

# Ver configuración local (NO commiteada)
cat .dvc/config.local

# Ver todos los remotes
dvc remote list -v

# Ver estado del cache remoto
dvc status --cloud

# Subir datos
dvc push

# Descargar datos
dvc pull

# Descargar solo un archivo específico
dvc pull data/raw/steel_energy_original.csv.dvc
```

## 🐛 Troubleshooting

### Error: "Anonymous caller does not have storage.objects.list access"

**Causa:** DVC no encuentra tus credenciales de GCP.

**Solución:**
```bash
# 1. Re-autenticarse
gcloud auth application-default login

# 2. Verificar que el archivo de credenciales existe
# Windows:
dir "$env:APPDATA\gcloud\application_default_credentials.json"

# Linux/Mac:
ls ~/.config/gcloud/application_default_credentials.json

# 3. Reconfigurar DVC
# Ejecuta el script de setup nuevamente
```

### Error: "Permission denied"

**Causa:** No tienes permisos en el proyecto GCP.

**Solución:** Contacta al admin del proyecto (Dante o Arthur) para que te otorgue el rol `Storage Object Admin` en el bucket `energy-opt-dvc-remote`.

### El archivo `.dvc/config.local` no existe

**Causa:** No has ejecutado la configuración local.

**Solución:** Ejecuta el paso 2 del setup manual o el script automático.

## 👥 Para Nuevos Miembros del Equipo

1. Clona el repositorio
2. Instala dependencias: `poetry install`
3. Ejecuta el script de setup: `.\scripts\setup_dvc_credentials.ps1` (Windows) o `./scripts/setup_dvc_credentials.sh` (Linux/Mac)
4. ¡Listo! Ya puedes hacer `dvc pull` para descargar los datos

---

## 🔐 Setup Alternativo - Método 2: Service Account

### ¿Cuándo usar Service Account?

Usa este método si:
- Necesitas configurar CI/CD (GitHub Actions, etc.)
- Quieres compartir credenciales con el equipo de forma segura
- Necesitas permisos específicos separados de usuarios

### Setup con Service Account

**Opción A: Script Automático**

```powershell
# Windows
.\scripts\setup_service_account.ps1

# Linux/Mac
chmod +x scripts/setup_service_account.sh
./scripts/setup_service_account.sh
```

**Opción B: Manual**

```bash
# 1. Crear Service Account
gcloud iam service-accounts create dvc-storage \
  --display-name="DVC Storage Access" \
  --project=mlops-equipo-37

# 2. Otorgar permisos
gcloud projects add-iam-policy-binding mlops-equipo-37 \
  --member="serviceAccount:dvc-storage@mlops-equipo-37.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# 3. Crear key file
mkdir -p config
gcloud iam service-accounts keys create config/gcs-service-account.json \
  --iam-account=dvc-storage@mlops-equipo-37.iam.gserviceaccount.com

# 4. Configurar DVC
dvc remote modify --local gcs-remote credentialpath config/gcs-service-account.json

# 5. Verificar
dvc status --cloud
```

### ⚠️ Seguridad con Service Account

- ❌ **NUNCA** commitees el archivo `.json` de la service account
- ✅ Compártelo usando Google Secret Manager
- ✅ Rota las keys periódicamente (cada 90 días)
- ✅ Usa permisos mínimos necesarios

### Compartir con el Equipo

Para compartir la service account key de forma segura:

```bash
# Subir a Secret Manager
gcloud secrets create dvc-service-account-key \
  --data-file=config/gcs-service-account.json \
  --project=mlops-equipo-37

# Otros miembros pueden descargarla
gcloud secrets versions access latest \
  --secret=dvc-service-account-key \
  --project=mlops-equipo-37 > config/gcs-service-account.json

# Configurar DVC
dvc remote modify --local gcs-remote credentialpath config/gcs-service-account.json
```

---

## 📊 Comparación de Métodos

| Característica | Cuenta Personal | Service Account |
|----------------|-----------------|-----------------|
| Setup | 2 minutos | 5 minutos |
| Seguridad | Alta (tu cuenta) | Alta (key file) |
| CI/CD | ❌ No | ✅ Sí |
| Compartir | Cada uno su cuenta | Una key compartida |
| Permisos | Tus permisos GCP | Permisos específicos |
| Rotación | Automática | Manual (90 días) |

## 📚 Referencias

- [DVC Documentation](https://dvc.org/doc)
- [DVC with GCS](https://dvc.org/doc/user-guide/data-management/remote-storage/google-cloud-storage)
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- [Service Account Best Practices](https://cloud.google.com/iam/docs/best-practices-service-accounts)
