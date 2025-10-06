# Configuración de DVC con Google Cloud Storage

Guía rápida para configurar DVC en tu máquina local.

## 🚀 Setup Rápido (5 minutos)

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

## 📚 Referencias

- [DVC Documentation](https://dvc.org/doc)
- [DVC with GCS](https://dvc.org/doc/user-guide/data-management/remote-storage/google-cloud-storage)
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
