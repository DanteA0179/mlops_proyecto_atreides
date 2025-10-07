# Configuración de Service Account para DVC

Guía para crear y configurar una Service Account de GCP para usar con DVC en CI/CD y desarrollo local.

## 🎯 ¿Cuándo usar Service Account vs Cuenta Personal?

### Cuenta Personal (Application Default Credentials)
- ✅ **Desarrollo local** - Rápido y fácil
- ✅ **Uso individual** - Cada dev usa su propia cuenta
- ✅ **Sin archivos de credenciales** - Más seguro
- ❌ No funciona en CI/CD

**Usar para:** Desarrollo diario del equipo

### Service Account
- ✅ **CI/CD** - GitHub Actions, GitLab CI, etc.
- ✅ **Compartir acceso** - Mismo archivo para todo el equipo
- ✅ **Automatización** - Scripts y pipelines
- ⚠️ Requiere manejo seguro del archivo JSON

**Usar para:** GitHub Actions y automatización

## 🚀 Crear Service Account

### Opción 1: Script Automático (Recomendado)

**Windows:**
```powershell
.\scripts\setup_service_account.ps1
```

**Linux/Mac:**
```bash
chmod +x scripts/setup_service_account.sh
./scripts/setup_service_account.sh
```

El script te pedirá:
1. **Project ID**: `mlops-equipo-37` (o tu proyecto)
2. **Service Account Name**: `dvc-storage` (recomendado)
3. **Bucket Name**: `energy-opt-dvc-remote` (o tu bucket)

### Opción 2: Manual

```bash
# 1. Configurar proyecto
export PROJECT_ID="mlops-equipo-37"
export SERVICE_ACCOUNT_NAME="dvc-storage"
export BUCKET_NAME="energy-opt-dvc-remote"

gcloud config set project $PROJECT_ID

# 2. Crear service account
gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
  --display-name="DVC Storage Access" \
  --description="Service account for DVC to access GCS bucket"

# 3. Otorgar permisos en el bucket
SERVICE_ACCOUNT_EMAIL="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"

gsutil iam ch "serviceAccount:${SERVICE_ACCOUNT_EMAIL}:roles/storage.objectAdmin" \
  "gs://$BUCKET_NAME"

# 4. Descargar credenciales
mkdir -p config
gcloud iam service-accounts keys create config/gcs-service-account.json \
  --iam-account=$SERVICE_ACCOUNT_EMAIL

# 5. Configurar DVC localmente
dvc remote modify --local gcs-remote credentialpath config/gcs-service-account.json

# 6. Verificar
dvc status --cloud
```

## 🔒 Seguridad

### ⚠️ IMPORTANTE

El archivo `config/gcs-service-account.json` contiene credenciales sensibles:

- ❌ **NUNCA** lo commitees a Git
- ❌ **NUNCA** lo compartas por email/Slack
- ❌ **NUNCA** lo subas a servicios públicos
- ✅ Está en `.gitignore` automáticamente
- ✅ Compártelo solo por canales seguros

### Compartir con el Equipo

**Opción 1: Cada miembro usa su cuenta personal** (Recomendado)
```bash
# Cada dev ejecuta:
./scripts/setup_dvc_credentials.sh
```

**Opción 2: Compartir service account**
- Usa Google Drive (carpeta privada del equipo)
- Usa 1Password o similar
- Usa Google Secret Manager (ver `SECRET_MANAGER_SETUP.md`)

### Para CI/CD (GitHub Actions)

1. **Copia el contenido del archivo:**
   ```bash
   # Windows
   Get-Content config/gcs-service-account.json | Set-Clipboard
   
   # Linux/Mac
   cat config/gcs-service-account.json | pbcopy  # Mac
   cat config/gcs-service-account.json | xclip   # Linux
   ```

2. **Agrégalo como GitHub Secret:**
   - Ve a tu repo → Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `GCP_SERVICE_ACCOUNT_KEY`
   - Value: Pega el contenido del JSON
   - Click "Add secret"

3. **Úsalo en tu workflow:**
   ```yaml
   - name: Setup DVC credentials
     run: |
       mkdir -p config
       echo '${{ secrets.GCP_SERVICE_ACCOUNT_KEY }}' > config/gcs-service-account.json
       dvc remote modify --local gcs-remote credentialpath config/gcs-service-account.json
   ```

## 🧹 Eliminar Service Account (Cleanup)

Si necesitas eliminar la service account:

```bash
# Listar keys
gcloud iam service-accounts keys list \
  --iam-account=dvc-storage@mlops-equipo-37.iam.gserviceaccount.com

# Eliminar key específica
gcloud iam service-accounts keys delete KEY_ID \
  --iam-account=dvc-storage@mlops-equipo-37.iam.gserviceaccount.com

# Eliminar service account completa
gcloud iam service-accounts delete \
  dvc-storage@mlops-equipo-37.iam.gserviceaccount.com
```

## 🔄 Rotar Credenciales

Se recomienda rotar las credenciales cada 90 días:

```bash
# 1. Crear nueva key
gcloud iam service-accounts keys create config/gcs-service-account-new.json \
  --iam-account=dvc-storage@mlops-equipo-37.iam.gserviceaccount.com

# 2. Probar nueva key
dvc remote modify --local gcs-remote credentialpath config/gcs-service-account-new.json
dvc status --cloud

# 3. Si funciona, eliminar key antigua
gcloud iam service-accounts keys list \
  --iam-account=dvc-storage@mlops-equipo-37.iam.gserviceaccount.com

gcloud iam service-accounts keys delete OLD_KEY_ID \
  --iam-account=dvc-storage@mlops-equipo-37.iam.gserviceaccount.com

# 4. Renombrar archivo
mv config/gcs-service-account-new.json config/gcs-service-account.json

# 5. Actualizar GitHub Secret con el nuevo contenido
```

## 🐛 Troubleshooting

### Error: "Service account already exists"

No es un error, el script continúa con la cuenta existente.

### Error: "Permission denied"

Tu cuenta de GCP no tiene permisos para crear service accounts. Necesitas el rol:
- `roles/iam.serviceAccountAdmin`
- `roles/iam.serviceAccountKeyAdmin`

Contacta al admin del proyecto.

### El archivo de credenciales no funciona

```bash
# Verificar que el archivo existe
ls -la config/gcs-service-account.json

# Verificar que es JSON válido
cat config/gcs-service-account.json | python -m json.tool

# Verificar configuración de DVC
cat .dvc/config.local

# Probar manualmente
export GOOGLE_APPLICATION_CREDENTIALS=config/gcs-service-account.json
dvc status --cloud
```

## 📊 Comparación: Personal vs Service Account

| Característica | Cuenta Personal | Service Account |
|----------------|-----------------|-----------------|
| Setup | 1 comando | Script o manual |
| Seguridad | ✅ Más seguro | ⚠️ Requiere cuidado |
| CI/CD | ❌ No funciona | ✅ Funciona |
| Compartir | ✅ Cada uno la suya | ⚠️ Compartir archivo |
| Rotación | Automática | Manual cada 90 días |
| Auditoría | Por usuario | Por service account |
| Costo | Gratis | Gratis |

## 📚 Referencias

- [GCP Service Accounts](https://cloud.google.com/iam/docs/service-accounts)
- [DVC with GCS](https://dvc.org/doc/user-guide/data-management/remote-storage/google-cloud-storage)
- [GitHub Actions Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)

---

**Siguiente paso:** Configura GitHub Actions - Ver `docs/GITHUB_ACTIONS_SETUP.md`
