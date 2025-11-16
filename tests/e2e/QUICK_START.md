# Quick Start - E2E Tests

## ⚡ Inicio Rápido

### Opción 1: Script Automatizado (Recomendado)

```bash
# Tests de pipeline (no requiere API, más rápido)
./scripts/run_e2e_tests.sh --pipeline

# Tests de API (inicia API automáticamente)
./scripts/run_e2e_tests.sh --api --start-api

# Todo con coverage
./scripts/run_e2e_tests.sh --all --start-api --coverage
```

### Opción 2: Pytest Directo

```bash
# Pipeline tests
poetry run pytest tests/e2e/test_pipeline_e2e.py -v

# API tests (requiere API corriendo en otra terminal)
# Terminal 1:
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# Terminal 2:
poetry run pytest tests/e2e/test_api_e2e.py -v
```

## 📋 Comandos Útiles

```bash
# Ver ayuda del script
./scripts/run_e2e_tests.sh --help

# Test específico
poetry run pytest tests/e2e/test_api_e2e.py::TestSinglePrediction -v

# Con más detalles
poetry run pytest tests/e2e/ -v -s

# Generar coverage
poetry run pytest tests/e2e/ --cov=src --cov-report=html
```

## ✅ Verificación Rápida

```bash
# Verificar que tests cargan correctamente
poetry run pytest tests/e2e/ --collect-only

# Contar tests
poetry run pytest tests/e2e/ --collect-only -q | grep "test session starts" -A 1
```

## 🔧 Troubleshooting Rápido

### API no responde
```bash
# Reiniciar API
pkill -f uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Datos no disponibles
```bash
# Descargar con DVC
dvc pull data/raw/Steel_industry_data.csv.dvc
```

### Dependencias faltantes
```bash
# Reinstalar
poetry install
```

## 📊 Tests Disponibles

- **API E2E**: 34 tests (API endpoints, validación, workflows)
- **Pipeline E2E**: 22 tests (data loading, training, MLflow)
- **Total**: 56 tests end-to-end

## 📖 Más Información

- [README detallado](README.md)
- [Documentación completa](../../docs/testing_e2e.md)
