# Documentación de Pruebas End-to-End (E2E)

## 📊 Resumen

El proyecto ahora incluye **56 pruebas end-to-end** que validan el funcionamiento completo del sistema desde múltiples perspectivas.

### Cobertura de Tests

| Tipo | Archivo | Tests | Descripción |
|------|---------|-------|-------------|
| **API E2E** | `test_api_e2e.py` | 34 | Pruebas completas de la API REST |
| **Pipeline E2E** | `test_pipeline_e2e.py` | 22 | Pruebas del pipeline de ML completo |
| **Total** | - | **56** | - |

## 🎯 Tests de API (`test_api_e2e.py`)

### Categorías de Tests

#### 1. API Lifecycle (3 tests)
- ✅ Verificación de servidor corriendo
- ✅ Endpoint raíz con información
- ✅ Documentación OpenAPI disponible

#### 2. Health Endpoints (3 tests)
- ✅ Health check básico
- ✅ Estado de modelo cargado
- ✅ Health check detallado

#### 3. Model Endpoints (2 tests)
- ✅ Información del modelo
- ✅ Métricas del modelo

#### 4. Single Prediction (4 tests)
- ✅ Predicción válida
- ✅ Predicción con carga ligera
- ✅ Predicción con carga máxima
- ✅ Intervalos de confianza

#### 5. Validation (6 tests)
- ✅ Factor de potencia inválido
- ✅ Valores negativos
- ✅ Tipo de carga inválido
- ✅ Día de semana inválido
- ✅ NSM inválido
- ✅ Campos faltantes

#### 6. Batch Prediction (4 tests)
- ✅ Batch válido
- ✅ Batch con un item
- ✅ Batch grande (50 items)
- ✅ Estadísticas de resumen

#### 7. Batch Validation (2 tests)
- ✅ Lista vacía
- ✅ Item inválido en batch

#### 8. End-to-End Workflows (4 tests)
- ✅ Workflow completo
- ✅ Múltiples predicciones secuenciales
- ✅ Diferentes tipos de carga
- ✅ Fin de semana vs día laboral

#### 9. Error Handling (4 tests)
- ✅ Endpoint inválido (404)
- ✅ Método HTTP incorrecto (405)
- ✅ JSON malformado
- ✅ Content-Type inválido

#### 10. Performance (2 tests)
- ✅ Tiempo de respuesta < 2s
- ✅ Eficiencia de batch vs individual

## 🔄 Tests de Pipeline (`test_pipeline_e2e.py`)

### Categorías de Tests

#### 1. Data Loading (3 tests)
- ✅ Carga de datos raw desde CSV
- ✅ Carga a DuckDB
- ✅ Verificación de calidad de datos

#### 2. Data Cleaning (4 tests)
- ✅ Manejo de valores faltantes
- ✅ Detección de outliers
- ✅ Conversión de tipos de datos
- ✅ Eliminación de duplicados

#### 3. Feature Engineering (3 tests)
- ✅ Creación de features temporales
- ✅ Codificación cíclica
- ✅ Pipeline de features integrado

#### 4. Training Pipeline (3 tests)
- ✅ Entrenamiento baseline (Linear Regression)
- ✅ Entrenamiento XGBoost
- ✅ Evaluación de modelos

#### 5. MLflow Integration (2 tests)
- ✅ MLflow disponible y configurado
- ✅ Logging de modelo a MLflow

#### 6. Complete Pipeline (2 tests)
- ✅ Workflow completo (datos → modelo)
- ✅ Reproducibilidad del pipeline

#### 7. Data Versioning (2 tests)
- ✅ DVC inicializado
- ✅ Datos rastreados con DVC

#### 8. Error Handling (3 tests)
- ✅ Path de datos inválido
- ✅ Columnas faltantes
- ✅ Input inválido al modelo

## 🚀 Uso

### Ejecución Rápida

```bash
# Solo tests de pipeline (no requiere API)
./scripts/run_e2e_tests.sh --pipeline

# Tests de API (requiere API corriendo)
./scripts/run_e2e_tests.sh --api

# Iniciar API y ejecutar tests automáticamente
./scripts/run_e2e_tests.sh --api --start-api

# Todos los tests con coverage
./scripts/run_e2e_tests.sh --all --coverage
```

### Ejecución Manual con Pytest

```bash
# Todos los tests E2E
poetry run pytest tests/e2e/ -v

# Solo API
poetry run pytest tests/e2e/test_api_e2e.py -v

# Solo pipeline
poetry run pytest tests/e2e/test_pipeline_e2e.py -v

# Test específico
poetry run pytest tests/e2e/test_api_e2e.py::TestSinglePrediction::test_predict_valid_request -v

# Con coverage
poetry run pytest tests/e2e/ --cov=src --cov-report=html
```

## 📋 Fixtures Disponibles

### Fixtures Globales (`conftest.py`)
- `setup_test_environment` - Configura entorno de pruebas

### Fixtures de API (`test_api_e2e.py`)
- `api_base_url` - URL base de la API
- `api_health_check` - Verifica API disponible
- `valid_prediction_request` - Request de predicción válido
- `valid_batch_request` - Request de batch válido

### Fixtures de Pipeline (`test_pipeline_e2e.py`)
- `temp_pipeline_dir` - Directorio temporal para tests
- `sample_data_path` - Ruta a datos de ejemplo
- `train_test_split` - Split de datos para entrenamiento

## 🎨 Estructura de Archivos

```
tests/e2e/
├── __init__.py
├── conftest.py              # Configuración pytest
├── README.md                # Guía de tests E2E
├── test_api_e2e.py         # Tests de API (34 tests)
└── test_pipeline_e2e.py    # Tests de pipeline (22 tests)
```

## ✅ Criterios de Éxito

### Para `test_api_e2e.py`
- ✅ API responde en todos los endpoints
- ✅ Validación de entrada funciona correctamente
- ✅ Errores se manejan apropiadamente
- ✅ Performance dentro de límites (<2s por predicción)
- ✅ Batch es más eficiente que individual

### Para `test_pipeline_e2e.py`
- ✅ Datos se cargan y procesan correctamente
- ✅ Feature engineering genera features esperados
- ✅ Modelos entrenan sin errores
- ✅ MLflow registra experimentos
- ✅ Pipeline es reproducible
- ✅ DVC rastrea archivos grandes

## 🐛 Troubleshooting

### API Tests

**Problema**: Tests fallan con "API is not running"

```bash
# Solución 1: Iniciar API manualmente
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Solución 2: Usar script con auto-start
./scripts/run_e2e_tests.sh --api --start-api
```

**Problema**: Tests fallan con "Model not loaded"

```bash
# Verificar modelo existe
ls models/

# Entrenar modelo si no existe
python src/models/train_model.py
```

### Pipeline Tests

**Problema**: Tests se saltan con "Sample data not available"

```bash
# Solución: Descargar datos con DVC
dvc pull

# O copiar manualmente
cp /path/to/Steel_industry_data.csv data/raw/
```

**Problema**: "Module not found"

```bash
# Solución: Instalar dependencias
poetry install

# Verificar instalación
poetry run python -c "import src"
```

## 📈 Integración CI/CD

### GitHub Actions Example

```yaml
name: E2E Tests

on: [push, pull_request]

jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install
      
      - name: Run pipeline E2E tests
        run: poetry run pytest tests/e2e/test_pipeline_e2e.py -v
      
      - name: Start API
        run: |
          poetry run uvicorn src.api.main:app &
          sleep 10
      
      - name: Run API E2E tests
        run: poetry run pytest tests/e2e/test_api_e2e.py -v
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## 📊 Métricas Actuales

- **Total de tests E2E**: 56
- **Tests de API**: 34
- **Tests de Pipeline**: 22
- **Cobertura esperada**: >70%
- **Tiempo de ejecución**: ~10-15 segundos (pipeline), variable (API)

## 🔮 Próximos Pasos

### Tests Adicionales Recomendados
- [ ] Tests de carga (stress testing)
- [ ] Tests de concurrencia (múltiples requests simultáneos)
- [ ] Tests de seguridad (SQL injection, XSS)
- [ ] Tests de integración con Dagster
- [ ] Tests de recuperación ante fallos

### Mejoras Sugeridas
- [ ] Agregar tests de performance benchmarking
- [ ] Implementar tests de regresión visual
- [ ] Agregar tests de compatibilidad de versiones
- [ ] Implementar tests de migración de datos

## 📚 Referencias

- [Documentación completa](tests/e2e/README.md)
- [Guía de testing](../README.md#testing)
- [Pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)

## 🤝 Contribuir

Para agregar nuevos tests E2E:

1. Seguir estructura existente
2. Usar fixtures apropiados
3. Documentar con docstrings
4. Incluir assertions claras
5. Manejar cleanup apropiadamente
6. Actualizar esta documentación

---

**Última actualización**: 2024-11-15  
**Versión**: 1.0  
**Mantenido por**: MLOps Team - Proyecto Atreides
