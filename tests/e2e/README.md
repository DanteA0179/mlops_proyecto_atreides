# End-to-End (E2E) Tests

Pruebas end-to-end que validan el funcionamiento completo del sistema, desde la carga de datos hasta las predicciones de la API.

## 📋 Contenido

### `test_api_e2e.py`
Pruebas E2E completas para la API de predicción de energía.

**Cobertura**:
- ✅ Ciclo de vida de la API (startup, health checks)
- ✅ Endpoints de predicción individual y batch
- ✅ Validación de entrada y manejo de errores
- ✅ Workflows completos end-to-end
- ✅ Pruebas de rendimiento

**Test Suites**:
- `TestAPILifecycle` - Inicialización y documentación
- `TestHealthEndpoints` - Health checks
- `TestModelEndpoints` - Información del modelo
- `TestSinglePrediction` - Predicciones individuales
- `TestSinglePredictionValidation` - Validación de entrada
- `TestBatchPrediction` - Predicciones en batch
- `TestBatchPredictionValidation` - Validación de batch
- `TestEndToEndWorkflow` - Workflows completos
- `TestErrorHandling` - Manejo de errores
- `TestPerformance` - Características de rendimiento

### `test_pipeline_e2e.py`
Pruebas E2E para el pipeline completo de ML (datos → modelo → evaluación).

**Cobertura**:
- ✅ Carga y procesamiento de datos
- ✅ Limpieza y calidad de datos
- ✅ Feature engineering completo
- ✅ Entrenamiento de modelos
- ✅ Integración con MLflow
- ✅ Pipeline completo reproducible
- ✅ Versionado de datos con DVC

**Test Suites**:
- `TestDataLoadingPipeline` - Carga de datos
- `TestDataCleaningPipeline` - Limpieza
- `TestFeatureEngineeringPipeline` - Feature engineering
- `TestTrainingPipeline` - Entrenamiento
- `TestMLflowIntegration` - Tracking con MLflow
- `TestCompletePipeline` - Pipeline completo
- `TestDataVersioning` - Versionado con DVC
- `TestPipelineErrorHandling` - Manejo de errores

## 🚀 Ejecución

### Requisitos Previos

#### Para `test_api_e2e.py`:
```bash
# Iniciar la API en otra terminal
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# O con auto-reload para desarrollo
uvicorn src.api.main:app --reload
```

#### Para `test_pipeline_e2e.py`:
```bash
# Asegurar que los datos estén disponibles
dvc pull data/raw/Steel_industry_data.csv.dvc

# O tener el archivo en data/raw/
```

### Ejecutar Todas las Pruebas E2E

```bash
# Todas las pruebas E2E
poetry run pytest tests/e2e/ -v

# Con salida detallada
poetry run pytest tests/e2e/ -v -s

# Con coverage
poetry run pytest tests/e2e/ --cov=src --cov-report=html
```

### Ejecutar Pruebas Específicas

```bash
# Solo pruebas de API
poetry run pytest tests/e2e/test_api_e2e.py -v

# Solo pruebas de pipeline
poetry run pytest tests/e2e/test_pipeline_e2e.py -v

# Solo una clase de tests
poetry run pytest tests/e2e/test_api_e2e.py::TestSinglePrediction -v

# Solo un test específico
poetry run pytest tests/e2e/test_api_e2e.py::TestSinglePrediction::test_predict_valid_request -v
```

### Ejecutar con Marcadores

```bash
# Solo tests rápidos (si se configuran)
poetry run pytest tests/e2e/ -m "not slow" -v

# Solo tests de performance
poetry run pytest tests/e2e/test_api_e2e.py::TestPerformance -v
```

## 📊 Coverage Esperado

Las pruebas E2E deben cubrir:

- ✅ **API Endpoints**: 100% de endpoints probados
- ✅ **Data Pipeline**: Flujo completo de datos
- ✅ **Feature Engineering**: Todos los transformers
- ✅ **Model Training**: Múltiples algoritmos
- ✅ **Error Handling**: Casos edge y errores

## 🔧 Configuración

### Variables de Entorno

Para `test_api_e2e.py`:
```bash
export API_BASE_URL=http://localhost:8000
```

### Fixtures Globales

Las fixtures están definidas en cada archivo de test:
- `api_base_url` - URL base de la API
- `api_health_check` - Verificación de API disponible
- `valid_prediction_request` - Request válido de ejemplo
- `sample_data_path` - Ruta a datos de ejemplo

## 📝 Escribir Nuevas Pruebas E2E

### Template para Prueba de API

```python
class TestNewAPIFeature:
    """Test suite for new API feature"""
    
    def test_new_endpoint(self, api_base_url: str, api_health_check):
        """Test new endpoint functionality"""
        response = requests.post(
            f"{api_base_url}/new_endpoint",
            json={"param": "value"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "expected_field" in data
```

### Template para Prueba de Pipeline

```python
class TestNewPipelineStage:
    """Test new pipeline stage"""
    
    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """Create sample data for testing"""
        return pl.DataFrame({"col": [1, 2, 3]})
    
    def test_new_transformation(self, sample_data: pl.DataFrame):
        """Test new data transformation"""
        from src.features.new_transformer import NewTransformer
        
        transformer = NewTransformer()
        df_transformed = transformer.fit_transform(sample_data)
        
        assert len(df_transformed) == len(sample_data)
        assert "new_feature" in df_transformed.columns
```

## 🐛 Troubleshooting

### API Tests Failing

**Problema**: `API is not running`
```bash
# Solución: Iniciar la API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**Problema**: `Model not loaded`
```bash
# Solución: Verificar que el modelo existe
ls models/

# O entrenar un modelo
python src/models/train_model.py
```

**Problema**: `Connection timeout`
```bash
# Solución: Aumentar timeout en fixture
# Editar api_health_check fixture y aumentar retry_delay
```

### Pipeline Tests Failing

**Problema**: `Sample data not available`
```bash
# Solución: Descargar datos
dvc pull

# O copiar manualmente
cp /path/to/Steel_industry_data.csv data/raw/
```

**Problema**: `Module not found`
```bash
# Solución: Instalar dependencias
poetry install

# Verificar PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

**Problema**: `MLflow not configured`
```bash
# Solución: Iniciar MLflow server
mlflow ui

# O configurar tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000
```

## 📈 CI/CD Integration

### GitHub Actions

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
      
      - name: Start API
        run: |
          poetry run uvicorn src.api.main:app &
          sleep 10
      
      - name: Run E2E tests
        run: poetry run pytest tests/e2e/ -v
```

## 🎯 Best Practices

1. **Independencia**: Cada test debe ser independiente
2. **Cleanup**: Usar fixtures para cleanup automático
3. **Timeouts**: Establecer timeouts razonables
4. **Datos**: Usar datos de prueba, no producción
5. **Assertions**: Assertions claras y específicas
6. **Documentación**: Docstrings descriptivos
7. **Performance**: Tests E2E pueden ser lentos, marcar los más lentos

## 📚 Referencias

- [Pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Requests Library](https://requests.readthedocs.io/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
