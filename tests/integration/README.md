# Integration Tests

Pruebas de integración que validan la interacción entre diferentes componentes del sistema.

## 📋 Contenido

### `test_data_pipeline_integration.py`
Pruebas de integración para el pipeline de datos.

**Cobertura**:
- ✅ Integración CSV → DuckDB
- ✅ Workflow completo: load → clean → store
- ✅ Quality checks con DuckDB
- ✅ Detección de outliers y almacenamiento
- ✅ Transformaciones encadenadas
- ✅ Persistencia de datos

**Test Classes**:
- `TestDataLoadingIntegration` - Carga de datos
- `TestDataQualityIntegration` - Calidad de datos
- `TestDataTransformationIntegration` - Transformaciones
- `TestDuckDBUtilsIntegration` - Utilidades DuckDB
- `TestDataPersistenceIntegration` - Persistencia
- `TestErrorHandlingIntegration` - Manejo de errores

### `test_feature_model_integration.py`
Pruebas de integración entre feature engineering y modelos.

**Cobertura**:
- ✅ Features temporales con modelos
- ✅ Pipelines completos de sklearn
- ✅ Train/test split con entrenamiento
- ✅ Evaluación de modelos
- ✅ Feature importance
- ✅ Persistencia de modelos
- ✅ Cross-validation

**Test Classes**:
- `TestFeatureEngineeringIntegration` - Feature engineering
- `TestModelPipelineIntegration` - Pipelines sklearn
- `TestTrainTestSplitIntegration` - Splits de datos
- `TestModelEvaluationIntegration` - Evaluación
- `TestFeatureImportanceIntegration` - Importancia
- `TestModelPersistenceIntegration` - Persistencia
- `TestCrossValidationIntegration` - Cross-validation
- `TestEndToEndModelWorkflow` - Workflow completo

### `test_api_service_integration.py`
Pruebas de integración para API y servicios.

**Cobertura**:
- ✅ Validación de request models
- ✅ Feature service transformations
- ✅ Model service predictions
- ✅ Workflow completo: request → prediction
- ✅ Batch predictions
- ✅ Error handling
- ✅ Response formatting
- ✅ Concurrency handling

**Test Classes**:
- `TestAPIModelIntegration` - Modelos de API
- `TestFeatureServiceIntegration` - Servicio de features
- `TestModelServiceIntegration` - Servicio de modelos
- `TestAPIServiceWorkflow` - Workflow completo
- `TestErrorHandlingIntegration` - Manejo de errores
- `TestResponseFormatting` - Formateo de respuestas
- `TestConcurrencyIntegration` - Concurrencia
- `TestValidationIntegration` - Validación

### `test_mlflow_integration.py`
Pruebas de integración con MLflow.

**Cobertura**:
- ✅ Logging de experimentos
- ✅ Model registry
- ✅ Comparación de modelos
- ✅ Artifacts logging
- ✅ Metric tracking
- ✅ Parameter logging
- ✅ Tags y metadata
- ✅ Workflow completo

**Test Classes**:
- `TestMLflowLoggingIntegration` - Logging
- `TestMLflowModelRegistry` - Registry
- `TestMLflowExperimentComparison` - Comparación
- `TestMLflowArtifacts` - Artifacts
- `TestMLflowMetricTracking` - Métricas
- `TestMLflowParameterLogging` - Parámetros
- `TestMLflowTags` - Tags
- `TestMLflowWorkflow` - Workflow completo
- `TestMLflowErrorHandling` - Manejo de errores
- `TestMLflowSearchAndQuery` - Búsqueda y consultas

## 🚀 Ejecución

### Ejecutar Todas las Pruebas de Integración

```bash
# Todas las pruebas
poetry run pytest tests/integration/ -v

# Con salida detallada
poetry run pytest tests/integration/ -v -s

# Con coverage
poetry run pytest tests/integration/ --cov=src --cov-report=html
```

### Ejecutar Pruebas Específicas

```bash
# Solo data pipeline
poetry run pytest tests/integration/test_data_pipeline_integration.py -v

# Solo feature-model
poetry run pytest tests/integration/test_feature_model_integration.py -v

# Solo API services
poetry run pytest tests/integration/test_api_service_integration.py -v

# Solo MLflow
poetry run pytest tests/integration/test_mlflow_integration.py -v

# Una clase específica
poetry run pytest tests/integration/test_data_pipeline_integration.py::TestDataLoadingIntegration -v
```

### Ejecutar con Marcadores

```bash
# Solo tests que requieren DB
poetry run pytest tests/integration/ -m "requires_db" -v

# Solo tests que requieren MLflow
poetry run pytest tests/integration/ -m "requires_mlflow" -v

# Excluir tests lentos
poetry run pytest tests/integration/ -m "not slow" -v
```

## 📊 Estadísticas

| Archivo | Tests | Clases | Descripción |
|---------|-------|--------|-------------|
| `test_data_pipeline_integration.py` | 15 | 6 | Pipeline de datos |
| `test_feature_model_integration.py` | 16 | 8 | Features y modelos |
| `test_api_service_integration.py` | 13 | 7 | API y servicios |
| `test_mlflow_integration.py` | 15 | 10 | MLflow tracking |
| **Total** | **59** | **31** | - |

## 🔧 Configuración

### Fixtures Disponibles

#### Globales (`conftest.py`)
- `setup_integration_environment` - Setup de entorno

#### Data Pipeline
- `sample_data` - Datos de ejemplo
- `temp_csv_file` - Archivo CSV temporal
- `temp_db_path` - Path de base de datos temporal

#### Feature-Model
- `sample_training_data` - Datos de entrenamiento

#### API Service
- `mock_trained_model` - Modelo entrenado mock
- `temp_model_file` - Archivo de modelo temporal

#### MLflow
- `mlflow_tracking_uri` - URI de tracking temporal
- `sample_ml_data` - Datos ML de ejemplo

## ✅ Criterios de Éxito

### Data Pipeline Integration
- ✅ CSV se carga correctamente a DuckDB
- ✅ Transformaciones funcionan en cadena
- ✅ Datos persisten correctamente
- ✅ Quality checks detectan problemas

### Feature-Model Integration
- ✅ Features se integran con modelos
- ✅ Pipelines ejecutan correctamente
- ✅ Modelos entrenan sin errores
- ✅ Persistencia funciona correctamente

### API Service Integration
- ✅ Requests se validan correctamente
- ✅ Features se transforman correctamente
- ✅ Modelos predicen correctamente
- ✅ Responses se formatean correctamente

### MLflow Integration
- ✅ Experimentos se registran correctamente
- ✅ Modelos se registran en registry
- ✅ Métricas se trackean correctamente
- ✅ Artifacts se almacenan correctamente

## 🐛 Troubleshooting

### Tests Fallan por Dependencias

```bash
# Reinstalar dependencias
poetry install

# Verificar imports
poetry run python -c "import src; import mlflow; import duckdb"
```

### Tests de DuckDB Fallan

```bash
# Verificar DuckDB instalado
poetry show duckdb

# Reinstalar si es necesario
poetry add duckdb
```

### Tests de MLflow Fallan

```bash
# Verificar MLflow instalado
poetry show mlflow

# Limpiar experimentos de test
rm -rf mlruns/
```

### Fixtures No Encontrados

```bash
# Verificar conftest.py existe
ls tests/integration/conftest.py

# Ejecutar con verbose
poetry run pytest tests/integration/ -v --fixtures
```

## 📝 Mejores Prácticas

### Escribir Tests de Integración

1. **Scope Apropiado**: Usar fixtures con scope adecuado
2. **Cleanup**: Siempre limpiar recursos temporales
3. **Aislamiento**: Tests no deben depender entre sí
4. **Datos Reales**: Usar datos representativos
5. **Assertions Claras**: Verificar comportamiento esperado

### Template para Nuevo Test

```python
class TestNewIntegration:
    """Test integration between X and Y"""
    
    def test_integration_workflow(self, fixture1, fixture2):
        """Test complete integration workflow"""
        # Setup
        component_a = ComponentA()
        component_b = ComponentB()
        
        # Execute
        result_a = component_a.process(fixture1)
        result_b = component_b.process(result_a, fixture2)
        
        # Verify
        assert result_b is not None
        assert result_b.property == expected_value
```

## 🎯 Cobertura Esperada

- **Data Pipeline**: >80% de funciones de pipeline cubiertas
- **Feature-Model**: >75% de transformers y pipelines
- **API Service**: >70% de servicios API
- **MLflow**: >70% de funciones de tracking

## 📚 Referencias

- [Pytest Documentation](https://docs.pytest.org/)
- [Integration Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
- [DuckDB Python API](https://duckdb.org/docs/api/python/overview)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/index.html)

## 🔄 Actualización

Cuando agregues nuevas funcionalidades:

1. Agregar tests de integración correspondientes
2. Actualizar este README
3. Verificar que fixtures sean reutilizables
4. Documentar nuevos marcadores si aplica
5. Actualizar estadísticas

---

**Última actualización**: 2024-11-15  
**Versión**: 1.0  
**Mantenido por**: MLOps Team - Proyecto Atreides
