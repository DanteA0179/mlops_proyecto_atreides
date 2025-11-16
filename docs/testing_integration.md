# Documentación de Pruebas de Integración

## 📊 Resumen

El proyecto ahora incluye **59 pruebas de integración** organizadas en 4 archivos que validan la interacción entre componentes del sistema.

### Cobertura de Tests

| Tipo | Archivo | Tests | Descripción |
|------|---------|-------|-------------|
| **Data Pipeline** | `test_data_pipeline_integration.py` | 15 | Integración de pipeline de datos |
| **Feature-Model** | `test_feature_model_integration.py` | 16 | Features y entrenamiento |
| **API Services** | `test_api_service_integration.py` | 13 | API y servicios |
| **MLflow** | `test_mlflow_integration.py` | 15 | Tracking y registry |
| **Total** | - | **59** | - |

## 🎯 Tests de Data Pipeline

### `test_data_pipeline_integration.py` (15 tests)

Pruebas de integración para el pipeline de datos, desde la carga hasta el almacenamiento.

#### Clases de Tests

**TestDataLoadingIntegration** (2 tests)
- ✅ `test_csv_to_duckdb_integration` - Carga CSV a DuckDB
- ✅ `test_load_clean_store_workflow` - Workflow completo

**TestDataQualityIntegration** (2 tests)
- ✅ `test_quality_checks_with_duckdb` - Quality checks
- ✅ `test_outlier_detection_with_storage` - Detección de outliers

**TestDataTransformationIntegration** (2 tests)
- ✅ `test_load_transform_validate_workflow` - Transformaciones
- ✅ `test_multiple_transformations_chain` - Cadena de transformaciones

**TestDuckDBUtilsIntegration** (2 tests)
- ✅ `test_utils_with_real_database` - Utils con DB real
- ✅ `test_query_functions_integration` - Funciones de query

**TestDataPersistenceIntegration** (2 tests)
- ✅ `test_save_load_consistency` - Consistencia de datos
- ✅ `test_multiple_table_operations` - Múltiples tablas

**TestErrorHandlingIntegration** (2 tests)
- ✅ `test_invalid_data_handling` - Datos inválidos
- ✅ `test_database_error_handling` - Errores de DB

## 🔄 Tests de Feature-Model

### `test_feature_model_integration.py` (16 tests)

Pruebas de integración entre feature engineering y modelos ML.

#### Clases de Tests

**TestFeatureEngineeringIntegration** (2 tests)
- ✅ `test_temporal_features_with_model` - Features temporales
- ✅ `test_feature_pipeline_integration` - Pipeline de features

**TestModelPipelineIntegration** (2 tests)
- ✅ `test_complete_sklearn_pipeline` - Pipeline sklearn completo
- ✅ `test_feature_model_pipeline_integration` - Features + modelo

**TestTrainTestSplitIntegration** (2 tests)
- ✅ `test_split_with_training` - Split con entrenamiento
- ✅ `test_feature_engineering_with_split` - Features con split

**TestModelEvaluationIntegration** (2 tests)
- ✅ `test_train_evaluate_workflow` - Train y evaluación
- ✅ `test_multiple_models_comparison` - Comparación de modelos

**TestFeatureImportanceIntegration** (1 test)
- ✅ `test_feature_importance_extraction` - Extracción de importancia

**TestModelPersistenceIntegration** (2 tests)
- ✅ `test_model_save_load_predictions` - Guardar/cargar modelo
- ✅ `test_pipeline_persistence` - Persistencia de pipeline

**TestCrossValidationIntegration** (1 test)
- ✅ `test_cross_validation_with_features` - Cross-validation

**TestEndToEndModelWorkflow** (1 test)
- ✅ `test_complete_workflow` - Workflow completo

## 🌐 Tests de API Services

### `test_api_service_integration.py` (13 tests)

Pruebas de integración para API y capa de servicios.

#### Clases de Tests

**TestAPIModelIntegration** (3 tests)
- ✅ `test_prediction_request_validation` - Validación de requests
- ✅ `test_invalid_request_raises_error` - Errores de validación
- ✅ `test_batch_request_validation` - Validación de batch

**TestFeatureServiceIntegration** (2 tests)
- ✅ `test_feature_transformation_integration` - Transformación de features
- ✅ `test_batch_feature_transformation` - Batch transformation

**TestModelServiceIntegration** (2 tests)
- ✅ `test_model_loading` - Carga de modelo
- ✅ `test_model_prediction` - Predicción

**TestAPIServiceWorkflow** (2 tests)
- ✅ `test_request_to_prediction_workflow` - Workflow completo
- ✅ `test_batch_workflow` - Workflow batch

**TestErrorHandlingIntegration** (2 tests)
- ✅ `test_invalid_feature_handling` - Features inválidos
- ✅ `test_model_not_found_handling` - Modelo no encontrado

**TestResponseFormatting** (1 test)
- ✅ `test_prediction_response_format` - Formato de respuesta

**TestConcurrencyIntegration** (1 test)
- ✅ `test_multiple_simultaneous_predictions` - Predicciones simultáneas

## 📈 Tests de MLflow

### `test_mlflow_integration.py` (15 tests)

Pruebas de integración con MLflow para tracking y registry.

#### Clases de Tests

**TestMLflowLoggingIntegration** (2 tests)
- ✅ `test_basic_experiment_logging` - Logging básico
- ✅ `test_model_training_with_logging` - Training con logging

**TestMLflowModelRegistry** (1 test)
- ✅ `test_register_model` - Registro de modelo

**TestMLflowExperimentComparison** (1 test)
- ✅ `test_compare_multiple_models` - Comparación de modelos

**TestMLflowArtifacts** (2 tests)
- ✅ `test_log_artifacts` - Logging de artifacts
- ✅ `test_log_multiple_artifacts` - Múltiples artifacts

**TestMLflowMetricTracking** (2 tests)
- ✅ `test_log_metrics_over_epochs` - Métricas por epoch
- ✅ `test_multiple_metric_logging` - Múltiples métricas

**TestMLflowParameterLogging** (2 tests)
- ✅ `test_log_model_parameters` - Parámetros de modelo
- ✅ `test_nested_parameters` - Parámetros anidados

**TestMLflowTags** (1 test)
- ✅ `test_set_tags` - Tags de experimentos

**TestMLflowWorkflow** (1 test)
- ✅ `test_complete_training_workflow` - Workflow completo

**TestMLflowErrorHandling** (2 tests)
- ✅ `test_handle_invalid_metric` - Métricas inválidas
- ✅ `test_handle_duplicate_run_name` - Nombres duplicados

**TestMLflowSearchAndQuery** (1 test)
- ✅ `test_search_runs` - Búsqueda de runs

## 🚀 Uso

### Ejecución Rápida

```bash
# Todas las pruebas de integración
poetry run pytest tests/integration/ -v

# Con coverage
poetry run pytest tests/integration/ --cov=src --cov-report=html

# Tests específicos
poetry run pytest tests/integration/test_data_pipeline_integration.py -v
poetry run pytest tests/integration/test_feature_model_integration.py -v
poetry run pytest tests/integration/test_api_service_integration.py -v
poetry run pytest tests/integration/test_mlflow_integration.py -v
```

### Ejecución por Marcadores

```bash
# Solo tests que requieren DB
poetry run pytest tests/integration/ -m "requires_db" -v

# Solo tests que requieren MLflow
poetry run pytest tests/integration/ -m "requires_mlflow" -v

# Excluir tests lentos
poetry run pytest tests/integration/ -m "not slow" -v
```

### Ejecución por Clase

```bash
# Una clase específica
poetry run pytest tests/integration/test_data_pipeline_integration.py::TestDataLoadingIntegration -v

# Un test específico
poetry run pytest tests/integration/test_mlflow_integration.py::TestMLflowLoggingIntegration::test_basic_experiment_logging -v
```

## 📋 Fixtures Disponibles

### Globales (`conftest.py`)
- `setup_integration_environment` - Configura entorno de testing

### Data Pipeline
- `sample_data` - DataFrame de ejemplo
- `temp_csv_file` - Archivo CSV temporal
- `temp_db_path` - Path de base de datos temporal

### Feature-Model
- `sample_training_data` - Datos de entrenamiento (X, y)

### API Services
- `mock_trained_model` - Modelo entrenado mock
- `temp_model_file` - Path a modelo guardado

### MLflow
- `mlflow_tracking_uri` - URI de tracking temporal
- `sample_ml_data` - Datos ML (X_train, X_test, y_train, y_test)

## 🎨 Estructura

```
tests/integration/
├── __init__.py
├── conftest.py                          # Configuración
├── README.md                            # Guía detallada
├── test_data_pipeline_integration.py   # Data pipeline (15 tests)
├── test_feature_model_integration.py   # Features + modelos (16 tests)
├── test_api_service_integration.py     # API services (13 tests)
└── test_mlflow_integration.py          # MLflow (15 tests)
```

## ✅ Criterios de Éxito

### Data Pipeline
- ✅ CSV se carga correctamente a DuckDB
- ✅ Transformaciones funcionan en cadena
- ✅ Datos persisten con consistencia
- ✅ Quality checks detectan problemas
- ✅ Errores se manejan correctamente

### Feature-Model
- ✅ Features se integran con modelos
- ✅ Pipelines sklearn ejecutan correctamente
- ✅ Splits de datos funcionan con training
- ✅ Modelos se guardan y cargan correctamente
- ✅ Cross-validation funciona

### API Services
- ✅ Requests se validan correctamente
- ✅ Features se transforman apropiadamente
- ✅ Servicios se integran correctamente
- ✅ Errores se propagan apropiadamente
- ✅ Responses tienen formato correcto

### MLflow
- ✅ Experimentos se registran correctamente
- ✅ Modelos se registran en registry
- ✅ Métricas se trackean correctamente
- ✅ Artifacts se almacenan correctamente
- ✅ Búsqueda de runs funciona

## 🐛 Troubleshooting

### ImportError en Tests

```bash
# Verificar instalación
poetry install

# Verificar imports
poetry run python -c "import src; print('OK')"
```

### Tests de DuckDB Fallan

```bash
# Reinstalar DuckDB
poetry add duckdb --force

# Verificar versión
poetry show duckdb
```

### Tests de MLflow Fallan

```bash
# Limpiar experimentos
rm -rf mlruns/

# Reinstalar MLflow
poetry add mlflow --force
```

### Fixtures No Disponibles

```bash
# Listar fixtures disponibles
poetry run pytest tests/integration/ --fixtures

# Verificar conftest.py
cat tests/integration/conftest.py
```

## 📊 Métricas Actuales

- **Total de tests de integración**: 59
- **Tests de data pipeline**: 15
- **Tests de feature-model**: 16
- **Tests de API services**: 13
- **Tests de MLflow**: 15
- **Cobertura esperada**: >70%
- **Tiempo de ejecución**: ~10-20 segundos

## 🔮 Próximos Pasos

### Tests Adicionales Recomendados
- [ ] Integración con Dagster
- [ ] Integración con base de datos real
- [ ] Tests de performance
- [ ] Tests de escalabilidad
- [ ] Integración con sistema de notificaciones

### Mejoras Sugeridas
- [ ] Agregar más fixtures compartidos
- [ ] Implementar test data builders
- [ ] Agregar tests de regresión
- [ ] Mejorar coverage de edge cases
- [ ] Agregar tests de concurrencia

## 📚 Referencias

- [README detallado](../../tests/integration/README.md)
- [Pytest Integration Testing](https://docs.pytest.org/en/stable/goodpractices.html)
- [MLflow Testing](https://mlflow.org/docs/latest/python_api/index.html)
- [DuckDB Testing](https://duckdb.org/docs/api/python/overview)

## 🤝 Contribuir

Para agregar nuevos tests de integración:

1. Identificar componentes a integrar
2. Crear fixtures apropiados
3. Escribir tests con assertions claras
4. Documentar casos de uso
5. Agregar cleanup apropiado
6. Actualizar esta documentación

---

**Última actualización**: 2024-11-15  
**Versión**: 1.0  
**Mantenido por**: MLOps Team - Proyecto Atreides
