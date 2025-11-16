# Testing Guide - Energy Optimization Copilot

Esta guía proporciona información completa sobre cómo ejecutar, escribir y mantener tests en el proyecto.

---

## Índice

1. [Visión General](#visión-general)
2. [Configuración Inicial](#configuración-inicial)
3. [Ejecutar Tests](#ejecutar-tests)
4. [Escribir Tests](#escribir-tests)
5. [Estructura de Tests](#estructura-de-tests)
6. [Mejores Prácticas](#mejores-prácticas)
7. [CI/CD Integration](#cicd-integration)
8. [Troubleshooting](#troubleshooting)

---

## Visión General

### Tipos de Tests

El proyecto implementa tres niveles de testing:

**1. Tests Unitarios (`tests/unit/`, `tests/test_*.py`)**
- Testean funciones individuales aisladas
- Rápidos de ejecutar (<1s por test)
- No requieren dependencias externas
- Coverage objetivo: >70% para componentes críticos

**2. Tests de Integración (`tests/integration/`)**
- Testean interacción entre componentes
- Pueden usar mocks para servicios externos
- Tiempo medio de ejecución

**3. Tests End-to-End (`tests/e2e/`)**
- Testean workflows completos
- Requieren servicios corriendo (API, DuckDB)
- Más lentos pero más realistas

### Tecnologías Usadas

- **pytest**: Framework de testing
- **pytest-cov**: Coverage reporting
- **pytest-asyncio**: Testing de código asíncrono
- **pytest-mock**: Mocking utilities
- **pytest-xdist**: Ejecución paralela
- **TestClient (FastAPI)**: Testing de API
- **Polars testing**: Comparación de DataFrames

---

## Configuración Inicial

### Instalar Dependencias

```bash
# Instalar todas las dependencias de desarrollo
poetry install

# Solo instalar dependencias de test
poetry install --with dev
```

### Verificar Instalación

```bash
# Verificar que pytest está instalado
poetry run pytest --version

# Listar todos los tests sin ejecutarlos
poetry run pytest --collect-only
```

---

## Ejecutar Tests

### Comandos Básicos

```bash
# Ejecutar todos los tests
poetry run pytest

# Tests con output verbose
poetry run pytest -v

# Tests con coverage
poetry run pytest --cov=src --cov-report=term-missing

# Tests con coverage HTML
poetry run pytest --cov=src --cov-report=html
# Abrir: htmlcov/index.html
```

### Tests por Categoría

```bash
# Solo tests unitarios
poetry run pytest tests/unit/ -v

# Solo tests de un archivo específico
poetry run pytest tests/test_clean_data.py -v

# Solo tests E2E
poetry run pytest tests/e2e/ -v

# Tests de API (requiere API corriendo)
poetry run pytest tests/e2e/test_api_e2e.py -v
```

### Filtrado de Tests

```bash
# Ejecutar tests por nombre
poetry run pytest -k "test_data_cleaning" -v

# Ejecutar tests por marca
poetry run pytest -m "unit" -v
poetry run pytest -m "integration" -v
poetry run pytest -m "e2e" -v

# Ejecutar un test específico
poetry run pytest tests/test_clean_data.py::TestConvertDataTypes::test_convert_string_to_float -v
```

### Tests Paralelos (Más Rápido)

```bash
# Usar todos los cores disponibles
poetry run pytest -n auto

# Usar 4 workers
poetry run pytest -n 4
```

### Tests con Debugging

```bash
# Mostrar prints en tests
poetry run pytest -s

# Entrar en debugger en fallos
poetry run pytest --pdb

# Mostrar locals en traceback
poetry run pytest -l

# Detener en primer fallo
poetry run pytest -x

# Ver solo último traceback
poetry run pytest --tb=short
```

---

## Escribir Tests

### Anatomía de un Test

```python
import pytest
import polars as pl
from src.utils.data_cleaning import convert_data_types

class TestConvertDataTypes:
    """Tests for convert_data_types function"""
    
    def test_convert_string_to_float(self):
        """Test conversion from String to Float64"""
        # ARRANGE - Preparar datos de entrada
        df = pl.DataFrame({
            'col1': ['1.5', '2.5', '3.5'],
            'col2': ['a', 'b', 'c']
        })
        schema = {'col1': pl.Float64, 'col2': pl.Utf8}
        
        # ACT - Ejecutar función bajo test
        result = convert_data_types(df, schema)
        
        # ASSERT - Verificar resultados
        assert result['col1'].dtype == pl.Float64
        assert result['col2'].dtype == pl.Utf8
        assert len(result) == 3
```

### Patrón AAA (Arrange-Act-Assert)

**SIEMPRE** estructurar tests con este patrón:

1. **Arrange**: Preparar datos y condiciones
2. **Act**: Ejecutar la función/método bajo test
3. **Assert**: Verificar que el resultado es el esperado

### Fixtures

#### Fixtures Built-in de Pytest

```python
def test_with_tmp_path(tmp_path):
    """Test usando directorio temporal"""
    file_path = tmp_path / "test.csv"
    df.write_csv(file_path)
    assert file_path.exists()

def test_with_monkeypatch(monkeypatch):
    """Test mockeando variables de entorno"""
    monkeypatch.setenv("MODEL_PATH", "/fake/path")
    result = load_model()
    assert result is not None
```

#### Fixtures Personalizadas

```python
# tests/conftest.py
import pytest
import polars as pl

@pytest.fixture
def sample_dataframe():
    """Fixture que retorna un DataFrame de ejemplo"""
    return pl.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': ['a', 'b', 'c', 'd', 'e']
    })

@pytest.fixture
def sample_dataframe_with_nulls():
    """Fixture con nulls"""
    return pl.DataFrame({
        'col1': [1, None, 3, None, 5],
        'col2': ['a', 'b', None, 'd', 'e']
    })

# Usar en tests
def test_with_fixture(sample_dataframe):
    result = process_data(sample_dataframe)
    assert len(result) == 5
```

#### Fixture Scopes

```python
@pytest.fixture(scope="session")
def database_connection():
    """Fixture que se ejecuta una vez por sesión"""
    conn = setup_database()
    yield conn
    conn.close()

@pytest.fixture(scope="module")
def ml_model():
    """Fixture que se ejecuta una vez por módulo"""
    return load_model()

@pytest.fixture(scope="function")  # Default
def temp_data():
    """Fixture que se ejecuta para cada test"""
    return generate_data()
```

### Parametrización

Ejecutar mismo test con múltiples inputs:

```python
import pytest

@pytest.mark.parametrize("input_value,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
    (10, 20),
])
def test_double(input_value, expected):
    """Test función double con múltiples inputs"""
    result = double(input_value)
    assert result == expected

@pytest.mark.parametrize("load_type", ["Light", "Medium", "Maximum"])
def test_predict_all_load_types(load_type):
    """Test predicción con todos los tipos de carga"""
    result = predict(load_type=load_type)
    assert result > 0
```

### Mocking

#### Mock Simple

```python
from unittest.mock import Mock, patch

def test_api_call_with_mock():
    """Test mockeando llamada a API externa"""
    mock_response = Mock()
    mock_response.json.return_value = {"data": [1, 2, 3]}
    
    with patch('requests.get', return_value=mock_response):
        result = fetch_data()
        assert result == [1, 2, 3]
```

#### Mock con pytest-mock

```python
def test_with_mocker(mocker):
    """Test usando pytest-mock plugin"""
    mock_model = mocker.Mock()
    mock_model.predict.return_value = [45.67]
    
    mocker.patch('src.api.services.model_service.load_model', 
                 return_value=mock_model)
    
    result = predict_api()
    assert result["predicted_usage_kwh"] == 45.67
```

### Testing de Exceptions

```python
import pytest

def test_exception_raised():
    """Test que se lanza excepción esperada"""
    with pytest.raises(ValueError):
        process_invalid_data()

def test_exception_message():
    """Test mensaje de excepción"""
    with pytest.raises(ValueError, match="Invalid load type"):
        process_invalid_data()
```

### Testing de APIs (FastAPI)

```python
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_endpoint():
    """Test endpoint de health check"""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

def test_predict_endpoint():
    """Test endpoint de predicción"""
    payload = {
        "lagging_reactive_power": 23.45,
        "leading_reactive_power": 12.30,
        "co2": 0.05,
        "lagging_power_factor": 0.85,
        "leading_power_factor": 0.92,
        "nsm": 36000,
        "day_of_week": 1,
        "load_type": "Medium"
    }
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "predicted_usage_kwh" in data
    assert data["predicted_usage_kwh"] > 0
```

### Testing de Polars DataFrames

```python
import polars as pl
from polars.testing import assert_frame_equal

def test_dataframe_equality():
    """Test que dos DataFrames son iguales"""
    df1 = pl.DataFrame({'col1': [1, 2, 3]})
    df2 = pl.DataFrame({'col1': [1, 2, 3]})
    
    assert_frame_equal(df1, df2)

def test_dataframe_schema():
    """Test schema de DataFrame"""
    df = pl.DataFrame({
        'col1': [1, 2, 3],
        'col2': [1.0, 2.0, 3.0]
    })
    
    assert df['col1'].dtype == pl.Int64
    assert df['col2'].dtype == pl.Float64
```

---

## Estructura de Tests

### Organización de Archivos

```
tests/
├── conftest.py              # Fixtures globales
├── __init__.py
│
├── unit/                    # Tests unitarios
│   ├── conftest.py          # Fixtures de unit tests
│   ├── test_api_endpoints.py
│   ├── test_duckdb_utils.py
│   ├── test_preprocessing_utils.py
│   └── ...
│
├── integration/             # Tests de integración
│   ├── conftest.py
│   └── ...
│
├── e2e/                     # Tests end-to-end
│   ├── conftest.py
│   ├── test_api_e2e.py
│   └── test_pipeline_e2e.py
│
└── test_*.py               # Tests legacy (refactorizar a unit/)
```

### Naming Conventions

**Archivos:**
- `test_<module>.py` para tests de un módulo
- `test_<feature>.py` para tests de una feature

**Clases:**
- `TestClassName` para agrupar tests relacionados
- Una clase por función/clase bajo test

**Funciones:**
- `test_<functionality>` describe qué se testea
- `test_<functionality>_<condition>` para casos específicos

**Ejemplos:**
```python
# Archivo: test_data_cleaning.py
class TestConvertDataTypes:
    def test_convert_string_to_float(self):
        pass
    
    def test_convert_with_nulls(self):
        pass
    
    def test_convert_empty_dataframe(self):
        pass
```

---

## Mejores Prácticas

### 1. Tests Independientes

**❌ Incorrecto:**
```python
class TestDataProcessing:
    def test_step_1(self):
        self.data = load_data()  # Guarda estado
    
    def test_step_2(self):
        result = process(self.data)  # Depende de test anterior
```

**✅ Correcto:**
```python
class TestDataProcessing:
    @pytest.fixture
    def data(self):
        return load_data()
    
    def test_step_1(self, data):
        result = step_1(data)
        assert result is not None
    
    def test_step_2(self, data):
        result = step_2(data)
        assert result is not None
```

### 2. Un Assert por Test (Preferiblemente)

**❌ Incorrecto:**
```python
def test_multiple_things():
    result = process_data()
    assert len(result) == 5
    assert result['col1'].sum() == 15
    assert result['col2'].mean() == 2.5
    assert result['col3'].max() == 100
    # Si falla el primero, no sabemos del resto
```

**✅ Correcto:**
```python
def test_result_length():
    result = process_data()
    assert len(result) == 5

def test_result_col1_sum():
    result = process_data()
    assert result['col1'].sum() == 15

def test_result_col2_mean():
    result = process_data()
    assert result['col2'].mean() == 2.5
```

### 3. Tests Descriptivos

**❌ Incorrecto:**
```python
def test_1():
    """Test data"""
    pass

def test_function():
    """Test that it works"""
    pass
```

**✅ Correcto:**
```python
def test_convert_negative_values_to_absolute():
    """Test that negative values are converted to absolute"""
    pass

def test_handle_missing_values_fills_with_median():
    """Test that missing values are filled with column median"""
    pass
```

### 4. Test Edge Cases

Siempre testear:
- Valores límite (0, max, min)
- Valores nulos/None
- DataFrames vacíos
- Listas vacías
- Strings vacíos
- Valores negativos
- Valores extremadamente grandes

```python
def test_with_empty_dataframe():
    df = pl.DataFrame()
    result = process(df)
    assert len(result) == 0

def test_with_null_values():
    df = pl.DataFrame({'col1': [1, None, 3]})
    result = process(df)
    assert result['col1'].null_count() == 0

def test_with_negative_values():
    df = pl.DataFrame({'col1': [-5, -10, -15]})
    result = process(df)
    assert all(result['col1'] >= 0)
```

### 5. Tests Rápidos

- Mock dependencias externas (APIs, DB, filesystems)
- Usa fixtures para datos reutilizables
- Evita sleep() cuando sea posible
- Usa datos pequeños pero representativos

### 6. Tests Mantenibles

```python
# Usa constantes para valores mágicos
EXPECTED_LENGTH = 100
THRESHOLD = 0.5

def test_processing():
    result = process_data()
    assert len(result) == EXPECTED_LENGTH
    assert result.mean() > THRESHOLD
```

### 7. Docstrings en Tests

```python
def test_convert_data_types_with_nulls():
    """
    Test that convert_data_types preserves null values.
    
    Given a DataFrame with null values,
    When convert_data_types is called,
    Then nulls should remain as nulls in the output.
    """
    pass
```

---

## CI/CD Integration

### GitHub Actions

Los tests se ejecutan automáticamente en CI con cada push/PR.

**Workflow:** `.github/workflows/tests.yml`

**Jobs:**
1. **test**: Ejecuta tests unitarios en Python 3.11 y 3.12
2. **lint**: Ejecuta Black y Ruff
3. **e2e-tests**: Ejecuta tests E2E
4. **security**: Ejecuta Bandit

**Ver resultados:**
- GitHub Actions tab en el repo
- Coverage report en Codecov
- Test results en PR checks

### Pre-commit Hooks

Los tests se ejecutan localmente antes de cada commit.

**Instalar hooks:**
```bash
poetry run pre-commit install
```

**Ejecutar manualmente:**
```bash
poetry run pre-commit run --all-files
```

**Configuración:** `.pre-commit-config.yaml`

### Coverage Thresholds

El proyecto requiere:
- Componentes críticos: >70% coverage
- API routes: >70% coverage
- Utils: >70% coverage
- Scripts de training: No requerido

**Verificar coverage:**
```bash
poetry run pytest --cov=src --cov-fail-under=70
```

---

## Troubleshooting

### Tests Fallan Localmente pero Pasan en CI

**Causa:** Diferencias en el entorno

**Solución:**
```bash
# Asegurarse de tener dependencias actualizadas
poetry install

# Limpiar cache de pytest
poetry run pytest --cache-clear

# Verificar versión de Python
python --version  # Debe ser 3.11+
```

### Tests Muy Lentos

**Solución:**
```bash
# Ejecutar en paralelo
poetry run pytest -n auto

# Identificar tests lentos
poetry run pytest --durations=10

# Ejecutar solo tests rápidos
poetry run pytest -m "not slow"
```

### Import Errors en Tests

**Causa:** Módulo src no está en PYTHONPATH

**Solución:**
```bash
# Instalar proyecto en modo editable
poetry install

# O agregar al inicio del test
import sys
sys.path.append('../..')
```

### Tests de API Fallan

**Causa:** API no está corriendo o modelo no está cargado

**Solución:**
```bash
# Para tests E2E que requieren API:
# 1. Iniciar API en otra terminal
uvicorn src.api.main:app --reload

# 2. Ejecutar tests E2E
poetry run pytest tests/e2e/test_api_e2e.py -v

# Para tests unitarios de API (con mocks):
# No requieren API corriendo
poetry run pytest tests/unit/test_api_endpoints.py -v
```

### Coverage Bajo

**Verificar qué falta testear:**
```bash
# Generar reporte HTML
poetry run pytest --cov=src --cov-report=html

# Abrir en navegador
open htmlcov/index.html

# Ver en terminal qué líneas faltan
poetry run pytest --cov=src --cov-report=term-missing
```

---

## Recursos Adicionales

### Documentación

- [Pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Polars Testing](https://pola-rs.github.io/polars/py-polars/html/reference/testing.html)

### Archivos del Proyecto

- **Epic 9 Summary:** `docs/epic-9-testing-summary.md`
- **Pytest Config:** `pyproject.toml` (section `[tool.pytest.ini_options]`)
- **Coverage Config:** `pyproject.toml` (section `[tool.coverage.run]`)
- **CI Workflow:** `.github/workflows/tests.yml`

### Comandos Útiles

```bash
# Ver fixtures disponibles
poetry run pytest --fixtures

# Ver marcadores disponibles
poetry run pytest --markers

# Ejecutar tests con output completo
poetry run pytest -vv --tb=long

# Generar reporte XML para CI
poetry run pytest --junitxml=junit.xml
```

---

**Versión:** 1.0  
**Última actualización:** 2025-11-15  
**Mantenido por:** MLOps Team
