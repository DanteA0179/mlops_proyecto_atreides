# Contributing to Energy Optimization Copilot

¡Gracias por tu interés en contribuir al proyecto! Este documento proporciona guías y mejores prácticas para colaborar efectivamente.

## 📋 Tabla de Contenidos

1. [Código de Conducta](#código-de-conducta)
2. [Cómo Contribuir](#cómo-contribuir)
3. [Configuración del Entorno](#configuración-del-entorno)
4. [Convenciones de Código](#convenciones-de-código)
5. [Git Workflow](#git-workflow)
6. [Testing](#testing)
7. [Documentación](#documentación)
8. [Revisión de Código](#revisión-de-código)

## 🤝 Código de Conducta

Este proyecto sigue el estándar de conducta profesional esperado en un ambiente académico de maestría:

- **Respeto**: Trata a todos los colaboradores con respeto y profesionalismo
- **Colaboración**: Comparte conocimiento y ayuda a tus compañeros
- **Responsabilidad**: Cumple con tus compromisos y deadlines del sprint
- **Comunicación**: Mantén al equipo informado de tu progreso y bloqueadores

## 🚀 Cómo Contribuir

### Tipos de Contribuciones

1. **Bug fixes**: Correcciones de errores
2. **Features**: Nuevas funcionalidades según el plan del proyecto
3. **Documentation**: Mejoras a la documentación
4. **Tests**: Nuevos tests o mejoras a la cobertura
5. **Refactoring**: Mejoras de código sin cambiar funcionalidad

### Proceso General

1. Revisa el backlog en Linear o GitHub Projects
2. Asigna una tarea a ti mismo
3. Crea un branch desde `develop`
4. Desarrolla tu cambio
5. Ejecuta tests localmente
6. Crea un Pull Request
7. Espera revisión de al menos 1 compañero
8. Merge cuando la PR sea aprobada

## ⚙️ Configuración del Entorno

### Prerrequisitos

```bash
# Versiones requeridas
Python 3.11+
Poetry 1.8+
Git 2.40+
DVC 3.48+
Docker 24.x (opcional)
```

### Setup Inicial

```bash
# 1. Clonar repositorio
git clone https://github.com/your-org/energy-optimization-copilot.git
cd energy-optimization-copilot

# 2. Instalar dependencias con Poetry
poetry install

# 3. Activar entorno virtual
poetry shell

# 4. Configurar pre-commit hooks
poetry run pre-commit install

# 5. Configurar DVC (si tienes acceso a GCS)
dvc pull

# 6. Copiar variables de entorno
cp .env.example .env
# Editar .env con tus valores locales
```

### Verificar Instalación

```bash
# Ejecutar tests
poetry run pytest

# Verificar linting
poetry run ruff check .
poetry run black --check .

# Verificar que imports funcionen
poetry run python -c "import polars, pandas, sklearn; print('✅ OK')"
```

## 📝 Convenciones de Código

### Python Style Guide

Seguimos [PEP 8](https://peps.python.org/pep-0008/) con las siguientes configuraciones:

- **Formateo**: Black (line-length=100)
- **Linting**: Ruff
- **Type hints**: Recomendado para funciones públicas
- **Docstrings**: Google style

### Ejemplo de Código

```python
"""Module docstring describing the module purpose."""

from typing import Optional
import polars as pl


def clean_steel_data(
    df: pl.DataFrame,
    remove_outliers: bool = True,
    outlier_std: float = 3.0
) -> pl.DataFrame:
    """Clean steel energy consumption dataset.

    Args:
        df: Raw DataFrame with steel data
        remove_outliers: Whether to remove outliers
        outlier_std: Number of standard deviations for outlier detection

    Returns:
        Cleaned DataFrame ready for feature engineering

    Raises:
        ValueError: If required columns are missing

    Example:
        >>> raw_df = pl.read_csv("data/raw/steel_dirty.csv")
        >>> clean_df = clean_steel_data(raw_df)
        >>> print(clean_df.shape)
        (34500, 10)
    """
    # Implementation here
    pass
```

### Convenciones de Nombres

```python
# Archivos
clean_data.py          # snake_case para módulos
BuildFeatures          # PascalCase solo para clases

# Variables y funciones
energy_consumption     # snake_case
calculate_rmse()       # snake_case con verbos

# Constantes
MAX_FEATURES = 10      # UPPER_SNAKE_CASE
MODEL_PATH = "models/" # UPPER_SNAKE_CASE

# Clases
class EnergyPredictor:  # PascalCase
    pass

# Privadas
_internal_helper()      # prefijo _ para funciones privadas
__private_attr         # prefijo __ para atributos muy privados
```

### Estructura de Archivos Python

```python
"""Module docstring."""

# 1. Future imports
from __future__ import annotations

# 2. Standard library
import os
import sys
from pathlib import Path

# 3. Third-party
import polars as pl
import numpy as np
from sklearn.metrics import mean_squared_error

# 4. Local imports
from src.utils.logging import setup_logger
from src.features.transformers import TimeSeriesFeatures

# 5. Constants
LOGGER = setup_logger(__name__)
DEFAULT_WINDOW = 24

# 6. Functions and classes
def main():
    pass
```

## 🌿 Git Workflow

### Branch Naming

```bash
# Formato: tipo/EP-XXX-descripcion-corta

feature/EP-001-project-setup
feature/EP-005-chronos-model
bugfix/EP-042-fix-parquet-schema
hotfix/critical-memory-leak
docs/update-readme
test/add-integration-tests
refactor/improve-data-pipeline
```

### Commits

Seguimos [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Formato
<type>(<scope>): <subject>

# Tipos
feat: Nueva funcionalidad
fix: Corrección de bug
docs: Cambios en documentación
style: Formateo, sin cambios de código
refactor: Refactoring de código
test: Agregar o modificar tests
chore: Tareas de mantenimiento
perf: Mejoras de performance
ci: Cambios en CI/CD

# Ejemplos
feat(data): add data cleaning pipeline
fix(models): correct RMSE calculation
docs: update README with DVC instructions
test(features): add tests for time series features
refactor(api): simplify prediction endpoint
chore: update dependencies
```

### Pull Requests

**Título del PR:**
```
<type>: <descripción> (#EP-XXX)

Ejemplo:
feat: Implement Chronos fine-tuning pipeline (#EP-005)
```

**Template del PR:**

```markdown
## 📋 Descripción
[Descripción clara del cambio]

## 🔗 Issue Relacionada
Closes #EP-XXX

## 🔄 Tipo de Cambio
- [ ] Bug fix (cambio que corrige un issue)
- [ ] Nueva feature (cambio que agrega funcionalidad)
- [ ] Breaking change (fix o feature que causa que funcionalidad existente no funcione como antes)
- [ ] Actualización de documentación

## ✅ Checklist
- [ ] Mi código sigue las convenciones del proyecto
- [ ] He realizado self-review de mi código
- [ ] He comentado código complejo cuando necesario
- [ ] He actualizado la documentación correspondiente
- [ ] Mis cambios no generan nuevas advertencias
- [ ] He agregado tests que prueban mi fix/feature
- [ ] Todos los tests nuevos y existentes pasan localmente
- [ ] He actualizado archivos DVC si cambié datos/modelos

## 🧪 Testing
[Describe cómo testeaste los cambios]

## 📸 Screenshots (si aplica)
[Agrega screenshots de cambios visuales]

## 📝 Notas Adicionales
[Cualquier información extra para los revisores]
```

### Merge Strategy

- **Squash and merge** para features pequeños
- **Merge commit** para features grandes con historia importante
- **Rebase and merge** para mantener historia lineal (preferido)

## 🧪 Testing

### Escribir Tests

```python
# tests/unit/test_data_cleaning.py
import pytest
import polars as pl
from src.data.clean_data import remove_outliers


class TestDataCleaning:
    """Test suite for data cleaning functions."""

    @pytest.fixture
    def sample_data(self):
        """Sample DataFrame with outliers."""
        return pl.DataFrame({
            "value": [1, 2, 3, 100, 4, 5, -50, 6]
        })

    def test_remove_outliers_default(self, sample_data):
        """Test outlier removal with default parameters."""
        result = remove_outliers(sample_data, column="value")
        assert len(result) < len(sample_data)
        assert result["value"].max() < 100

    def test_remove_outliers_custom_std(self, sample_data):
        """Test outlier removal with custom std."""
        result = remove_outliers(sample_data, column="value", std=2.0)
        assert len(result) > 0

    @pytest.mark.parametrize("std,expected_len", [
        (1.0, 6),
        (2.0, 7),
        (3.0, 8),
    ])
    def test_remove_outliers_parametrized(self, sample_data, std, expected_len):
        """Test outlier removal with different std values."""
        result = remove_outliers(sample_data, column="value", std=std)
        assert len(result) == expected_len
```

### Ejecutar Tests

```bash
# Todos los tests
poetry run pytest

# Tests específicos
poetry run pytest tests/unit/test_data_cleaning.py

# Con coverage
poetry run pytest --cov=src --cov-report=html

# Solo tests rápidos (omitir tests lentos)
poetry run pytest -m "not slow"

# Ver output detallado
poetry run pytest -v -s
```

### Coverage Requirements

- **Unit tests**: >70% coverage mínimo
- **Integration tests**: Paths críticos cubiertos
- **E2E tests**: Flujos principales de usuario

## 📚 Documentación

### Docstrings

Usamos [Google Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings):

```python
def train_model(
    data: pl.DataFrame,
    model_type: str = "chronos",
    hyperparameters: Optional[dict] = None
) -> tuple[object, dict]:
    """Train ML model on energy consumption data.

    Trains either a classical model (XGBoost, LightGBM) or a Foundation
    Model (Chronos, TimesFM) depending on model_type parameter.

    Args:
        data: Training data with features and target
        model_type: Type of model to train. One of: 'chronos', 'xgboost',
            'lightgbm', 'timesfm'
        hyperparameters: Optional dict with hyperparameters. If None,
            uses defaults from config/model_config.yaml

    Returns:
        Tuple containing:
            - Trained model object
            - Dict with training metrics (rmse, mae, cv)

    Raises:
        ValueError: If model_type is not recognized
        RuntimeError: If training fails after max_retries

    Example:
        >>> import polars as pl
        >>> data = pl.read_parquet("data/processed/train.parquet")
        >>> model, metrics = train_model(data, model_type="chronos")
        >>> print(f"RMSE: {metrics['rmse']:.4f}")
        RMSE: 0.2015

    Note:
        This function logs all metrics to MLflow automatically.
        Check MLflow UI for detailed training history.
    """
    pass
```

### README Updates

Actualiza el README cuando:
- Agregas nuevas features
- Cambias proceso de setup
- Modificas estructura del proyecto
- Agregas nuevas dependencias

### Documentación en Markdown

```markdown
# Título de Documento

Breve descripción del contenido.

## Sección Principal

### Subsección

- Usa listas para enumerar items
- Incluye ejemplos de código
- Agrega diagramas cuando ayuden

\```python
# Código de ejemplo
def example():
    pass
\```

## Referencias

- [Link relevante](https://example.com)
```

## 🔍 Revisión de Código

### Como Autor del PR

1. **Self-review**: Revisa tu código antes de crear el PR
2. **Descripción clara**: Explica qué cambios hiciste y por qué
3. **Tests**: Asegúrate de que todos los tests pasen
4. **Tamaño**: Mantén PRs pequeños (<500 líneas cuando posible)
5. **Responde rápido**: Atiende comentarios de revisores en <24h

### Como Revisor

1. **Tiempo**: Revisa PRs en <48h
2. **Constructivo**: Provee feedback específico y constructivo
3. **Pregunta**: Si no entiendes algo, pregunta
4. **Aprueba o rechaza**: No dejes PRs sin decision
5. **Checklist de revisión**:
   - [ ] Código sigue convenciones del proyecto
   - [ ] Tests cubren los cambios
   - [ ] Documentación actualizada
   - [ ] No hay código comentado/dead code
   - [ ] Variables bien nombradas
   - [ ] Funciones son cohesivas (una responsabilidad)
   - [ ] No hay hardcoded values (usar config)

### Comentarios en Review

```markdown
# Sugerencias constructivas

❌ "Este código está mal"
✅ "Considera usar list comprehension aquí para mejor legibilidad:
   `result = [x*2 for x in items]`"

❌ "No me gusta esto"
✅ "Esta función hace múltiples cosas. ¿Podrías dividirla en
   `load_data()` y `preprocess_data()` para mejor mantenibilidad?"

❌ "Falta documentación"
✅ "Agrega docstring explicando qué hace esta función y qué
   retorna. Ver CONTRIBUTING.md para formato."
```

## 📞 Contacto y Soporte

### Canales de Comunicación

- **Daily Standups**: Discord/Slack (martes-viernes, asíncrono)
- **Sprint Planning**: Reunión síncrona (lunes, inicio de sprint)
- **Bloqueadores**: Notificar en Discord inmediatamente
- **Code Review**: Comentarios en GitHub PR

### Preguntas Frecuentes

**P: ¿Puedo trabajar en múltiples tareas simultáneamente?**
R: No. Enfócate en una tarea hasta completarla (Definition of Done).

**P: ¿Qué hago si encuentro un bug?**
R: Crea un issue en GitHub/Linear y notifica en Daily Standup.

**P: ¿Cómo reporto problemas con DVC/MLflow?**
R: Primero consulta documentación, luego pregunta al MLOps engineer (Arthur).

**P: ¿Debo hacer commit de mis notebooks?**
R: Sí, pero usa `nbstripout` para limpiar outputs (configurado en pre-commit).

---

**¡Gracias por contribuir! 🚀**

*Última actualización: Octubre 2025*
