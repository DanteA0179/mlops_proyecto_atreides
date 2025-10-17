# Análisis Exploratorio de Datos (EDA)

Este directorio contiene notebooks de análisis exploratorio para el proyecto Energy Optimization Copilot.

## 📋 Notebooks Disponibles

### `03_eda_refact.ipynb` ⭐ **RECOMENDADO - Dataset Limpio**

Notebook refactorizado siguiendo las mejores prácticas del proyecto (ver [AGENTS.md](../../AGENTS.md)).

**Características:**
- ✅ Código en inglés, documentación en español
- ✅ Uso de funciones reutilizables de `src/utils/`
- ✅ Visualizaciones interactivas con Plotly
- ✅ Estadísticas descriptivas completas
- ✅ 6+ visualizaciones requeridas por US-008
- ✅ Exportación automática de figuras a `reports/figures/`
- ✅ Sección de conclusiones exhaustiva

**Dataset:** `steel_cleaned.parquet` (limpio, versionado con DVC)

**Contenido:**
1. Setup y Configuración
2. Carga de Datos (usando `duckdb_utils`)
3. Estadísticas Descriptivas
4. Visualizaciones:
   - Distribución de Usage_kWh (target)
   - Matriz de Correlación
   - Consumo por Hora del Día
   - Consumo por Día de Semana
   - Scatter: CO2 vs Usage_kWh
   - Scatter Matrix: Top 5 Features
5. Conclusiones

**Ejecución:**
```bash
# Desde la raíz del proyecto
cd notebooks/exploratory
poetry run jupyter lab 03_eda_refact.ipynb
```

---

### `03b_eda_refact_orig.ipynb` 🔍 **Dataset Original**

Análisis del dataset ORIGINAL (sin limpiar) para comparación y validación del proceso de limpieza.

**Características:**
- ✅ Misma estructura que `03_eda_refact.ipynb`
- ✅ Lee directamente desde CSV (no requiere DuckDB)
- ✅ Análisis de calidad de datos incluido
- ✅ Comparación lado a lado: Original vs Limpio
- ✅ Validación del proceso de data cleaning

**Dataset:** `steel_energy_original.csv` (original sin procesar)

**Contenido adicional:**
1. Análisis de Calidad de Datos
   - Valores nulos
   - Duplicados
   - Outliers
2. Comparación Original vs Limpio
   - Diferencias estadísticas
   - Visualizaciones comparativas
   - Impacto del proceso de limpieza
3. Conclusiones sobre Data Quality

**Ejecución:**
```bash
# Desde la raíz del proyecto
cd notebooks/exploratory
poetry run jupyter lab 03b_eda_refact_orig.ipynb
```

### Otros Notebooks

- `EDA_Erick.ipynb`: Versión inicial de Erick (POO approach)
- `eda_basic,_dante.ipynb`: Análisis básico de Dante

## 🛠️ Funciones Reutilizables

El notebook `03_eda_refact.ipynb` utiliza funciones de:

### `src/utils/eda_plots.py`

```python
from src.utils.eda_plots import (
    plot_distribution,           # Distribuciones con histograma + KDE
    plot_correlation_heatmap,    # Matriz de correlación interactiva
    plot_time_series,            # Series temporales con agregación
    plot_box_by_category,        # Box plots por categoría
    plot_scatter,                # Scatter plots con trendline
    plot_scatter_matrix,         # Pairplot interactivo
    get_top_correlated_features  # Top N features correlacionadas
)
```

### `src/utils/duckdb_utils.py`

```python
from src.utils.duckdb_utils import (
    setup_database,      # Setup automático de DB
    quick_query,         # Query rápido sin manejar conexión
    get_stats_by_column, # Estadísticas por columna
    get_temporal_stats,  # Análisis temporal
    get_correlation      # Correlaciones
)
```

## 📊 Figuras Generadas

Todas las visualizaciones se exportan automáticamente a:

```
reports/figures/
# Dataset Limpio (03_eda_refact.ipynb)
├── 01_usage_distribution.html       # Distribución de Usage_kWh
├── 02_correlation_matrix.html       # Matriz de correlación
├── 03_hourly_consumption.html       # Consumo por hora
├── 04_weekday_consumption.html      # Consumo por día de semana
├── 04b_weekstatus_consumption.html  # Weekday vs Weekend
├── 04c_loadtype_consumption.html    # Consumo por Load_Type
├── 05_co2_vs_usage.html            # CO2 vs Usage scatter
├── 06_scatter_matrix.html          # Scatter matrix top 5
├── 07_power_factor_dist.html       # Distribución power factor
├── 08_reactive_power_vs_usage.html # Reactive power scatter
│
# Dataset Original (03b_eda_refact_orig.ipynb)
├── orig_01_usage_distribution.html       # Distribución original
├── orig_02_correlation_matrix.html       # Correlación original
├── orig_03_hourly_consumption.html       # Consumo por hora original
├── orig_04_weekday_consumption.html      # Por día de semana original
├── orig_04b_weekstatus_consumption.html  # Weekday vs Weekend original
├── orig_04c_loadtype_consumption.html    # Por Load_Type original
├── orig_05_co2_vs_usage.html            # CO2 vs Usage original
├── orig_06_scatter_matrix.html          # Scatter matrix original
│
# Comparaciones
└── comparison_distributions.html    # Comparación lado a lado
```

## 🎯 User Story US-008

### Criterios de Aceptación ✅

- [x] Notebook `03_eda_refact.ipynb` creado
- [x] Estadísticas descriptivas completas
- [x] Distribución de cada variable (histogramas + KDE)
- [x] Correlation matrix + heatmap
- [x] Pairplots de variables clave
- [x] Boxplots por Load_Type
- [x] Sección de conclusiones escritas
- [x] Exportar figuras a `reports/figures/`

### Visualizaciones Requeridas ✅

1. ✅ Distribución de Usage_kWh (target)
2. ✅ Correlation heatmap (todas las variables)
3. ✅ Consumo por hora del día (line plot)
4. ✅ Consumo por día de semana (box plot)
5. ✅ Scatter: CO2 vs Usage_kWh
6. ✅ Scatter matrix: top 5 features correlacionadas

## 🧪 Testing

Las funciones de visualización tienen cobertura de tests >70%:

```bash
# Ejecutar tests
poetry run pytest tests/test_eda_plots.py -v

# Con coverage
poetry run pytest tests/test_eda_plots.py --cov=src.utils.eda_plots --cov-report=html
```

## 📚 Mejores Prácticas Seguidas

Según [AGENTS.md](../../AGENTS.md):

1. ✅ **Idioma**: Texto en español, código en inglés
2. ✅ **Funciones Reutilizables**: Todo en `src/utils/`, no código inline
3. ✅ **Type Hints**: Todas las funciones tienen type hints
4. ✅ **Docstrings**: Estilo Google en todas las funciones
5. ✅ **Visualizaciones**: Polars + Plotly para velocidad e interactividad
6. ✅ **Estructura**: Orden estándar de notebooks
7. ✅ **Testing**: >70% coverage en utilidades

## 💡 Tips de Uso

### Modificar Colores

```python
# Personalizar colores en visualizaciones
fig = plot_distribution(
    df,
    'Usage_kWh',
    color='#e74c3c'  # Rojo personalizado
)
```

### Agregar Más Features

```python
# Obtener top 10 features correlacionadas
top_10 = get_top_correlated_features(
    df,
    'Usage_kWh',
    n=10,
    exclude_columns=['date', 'NSM']  # Excluir temporales
)
```

### Exportar a PNG

```python
# Guardar como imagen estática
fig = plot_correlation_heatmap(df)
fig.write_image("reports/figures/correlation.png")  # Requiere kaleido
```

## 🔗 Referencias

- [AGENTS.md](../../AGENTS.md): Estándares del proyecto
- [Plotly Documentation](https://plotly.com/python/)
- [Polars Documentation](https://pola-rs.github.io/polars/)
- [DuckDB Documentation](https://duckdb.org/docs/)

---

**Última actualización**: 2025-10-16
**Responsable**: MLOps Team - Proyecto Atreides
