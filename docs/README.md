# 📚 Documentación del Proyecto

## 📋 Índice de Documentos

### 🎯 US-005: Análisis de Calidad de Datos

| Documento | Descripción | Audiencia | Tiempo de Lectura |
|-----------|-------------|-----------|-------------------|
| [**US-005_Executive_Summary.md**](US-005_Executive_Summary.md) | ⭐ Resumen ejecutivo de una página - INICIO AQUÍ | Todos | 2 min |
| [**US-005_Resumen_Analisis_Calidad_Datos.md**](US-005_Resumen_Analisis_Calidad_Datos.md) | Resumen ejecutivo completo del análisis de calidad de datos con hallazgos detallados y recomendaciones | Data Engineers, Data Scientists | 30-40 min |
| [**US-005_to_US-006_Handoff.md**](US-005_to_US-006_Handoff.md) | Documento de transición entre US-005 y US-006 con plan de implementación | Data Engineers | 15-20 min |
| [**US-006_Quick_Reference.md**](US-006_Quick_Reference.md) | Guía rápida de implementación con código base del pipeline de limpieza | Data Engineers (implementación) | 10-15 min |
| [**US-005_Summary_Table.md**](US-005_Summary_Table.md) | Resumen visual en formato de tablas para referencia rápida | Todos | 5-10 min |
| [**US-005_to_US-006_Workflow.md**](US-005_to_US-006_Workflow.md) | Diagrama de flujo visual del proceso completo US-005 → US-006 | Todos | 5 min |

---

## 🗂️ Estructura de Documentación

```
docs/
├── README.md                                    # Este archivo (índice)
├── US-005_Executive_Summary.md                 # ⭐ Resumen de 1 página - INICIO
├── US-005_Resumen_Analisis_Calidad_Datos.md   # Resumen ejecutivo completo
├── US-005_to_US-006_Handoff.md                # Documento de transición
├── US-006_Quick_Reference.md                   # Guía rápida de implementación
├── US-005_Summary_Table.md                     # Resumen en tablas
└── US-005_to_US-006_Workflow.md               # Diagrama de flujo visual
```

---

## 📖 Guía de Lectura por Rol

### 👨‍💻 Data Engineer (Implementador de US-006)

**Lectura recomendada en orden:**

1. **Inicio rápido:** [US-006_Quick_Reference.md](US-006_Quick_Reference.md)
   - Código base listo para usar
   - Comandos de ejecución
   - Troubleshooting común

2. **Contexto detallado:** [US-005_to_US-006_Handoff.md](US-005_to_US-006_Handoff.md)
   - Plan de implementación completo
   - Criterios de aceptación
   - Checklist de validación

3. **Referencia rápida:** [US-005_Summary_Table.md](US-005_Summary_Table.md)
   - Tablas de problemas identificados
   - Métricas de éxito
   - Funciones disponibles

4. **Análisis completo (opcional):** [US-005_Resumen_Analisis_Calidad_Datos.md](US-005_Resumen_Analisis_Calidad_Datos.md)
   - Hallazgos detallados
   - Justificación de decisiones

### 👨‍🔬 Data Scientist / ML Engineer

**Lectura recomendada en orden:**

1. **Resumen ejecutivo:** [US-005_Resumen_Analisis_Calidad_Datos.md](US-005_Resumen_Analisis_Calidad_Datos.md)
   - Problemas de calidad identificados
   - Impacto en modelado
   - Recomendaciones

2. **Tablas de referencia:** [US-005_Summary_Table.md](US-005_Summary_Table.md)
   - Estadísticas descriptivas
   - Comparación limpio vs sucio
   - Métricas de calidad

### 👨‍💼 Project Manager / Scrum Master

**Lectura recomendada en orden:**

1. **Resumen visual:** [US-005_Summary_Table.md](US-005_Summary_Table.md)
   - Timeline y estimaciones
   - Entregables completados
   - Próximos pasos

2. **Handoff:** [US-005_to_US-006_Handoff.md](US-005_to_US-006_Handoff.md)
   - Criterios de aceptación US-006
   - Plan de implementación
   - Checklist de validación

---

## 🎯 Resumen Ejecutivo de US-005

### ✅ Estado: COMPLETADO

**Objetivo:** Realizar análisis exhaustivo de calidad de datos del dataset sucio comparándolo con el dataset de referencia limpio.

### 📊 Hallazgos Principales

| Problema | Severidad | Cantidad | Acción US-006 |
|----------|-----------|----------|---------------|
| Tipos de datos incorrectos | 🔴 CRÍTICA | 7 columnas | Convertir String → numérico |
| Valores nulos | 🔴 CRÍTICA | Miles | Imputar o eliminar |
| Violaciones de rango | 🔴 CRÍTICA | Miles | Corregir o marcar nulo |
| Filas adicionales | 🔴 CRÍTICA | +700 | Analizar y eliminar |
| Duplicados exactos | 🟡 ALTA | Cientos | Eliminar |
| Valores atípicos | 🟡 ALTA | Miles | Capping/transformación |
| Columna tipos mixtos | 🟡 ALTA | 1 columna | Eliminar |

### 📦 Entregables

- ✅ Notebook de análisis completo (`notebooks/exploratory/00_data_analysis.ipynb`)
- ✅ 7 módulos utilitarios en `src/utils/` con 30+ funciones
- ✅ Documentación en español
- ✅ 4 documentos de referencia en `docs/`
- ✅ Visualizaciones y comparaciones

### 🎯 Próximo Paso: US-006

**Objetivo:** Crear pipeline de limpieza reproducible que transforme el dataset sucio en limpio con >99% match con referencia.

**Entregables US-006:**
- Script `src/data/clean_data.py`
- Dataset limpio `data/processed/steel_cleaned.parquet`
- Tests unitarios con cobertura >70%
- Versionado DVC (tag: data-v1.0)

---

## 🛠️ Funciones Utilitarias Disponibles

### Resumen de Módulos

| Módulo | Funciones | Propósito |
|--------|-----------|-----------|
| `data_quality.py` | 6 | Análisis de calidad general |
| `outlier_detection.py` | 4 | Detección de valores atípicos |
| `duplicate_detection.py` | 3 | Detección de duplicados |
| `range_validation.py` | 4 | Validación de rangos |
| `visualization.py` | 7 | Visualizaciones |
| `load_datasets.py` | 1 | Carga de datos |
| **Total** | **25** | |

### Uso Rápido

```python
# Importar todas las funciones
from src.utils import *

# Cargar datos
df = load_dataset("data/raw/steel_energy_modified.csv")

# Analizar calidad
nulls = analyze_nulls(df)
outliers = analyze_outliers_all_columns(df)
duplicates = detect_duplicates_exact(df)
validation = validate_ranges(df, range_rules)

# Visualizar
visualize_nulls(nulls)
visualize_outliers_boxplots(df, numeric_cols)
```

---

## 📚 Recursos Adicionales

### Notebooks

| Notebook | Descripción | Ubicación |
|----------|-------------|-----------|
| Análisis de calidad | Análisis completo con visualizaciones | `notebooks/exploratory/00_data_analysis.ipynb` |
| Análisis en español | Traducción del análisis | `notebooks/exploratory/00_analisis_datos.md` |

### Datos

| Dataset | Descripción | Ubicación |
|---------|-------------|-----------|
| Dataset sucio | Datos con problemas de calidad | `data/raw/steel_energy_modified.csv` |
| Dataset limpio | Datos de referencia | `data/raw/steel_energy_original.csv` |

### Código

| Módulo | Descripción | Ubicación |
|--------|-------------|-----------|
| Utilidades | Funciones de análisis de calidad | `src/utils/` |
| Tests | Tests unitarios (futuro) | `tests/` |

---

## 🔗 Enlaces Útiles

### Documentación Externa

- [Polars Documentation](https://pola-rs.github.io/polars/) - Librería de procesamiento de datos
- [DVC Documentation](https://dvc.org/doc) - Versionado de datos
- [Pytest Documentation](https://docs.pytest.org/) - Framework de testing

### Proyecto

- [Plan de Proyecto](../context/PlaneacionProyecto.md) - Plan completo del proyecto
- [README Principal](../README.md) - README del repositorio
- [STRUCTURE.md](../STRUCTURE.md) - Estructura del proyecto

---

## 📝 Convenciones de Documentación

### Formato de Documentos

- **Markdown:** Todos los documentos en formato `.md`
- **Emojis:** Uso de emojis para mejorar legibilidad
- **Tablas:** Formato de tablas para datos estructurados
- **Código:** Bloques de código con syntax highlighting

### Nomenclatura

- `US-XXX_Nombre_Documento.md` - Documentos de User Stories
- `README.md` - Índices y guías
- `UPPERCASE.md` - Documentos de configuración/estructura

### Secciones Estándar

1. **Título y descripción**
2. **Tabla de contenidos** (documentos largos)
3. **Resumen ejecutivo**
4. **Contenido principal**
5. **Referencias**
6. **Metadata** (autor, fecha, versión)

---

## ✅ Checklist de Documentación

### US-005

- [x] Análisis completo documentado
- [x] Hallazgos identificados y priorizados
- [x] Recomendaciones para US-006 definidas
- [x] Funciones utilitarias documentadas
- [x] Visualizaciones generadas
- [x] Traducción al español completada
- [x] Documentos de referencia creados

### US-006 (Pendiente)

- [ ] Script de limpieza documentado
- [ ] Tests unitarios documentados
- [ ] Reporte de limpieza generado
- [ ] Proceso de versionado documentado
- [ ] Guía de uso del dataset limpio

---

## 🤝 Contribuciones

Para contribuir a la documentación:

1. Seguir convenciones de formato
2. Actualizar índice (este README)
3. Incluir metadata (autor, fecha, versión)
4. Revisar ortografía y gramática
5. Validar enlaces y referencias

---

## 📞 Contacto

**Equipo de Desarrollo:**
- **Juan** - Data Engineer (DE)
- **Erick** - Data Scientist (DS)
- **Julian** - Machine Learning Engineer (MLE)
- **Dante** - Software Engineer & Scrum Master (SE)
- **Arthur** - MLOps/SRE Engineer (MLOps)

---

**Última actualización:** 2024  
**Versión:** 1.0  
**Mantenedor:** Data Engineer (Juan)
