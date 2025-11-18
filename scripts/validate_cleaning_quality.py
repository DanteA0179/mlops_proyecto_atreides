"""
Validación completa de calidad de limpieza
Compara dataset de referencia vs dataset limpio
"""

from pathlib import Path

import polars as pl

project_root = Path().resolve()
reference_path = project_root / "data/raw/steel_energy_original.csv"
cleaned_path = project_root / "data/processed/steel_cleaned.parquet"

print("=" * 80)
print("VALIDACIÓN DE CALIDAD DE LIMPIEZA DE DATOS")
print("=" * 80)

# ============================================================================
# 1. CARGAR DATASETS
# ============================================================================
print("\n1. CARGANDO DATASETS...")
df_reference = pl.read_csv(reference_path)
df_cleaned = pl.read_parquet(cleaned_path)

print(f"   Referencia: {df_reference.shape}")
print(f"   Limpio:     {df_cleaned.shape}")

# ============================================================================
# 2. VALIDACIÓN DE ESTRUCTURA
# ============================================================================
print(f"\n{'='*80}")
print("2. VALIDACIÓN DE ESTRUCTURA")
print(f"{'='*80}")

# Columnas
ref_cols = set(df_reference.columns)
clean_cols = set(df_cleaned.columns)

print("\n✓ Columnas:")
if ref_cols == clean_cols:
    print(f"  ✅ Mismo esquema: {len(ref_cols)} columnas")
else:
    print("  ❌ Esquema diferente")
    if ref_cols - clean_cols:
        print(f"     Faltantes en limpio: {ref_cols - clean_cols}")
    if clean_cols - ref_cols:
        print(f"     Extra en limpio: {clean_cols - ref_cols}")

# Tipos
print("\n✓ Tipos de datos:")
type_matches = 0
for col in sorted(ref_cols & clean_cols):
    ref_type = str(df_reference[col].dtype)
    clean_type = str(df_cleaned[col].dtype)
    if ref_type == clean_type:
        type_matches += 1
    else:
        print(f"  ⚠️  {col}: {ref_type} → {clean_type}")

if type_matches == len(ref_cols & clean_cols):
    print(f"  ✅ Todos los tipos coinciden ({type_matches}/{len(ref_cols & clean_cols)})")

# Filas
print("\n✓ Cantidad de filas:")
row_diff = len(df_cleaned) - len(df_reference)
row_diff_pct = abs(row_diff) / len(df_reference) * 100
print(f"  Referencia: {len(df_reference):,}")
print(f"  Limpio:     {len(df_cleaned):,}")
print(f"  Diferencia: {row_diff:+,} ({row_diff_pct:.2f}%)")

if abs(row_diff) <= 120:
    print("  ✅ Dentro de tolerancia (±120 filas)")
else:
    print("  ❌ Fuera de tolerancia")

# ============================================================================
# 3. VALIDACIÓN DE CALIDAD DE DATOS (ALINEADO POR FECHA)
# ============================================================================
print(f"\n{'='*80}")
print("3. VALIDACIÓN DE CALIDAD DE DATOS")
print(f"{'='*80}")

# Join por fecha
df_joined = df_reference.join(df_cleaned, on="date", how="inner", suffix="_clean")
print(f"\n✓ Filas alineadas por fecha: {len(df_joined):,}")
print(f"  Cobertura: {len(df_joined)/len(df_reference)*100:.2f}%")

if len(df_joined) == 0:
    print("\n❌ ERROR: No hay filas comunes. Los timestamps no coinciden.")
    exit(1)

# Comparar valores columna por columna
print("\n✓ Comparación de valores por columna:\n")

base_cols = [col for col in df_reference.columns if col != "date"]
comparison_results = []

for col in sorted(base_cols):
    ref_col = col
    clean_col = f"{col}_clean"

    if clean_col not in df_joined.columns:
        continue

    ref_vals = df_joined[ref_col]
    clean_vals = df_joined[clean_col]

    # Contar coincidencias exactas (con tolerancia para floats)
    if ref_vals.dtype in [pl.Float64, pl.Float32]:
        # Tolerancia de 0.01 para floats
        matches = sum(
            1
            for r, c in zip(ref_vals.to_list(), clean_vals.to_list(), strict=False)
            if (r is None and c is None) or (r is not None and c is not None and abs(r - c) < 0.01)
        )
    else:
        # Coincidencia exacta para enteros y strings
        matches = (ref_vals == clean_vals).sum()

    total = len(df_joined)
    match_pct = (matches / total * 100) if total > 0 else 0
    mismatches = total - matches

    comparison_results.append(
        {
            "column": col,
            "matches": matches,
            "total": total,
            "match_pct": match_pct,
            "mismatches": mismatches,
        }
    )

    # Mostrar resultado
    status = "✅" if match_pct >= 99 else "⚠️" if match_pct >= 95 else "❌"
    print(f"  {status} {col:50} {match_pct:6.2f}% ({matches:,}/{total:,})")

# ============================================================================
# 4. ANÁLISIS DE DIFERENCIAS
# ============================================================================
print(f"\n{'='*80}")
print("4. ANÁLISIS DE DIFERENCIAS")
print(f"{'='*80}")

# Columnas con más diferencias
mismatches_sorted = sorted(comparison_results, key=lambda x: x["mismatches"], reverse=True)

print("\nTop 3 columnas con más diferencias:\n")
for i, result in enumerate(mismatches_sorted[:3], 1):
    if result["mismatches"] > 0:
        col = result["column"]
        print(f"{i}. {col}")
        print(f"   Diferencias: {result['mismatches']:,} ({100-result['match_pct']:.2f}%)")

        # Mostrar ejemplos
        ref_col = col
        clean_col = f"{col}_clean"

        # Encontrar filas con diferencias
        if df_joined[ref_col].dtype in [pl.Float64, pl.Float32]:
            mismatch_mask = [
                abs(r - c) >= 0.01 if r is not None and c is not None else r != c
                for r, c in zip(
                    df_joined[ref_col].to_list(), df_joined[clean_col].to_list(), strict=False
                )
            ]
        else:
            mismatch_mask = (df_joined[ref_col] != df_joined[clean_col]).to_list()

        mismatch_rows = df_joined.filter(pl.Series(mismatch_mask))

        if len(mismatch_rows) > 0:
            print("   Ejemplos (primeros 3):")
            for idx in range(min(3, len(mismatch_rows))):
                row = mismatch_rows[idx]
                ref_val = row[ref_col]
                clean_val = row[clean_col]
                date_val = row["date"]
                print(f"     {date_val}: {ref_val} → {clean_val}")
        print()

# ============================================================================
# 5. ESTADÍSTICAS DESCRIPTIVAS
# ============================================================================
print(f"{'='*80}")
print("5. COMPARACIÓN ESTADÍSTICA")
print(f"{'='*80}\n")

numeric_cols = [col for col in base_cols if df_reference[col].dtype in [pl.Float64, pl.Int64]]

print(f"{'Columna':<50} {'Métrica':<10} {'Ref':>12} {'Limpio':>12} {'Diff %':>10}")
print("-" * 100)

for col in numeric_cols[:5]:  # Primeras 5 columnas numéricas
    ref_col = col
    clean_col = f"{col}_clean"

    if clean_col not in df_joined.columns:
        continue

    ref_vals = df_joined[ref_col]
    clean_vals = df_joined[clean_col]

    # Estadísticas
    stats = [
        ("Mean", ref_vals.mean(), clean_vals.mean()),
        ("Median", ref_vals.median(), clean_vals.median()),
        ("Std", ref_vals.std(), clean_vals.std()),
    ]

    for stat_name, ref_stat, clean_stat in stats:
        if ref_stat is not None and clean_stat is not None and ref_stat != 0:
            diff_pct = abs((clean_stat - ref_stat) / ref_stat * 100)
            status = "✅" if diff_pct < 1 else "⚠️" if diff_pct < 5 else "❌"
            print(
                f"{col:<50} {stat_name:<10} {ref_stat:>12.2f} {clean_stat:>12.2f} {diff_pct:>9.2f}% {status}"
            )

# ============================================================================
# 6. PUNTUACIÓN FINAL
# ============================================================================
print(f"\n{'='*80}")
print("6. PUNTUACIÓN FINAL DE CALIDAD")
print(f"{'='*80}")

# Calcular puntuación global
overall_matches = sum(r["matches"] for r in comparison_results)
overall_total = sum(r["total"] for r in comparison_results)
overall_match_pct = (overall_matches / overall_total * 100) if overall_total > 0 else 0

print(f"\n  Match global: {overall_match_pct:.2f}%")
print(f"  Valores coincidentes: {overall_matches:,} / {overall_total:,}")
print(f"  Valores diferentes: {overall_total - overall_matches:,}")

# Cobertura
coverage_pct = (len(df_joined) / len(df_reference)) * 100

# Evaluación
print(f"\n  Cobertura de filas: {coverage_pct:.2f}%")

# Criterios
print(f"\n{'='*80}")
print("EVALUACIÓN DE CRITERIOS")
print(f"{'='*80}")

checks = [
    ("Schema match", ref_cols == clean_cols, True),
    ("Row coverage (≥99%)", coverage_pct >= 99, False),
    ("Data quality (≥99%)", overall_match_pct >= 99, True),
    ("Data quality (≥95%)", overall_match_pct >= 95, False),
    ("No nulls in cleaned", df_cleaned.null_count().sum_horizontal()[0] == 0, True),
]

for check_name, passed, critical in checks:
    status = "✅" if passed else ("❌" if critical else "⚠️")
    result = "PASS" if passed else ("FAIL" if critical else "ACCEPTABLE")
    print(f"  {status} {check_name:<30} {result}")

# Veredicto final
print(f"\n{'='*80}")
print("VEREDICTO FINAL")
print(f"{'='*80}\n")

if overall_match_pct >= 99:
    print("✅ EXCELENTE: Limpieza cumple criterio de ≥99%")
    print("   Los datos limpios son altamente confiables.")
elif overall_match_pct >= 97:
    print("⚠️  BUENO: Limpieza alcanza 97%")
    print("   Calidad aceptable pero por debajo del criterio de 99%.")
    print("   Revisar si las diferencias son por:")
    print("   - Capping de outliers (aceptable)")
    print("   - Imputación de valores (aceptable)")
    print("   - Errores en la limpieza (NO aceptable)")
elif overall_match_pct >= 95:
    print("⚠️  ACEPTABLE: Limpieza alcanza 95-97%")
    print("   Revisar proceso de limpieza.")
else:
    print("❌ INSUFICIENTE: Limpieza <95%")
    print("   Revisar y corregir proceso de limpieza.")

print(f"\n{'='*80}")
