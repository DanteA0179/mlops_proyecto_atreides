"""
Create visualizations for Chronos-2 evaluation results.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

# Load results
results_file = Path("models/foundation/chronos2_results_20251027_230102.json")
with open(results_file) as f:
    results = json.load(f)

# Load test data to get predictions
data_dir = Path("data/processed")
df_test = pl.read_parquet(data_dir / "steel_preprocessed_test.parquet")
y_true = df_test["Usage_kWh"].to_numpy()

# Recreate predictions (we need to save them in the main script, but for now load from MLflow)
# For now, let's create visualization templates

print("Creating Chronos-2 visualizations...")

# Create output directory
output_dir = Path("reports/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Create comparison bar plot
fig, ax = plt.subplots(figsize=(10, 6))

models = ["CUBIST\nBenchmark", "Target\n(-15%)", "XGBoost\nOptimized", "Chronos-2\nZero-Shot"]
rmse_norm = [0.2410, 0.2050, 0.3615, 1.1902]
colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]

bars = ax.bar(models, rmse_norm, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

# Add value labels on bars
for bar, value in zip(bars, rmse_norm, strict=False):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.02,
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

# Add target line
ax.axhline(y=0.2050, color="green", linestyle="--", linewidth=2, label="Target (0.2050)", alpha=0.7)
ax.axhline(y=0.2410, color="blue", linestyle="--", linewidth=2, label="CUBIST (0.2410)", alpha=0.7)

ax.set_ylabel("RMSE Normalized", fontsize=12, fontweight="bold")
ax.set_title(
    "Model Performance Comparison\nSteel Energy Consumption Forecasting",
    fontsize=14,
    fontweight="bold",
    pad=20,
)
ax.set_ylim(0, 1.4)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig(output_dir / "chronos2_model_comparison.png", dpi=300, bbox_inches="tight")
print(f"Saved: {output_dir / 'chronos2_model_comparison.png'}")

# Create metrics comparison table plot
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis("tight")
ax.axis("off")

metrics_data = [
    ["Metric", "XGBoost", "Chronos-2", "Winner"],
    ["RMSE (kWh)", "12.84", "42.29", "XGBoost (3.3x)"],
    ["MAE (kWh)", "3.53", "25.59", "XGBoost (7.3x)"],
    ["R²", "0.87", "-0.42", "XGBoost"],
    ["MAPE (%)", "31.5%", "77.0%", "XGBoost (2.4x)"],
]

table = ax.table(
    cellText=metrics_data,
    cellLoc="center",
    loc="center",
    colWidths=[0.25, 0.25, 0.25, 0.25],
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor("#3498db")
    table[(0, i)].set_text_props(weight="bold", color="white")

# Style data rows
for i in range(1, 5):
    for j in range(4):
        if j == 3:  # Winner column
            table[(i, j)].set_facecolor("#f39c12")
        else:
            table[(i, j)].set_facecolor("#ecf0f1" if i % 2 == 0 else "white")

plt.title("Detailed Metrics Comparison", fontsize=14, fontweight="bold", pad=20)
plt.savefig(output_dir / "chronos2_metrics_table.png", dpi=300, bbox_inches="tight")
print(f"Saved: {output_dir / 'chronos2_metrics_table.png'}")

print("\nVisualization creation completed!")
print(f"Chronos-2 RMSE: {results['metrics']['rmse']:.2f} kWh")
print("XGBoost RMSE: 12.84 kWh")
print("Performance gap: 3.3x worse")
