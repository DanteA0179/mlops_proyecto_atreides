"""Quick test to verify Dagster jobs load correctly."""

from src.dagster_pipeline.definitions import defs

print("✓ Dagster definitions loaded successfully")
print(f"✓ Jobs registered: {len(defs.jobs)}")

for job in defs.jobs:
    print(f"  - {job.name}")

print("\nAll jobs loaded without errors!")
