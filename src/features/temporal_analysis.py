"""
Temporal Analysis Pipeline

CLI tool for time series analysis and feature engineering.
Implements STL decomposition, ACF/PACF analysis, and seasonality detection.

Usage:
    python -m src.features.temporal_analysis --help
    python -m src.features.temporal_analysis --dataset data/processed/steel_cleaned.parquet --output reports/temporal
    python -m src.features.temporal_analysis --load-type "Light Load" --period 24
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import polars as pl

from src.utils.time_series import (
    analyze_seasonality_by_group,
    extract_seasonal_pattern,
    perform_stl_decomposition,
    plot_acf_pacf,
    plot_seasonal_pattern,
    plot_seasonality_comparison,
    plot_stl_components,
)

# DuckDB imports removed - using Parquet directly

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TemporalAnalysisPipeline:
    """
    Pipeline for comprehensive time series analysis.

    Attributes
    ----------
    data_path : Path
        Path to DuckDB database
    output_path : Path
        Directory for output files
    df : pl.DataFrame
        Loaded dataset
    """

    def __init__(self, data_path: str, output_path: str, load_type: str | None = None):
        """
        Initialize the temporal analysis pipeline.

        Parameters
        ----------
        data_path : str
            Path to Parquet file
        output_path : str
            Directory for output files
        load_type : str, optional
            Filter by specific Load_Type
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.load_type = load_type
        self.df = None

        logger.info(f"Initialized pipeline with data: {self.data_path}")
        logger.info(f"Output directory: {self.output_path}")

    def load_data(self) -> None:
        """
        Load data from Parquet file.
        """
        logger.info("Loading data from Parquet...")

        # Read parquet file
        self.df = pl.read_parquet(self.data_path)

        # Select relevant columns and sort
        self.df = self.df.select(
            [
                "date",
                "Usage_kWh",
                "Lagging_Current_Reactive.Power_kVarh",
                "Leading_Current_Reactive_Power_kVarh",
                "CO2(tCO2)",
                "Lagging_Current_Power_Factor",
                "Leading_Current_Power_Factor",
                "NSM",
                "WeekStatus",
                "Day_of_week",
                "Load_Type",
            ]
        ).sort("date")

        # Filter by Load_Type if specified
        if self.load_type:
            original_rows = len(self.df)
            self.df = self.df.filter(pl.col("Load_Type") == self.load_type)
            logger.info(
                f"Filtered to Load_Type '{self.load_type}': {len(self.df)} rows (from {original_rows})"
            )

        logger.info(f"Loaded {len(self.df)} rows")
        logger.info(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")

    def run_stl_analysis(self, period: int = 24, seasonal: int = 7, robust: bool = True) -> dict:
        """
        Perform STL decomposition analysis.

        Parameters
        ----------
        period : int
            Seasonal period (default: 24 for hourly data)
        seasonal : int
            Length of seasonal smoother
        robust : bool
            Use robust fitting

        Returns
        -------
        dict
            Decomposition metadata
        """
        logger.info("Running STL decomposition...")

        decomp_df, metadata = perform_stl_decomposition(
            self.df,
            time_column="date",
            value_column="Usage_kWh",
            period=period,
            seasonal=seasonal,
            robust=robust,
        )

        # Plot components
        suffix = f"_{self.load_type.replace(' ', '_')}" if self.load_type else ""
        plot_stl_components(
            decomp_df,
            title=f'STL Decomposition{f" - {self.load_type}" if self.load_type else ""}',
            output_path=str(self.output_path / f"stl_decomposition{suffix}.png"),
        )

        logger.info(
            f"STL decomposition completed. Seasonal strength: {metadata['seasonal_strength']:.4f}"
        )

        return metadata

    def run_acf_pacf_analysis(self, nlags: int = 48) -> None:
        """
        Perform ACF/PACF analysis.

        Parameters
        ----------
        nlags : int
            Number of lags to analyze
        """
        logger.info("Running ACF/PACF analysis...")

        usage_series = self.df.select("Usage_kWh").to_series()

        suffix = f"_{self.load_type.replace(' ', '_')}" if self.load_type else ""
        plot_acf_pacf(
            usage_series,
            nlags=nlags,
            title=f'ACF/PACF Analysis{f" - {self.load_type}" if self.load_type else ""}',
            output_path=str(self.output_path / f"acf_pacf{suffix}.png"),
        )

        logger.info(f"ACF/PACF analysis completed with {nlags} lags")

    def run_seasonality_analysis(self, group_column: str = "Load_Type", period: int = 24) -> dict:
        """
        Analyze seasonality by group.

        Parameters
        ----------
        group_column : str
            Column to group by
        period : int
            Seasonal period

        Returns
        -------
        dict
            Seasonality results by group
        """
        logger.info(f"Analyzing seasonality by {group_column}...")

        results = analyze_seasonality_by_group(
            self.df,
            group_column=group_column,
            value_column="Usage_kWh",
            time_column="date",
            period=period,
        )

        # Plot comparison
        plot_seasonality_comparison(
            results,
            metric="seasonal_strength",
            title=f"Seasonal Strength by {group_column}",
            output_path=str(self.output_path / f"seasonality_{group_column.lower()}.png"),
        )

        # Log results
        for group, metadata in results.items():
            if "error" not in metadata:
                logger.info(
                    f"{group}: seasonal_strength={metadata['seasonal_strength']:.4f}, "
                    f"trend_strength={metadata['trend_strength']:.4f}"
                )

        return results

    def extract_and_plot_pattern(
        self, decomp_df: pd.DataFrame, period: int = 24, period_label: str = "Hour of Day"
    ) -> pd.DataFrame:
        """
        Extract and visualize seasonal pattern.

        Parameters
        ----------
        decomp_df : pd.DataFrame
            STL decomposition result
        period : int
            Seasonal period
        period_label : str
            Label for period axis

        Returns
        -------
        pd.DataFrame
            Seasonal pattern
        """
        logger.info("Extracting seasonal pattern...")

        pattern = extract_seasonal_pattern(decomp_df, period=period)

        suffix = f"_{self.load_type.replace(' ', '_')}" if self.load_type else ""
        plot_seasonal_pattern(
            pattern,
            period_label=period_label,
            title=f'Seasonal Pattern{f" - {self.load_type}" if self.load_type else ""}',
            output_path=str(self.output_path / f"seasonal_pattern{suffix}.png"),
        )

        # Identify peak and valley
        max_idx = pattern["seasonal_value"].idxmax()
        min_idx = pattern["seasonal_value"].idxmin()

        logger.info(
            f"Peak hour: {int(pattern.loc[max_idx, 'period_index'])}, "
            f"Valley hour: {int(pattern.loc[min_idx, 'period_index'])}"
        )

        return pattern

    def generate_report(self, results: dict) -> None:
        """
        Generate analysis report.

        Parameters
        ----------
        results : dict
            Analysis results
        """
        logger.info("Generating analysis report...")

        report_path = self.output_path / "temporal_analysis_report.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("TEMPORAL ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")

            # Dataset info
            f.write("1. DATASET INFORMATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Data path: {self.data_path}\n")
            f.write(f"Total rows: {len(self.df)}\n")
            f.write(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}\n")
            if self.load_type:
                f.write(f"Load Type filter: {self.load_type}\n")
            f.write("\n")

            # STL results
            if "stl_metadata" in results:
                f.write("2. STL DECOMPOSITION RESULTS\n")
                f.write("-" * 70 + "\n")
                for key, value in results["stl_metadata"].items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n")

            # Seasonality results
            if "seasonality_results" in results:
                f.write("3. SEASONALITY ANALYSIS BY GROUP\n")
                f.write("-" * 70 + "\n")
                for group, metadata in results["seasonality_results"].items():
                    if "error" not in metadata:
                        f.write(f"\n{group}:\n")
                        f.write(f"  Seasonal Strength: {metadata['seasonal_strength']:.4f}\n")
                        f.write(f"  Trend Strength: {metadata['trend_strength']:.4f}\n")
                        f.write(f"  Seasonal P2T: {metadata['seasonal_peak_to_trough']:.2f} kWh\n")
                f.write("\n")

            f.write("=" * 70 + "\n")
            f.write("REPORT GENERATED\n")
            f.write("=" * 70 + "\n")

        logger.info(f"Report saved to: {report_path}")

    def run_full_pipeline(self, period: int = 24, seasonal: int = 7, nlags: int = 48) -> dict:
        """
        Run complete temporal analysis pipeline.

        Parameters
        ----------
        period : int
            Seasonal period
        seasonal : int
            Seasonal smoother length
        nlags : int
            Number of ACF/PACF lags

        Returns
        -------
        dict
            Complete analysis results
        """
        logger.info("=" * 70)
        logger.info("STARTING FULL TEMPORAL ANALYSIS PIPELINE")
        logger.info("=" * 70)

        results = {}

        # Load data
        self.load_data()

        # STL analysis
        stl_metadata = self.run_stl_analysis(period=period, seasonal=seasonal)
        results["stl_metadata"] = stl_metadata

        # ACF/PACF analysis
        self.run_acf_pacf_analysis(nlags=nlags)

        # Seasonality analysis (skip if filtering by Load_Type)
        if not self.load_type:
            seasonality_results = self.run_seasonality_analysis(
                group_column="Load_Type", period=period
            )
            results["seasonality_results"] = seasonality_results

        # Generate report
        self.generate_report(results)

        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        return results


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Temporal Analysis Pipeline for Steel Energy Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/steel_cleaned.parquet",
        help="Path to Parquet file (default: data/processed/steel_cleaned.parquet)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="reports/temporal",
        help="Output directory for results (default: reports/temporal)",
    )

    parser.add_argument(
        "--load-type", type=str, default=None, help="Filter by specific Load_Type (optional)"
    )

    parser.add_argument(
        "--period",
        type=int,
        default=24,
        help="Seasonal period (default: 24 for hourly data with daily seasonality)",
    )

    parser.add_argument(
        "--seasonal", type=int, default=7, help="Seasonal smoother length (default: 7)"
    )

    parser.add_argument(
        "--nlags", type=int, default=48, help="Number of ACF/PACF lags (default: 48)"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize and run pipeline
    try:
        pipeline = TemporalAnalysisPipeline(
            data_path=args.dataset, output_path=args.output, load_type=args.load_type
        )

        pipeline.run_full_pipeline(period=args.period, seasonal=args.seasonal, nlags=args.nlags)

        logger.info(f"\n✓ Analysis complete! Results saved to: {args.output}")
        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
