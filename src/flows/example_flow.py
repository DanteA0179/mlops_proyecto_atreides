"""
Example Prefect Flow - Energy Optimization

Simple flow to demonstrate Prefect orchestration for US-004.
This flow will be expanded in US-005 for data ingestion.
"""

from datetime import datetime

from prefect import flow, task
from prefect.logging import get_run_logger


@task(name="log_start", retries=2)
def log_start() -> dict[str, str]:
    """Log the start of the flow."""
    logger = get_run_logger()
    timestamp = datetime.utcnow().isoformat()
    logger.info(f"Flow started at {timestamp}")
    return {"start_time": timestamp, "status": "started"}


@task(name="process_data", retries=2)
def process_data(start_info: dict[str, str]) -> dict[str, str]:
    """Simulate data processing."""
    logger = get_run_logger()
    logger.info("Processing data...")

    # Simulate some work
    result = {
        "start_time": start_info["start_time"],
        "processed_records": 100,
        "status": "processed",
    }

    logger.info(f"Processed {result['processed_records']} records")
    return result


@task(name="log_completion", retries=2)
def log_completion(result: dict[str, str]) -> dict[str, str]:
    """Log the completion of the flow."""
    logger = get_run_logger()
    end_time = datetime.utcnow().isoformat()
    logger.info(f"Flow completed at {end_time}")

    return {
        **result,
        "end_time": end_time,
        "status": "completed",
    }


@flow(name="example-energy-flow", log_prints=True)
def example_energy_flow() -> dict[str, str]:
    """
    Example Prefect flow for energy optimization.

    This demonstrates:
    - Task orchestration
    - Logging
    - Error handling with retries

    Returns:
        Dict with flow execution results
    """
    logger = get_run_logger()
    logger.info("Starting example energy optimization flow")

    # Execute tasks in sequence
    start_info = log_start()
    processed = process_data(start_info)
    result = log_completion(processed)

    logger.info("Flow execution completed successfully")
    return result


if __name__ == "__main__":
    # Run the flow locally for testing
    result = example_energy_flow()
    print(f"\nFlow Result: {result}")
