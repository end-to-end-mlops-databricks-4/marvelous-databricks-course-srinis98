"""Spark Configuration module for local testing."""

from pydantic_settings import BaseSettings


class SparkConfig(BaseSettings):
    """Configuration class for application settings.

    This class loads settings from environment variables or a specified `.env` file
    and provides validation for the defined attributes.
    """

    master: str = "local[1]"
    app_name: str = "local_test"
    spark_executor_cores: str = "1"
    spark_executor_instances: str = "1"
    spark_sql_shuffle_partitions: str = "1"
    spark_driver_bindAddress: str = "127.0.0.1"


# Load configuration from environment variables
spark_config = SparkConfig()
