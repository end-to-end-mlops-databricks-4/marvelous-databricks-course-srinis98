# Databricks notebook source

# % pip install -e ..
# %restart_python

# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------
import pandas as pd
import yaml
from loguru import logger
from pyspark.sql import SparkSession

from term_deposit.config import ProjectConfig
from term_deposit.data_processor import DataProcessor

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
# Load the term deposit dataset
filepath = "../data/data.csv"

# Load the data
df = pd.read_csv(filepath)

# COMMAND ----------
# Process the term deposit data
spark = SparkSession.builder.getOrCreate()
data_processor = DataProcessor(df, config, spark)

data_processor.preprocess()

logger.info("Data preprocessing is completed.")

# COMMAND ----------

# Split the data
X_train, X_test = data_processor.split_data()
logger.info(f"Training set shape: {X_train.shape}")
logger.info(f"Test set shape: {X_test.shape}")

# COMMAND ----------
# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)

# Enable change data feed (only once!)
logger.info("Enable change data feed")
data_processor.enable_change_data_feed()
