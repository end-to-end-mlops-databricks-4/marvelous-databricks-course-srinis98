# Databricks notebook source

# % pip install -e ..
# %restart_python

# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------
from loguru import logger
import yaml
import sys
from pyspark.sql import SparkSession
import pandas as pd

from age_of_abalone.config import ProjectConfig
from age_of_abalone.data_processor import DataProcessor

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

# Load the abelone dataset
spark = SparkSession.builder.getOrCreate()

filepath = "../data/abalone.csv"

# Load the data
df = pd.read_csv(filepath)


# COMMAND ----------
# Process the abelone dataset

data_processor = DataProcessor(df, config, spark)
data_processor.preprocess()

logger.info("Data preprocessing is completed.")

# COMMAND ----------

# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# COMMAND ----------
# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)

# Enable change data feed (only once!)
logger.info("Enable change data feed")
data_processor.enable_change_data_feed()
# COMMAND ----------