# Databricks notebook source

# % pip install -e ..
# %restart_python

# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------
from loguru import logger
import yaml
import pandas as pd

from term_deposit.config import ProjectConfig

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
# Load the house prices dataset
import pandas as pd
filepath = "../data/data.csv"

# Load the data
df = pd.read_csv(filepath)
