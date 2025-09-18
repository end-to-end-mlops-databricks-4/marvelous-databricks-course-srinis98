"""Dataloader fixture."""

import pytest
from loguru import logger

from term_deposit import PROJECT_DIR
from term_deposit.config import ProjectConfig, Tags


@pytest.fixture(scope="session")
def config() -> ProjectConfig:
    """Load and return the project configuration.

    This fixture reads the project configuration from a YAML file and returns a ProjectConfig object.

    :return: The loaded project configuration
    """
    config_file_path = (PROJECT_DIR / "project_config.yml").resolve()
    logger.info(f"Current config file path: {config_file_path.as_posix()}")
    config = ProjectConfig.from_yaml(config_file_path.as_posix())
    return config


@pytest.fixture(scope="session")
def tags() -> Tags:
    """Create and return a Tags instance for the test session.

    This fixture provides a Tags object with predefined values for git_sha, branch, and job_run_id.
    """
    return Tags(git_sha="wxyz", branch="test", job_run_id="9")
