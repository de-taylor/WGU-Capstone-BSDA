"""PyTest Unit Testing for the src.utilities module."""

# PyTest
import pytest
# Python Standard Library Modules
import logging
from logging.handlers import RotatingFileHandler
import sys

# imports
from ..src.utilities import new_logger

# mock up external dependencies

# Unit Tests: src.utilities.new_logger
def test_new_logger_creates_logger(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    logger = new_logger("test_logger", str(log_dir))

    assert isinstance(logger, logging.Logger)  # verify Logger object
    assert logger.name == "test_logger"  # verify Logger object initialized correctly
    # verify handlers were instantiated correctly
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    assert any(isinstance(h, RotatingFileHandler) for h in logger.handlers)