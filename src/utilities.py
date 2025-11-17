"""The utilities module provides common utilities used throughout the Capstone project.

This project contains common functions that are vital to the overall success of the Capstone project. Rather than duplicating code, this sub-module was created in order to smooth over the development process, and provide standardization for common tasks.

The functions in this module include creating a uniform logger...
"""
# Imports
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys


def new_logger(logger_name: str, rel_dir_path: str, max_log_size: int = 52736, backup_count=2) -> logging.Logger:
    """Standardizes logs across the project for easier troubleshooting.

    The project logger utilizes two handlers: a RotatingFileHandler and a StreamHandler. The RotatingFileHandler is configurable, allowing for logs of various sizes and different numbers of backup files in the logging directory.

    Incorporates redirection of stderr and stdout to the logger.

    Args:
        logger_name (str):
            The part of the program being logged. Required.
        rel_dir_path (str):
            The relative path to the logging directory from that part of the program. Required.
        max_log_size (int):
            The maximum size of the log before rolling over, in bytes. Defaults to 52736.
        backup_count (int):
            The number of backups to keep for each log. Defaults to 2.

    Returns:
        An object of type `logging.Logger` that is fully configured for the part of the program from which it was called.
    """
    logging.captureWarnings(True)
    # basic logger object, uses the required parameter logger_name to differentiate in the logs
    logger = logging.getLogger(logger_name)
    # extremely detailed logs for data science project
    logger.setLevel(logging.DEBUG)

    # Creating Handlers
    # check to make sure directory exists for the rotating file log
    os.makedirs(Path(rel_dir_path), exist_ok=True)

    rfh = RotatingFileHandler(f'{rel_dir_path}/{logger_name}.log',mode='a',maxBytes=max_log_size,backupCount=backup_count,encoding='utf-8')
    rfh.setLevel(logging.DEBUG)
    # stream being the console output
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)

    # Creating Formatter
    # common formatter for all logs in project
    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S %z")
    # add formatter to both handlers
    rfh.setFormatter(fmt)
    ch.setFormatter(fmt)

    # add both handlers to main logger, if they don't already exist
    if not logger.handlers:
        logger.addHandler(rfh)
        logger.addHandler(ch)

    return logger