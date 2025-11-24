"""The utilities module provides common utilities used throughout the Capstone project.

This project contains common functions that are vital to the overall success of the Capstone project. Rather than duplicating code, this sub-module was created in order to smooth over the development process, and provide standardization for common tasks.

The functions in this module include creating a uniform logger, robustly loading data from the FRED API, and committing data files to Weights&Biases as needed.
"""
# Imports
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
import sys
import time
from urllib3.util.retry import Retry


def new_logger(logger_name: str, rel_dir_path: str, max_log_size: int = 52736, backup_count=2, log_level=logging.DEBUG) -> logging.Logger:
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
        log_level (int):
            The minimum level of logging to use. Will take one of the following values:
                logging.DEBUG (10)
                logging.INFO (20)
                logging.WARNING (30)
                logging.ERROR (40)
                logging.CRITICAL (50)

    Returns:
        An object of type `logging.Logger` that is fully configured for the part of the program from which it was called.
    """
    logging.captureWarnings(True)
    # basic logger object, uses the required parameter logger_name to differentiate in the logs
    logger = logging.getLogger(logger_name)

    # a list of dictionary objects in form "level" and "message". Will be used to log any pre-logging setup warnings for later use.
    pre_log_messages = []

    if log_level not in [x * 10 for x in range(1,6)]:
        pre_log_messages.append(
            {
                "level": logging.WARNING,
                "message": f"Unable to use value of '{log_level}' for logging. Used {logging.DEBUG} instead."
            }
        )
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(log_level)

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

    if len(pre_log_messages) > 0:
        for message in pre_log_messages:
            logger.log(level=message["level"], msg=message["message"])

    return logger

# Utilities Module-Wide Logging
util_logger = new_logger(__name__, 'logs/utils')

# TODO: fully implement the custom FetchError class
class FetchError(Exception):
    # need to extend to be a custom, fully implemented Exception class
    pass

# How to format the metadata sidecars for robust implementation
CACHE_META_SUFFIX = ".meta.json"


def _make_session(retries=3, backoff=0.5) -> requests.Session:
    """Creates a Session object with a specific number of retries.

    The Session object has an HTTPAdapter mounted to it with a specific Retry() pool assigned. This Retry pool is configurable for the number of retries (default to 3) and the backoff to apply (default to 0.5). For this application, the only allowable method to retry is GET, and retries are forced when faced with error codes 429 (limit exceeded), 500, 502, 503, and 504 (server-side errors).

    The automated retries allow this Session to become much more stable and grants the ability to handle errors more gracefully, such as timeouts and API rate limits being exceeded.

    Args:
        retries (int):
            The number of times to retry the connection on this Session.
        backoff (float):
            The backoff factor to use when calculating the wait time between retry attempts.

    Returns:
        requests.Session object with the configured HTTPAdapter.
    """

    s = requests.Session()
    util_logger.debug(f"Creating a new Session object to reach the FRED API: max retries={retries}, backoff_factor={backoff}")
    r = Retry(
        total=retries,
        backoff_factor=backoff,
        backoff_jitter=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods={'GET'}
    )
    s.mount("https://", HTTPAdapter(max_retries=r))
    util_logger.debug("Mounted the custom https:// HTTPAdapter with the retry pool.")
    return s

def _cache_paths(dest: Path, series_id: str, extension: str) -> tuple[Path, Path]:
    """Returns the data file and metadata file paths given a specific destination and Series ID.

    Uses the global variable CACHE_META_SUFFIX in order to supply the metadata pathway.

    Args:
        dest (Path):
            A Path object that defines the given original data destination on disk.
        series_id (str):
            The FRED series indicator. See https://fred.stlouisfed.org/docs/api/fred/ for more information.
        extension (str):
            The extension that is used to save the data, e.g. 'csv', 'feather', 'parquet'.

    Returns:
        tuple[Path, Path]: A tuple of Path objects in the order data_path, meta_path.
    """

    data_path = dest / f"{series_id}.orig.{extension}"
    util_logger.debug(f"Setting up {data_path}")
    meta_path = dest / f"{series_id}.orig{CACHE_META_SUFFIX}"
    util_logger.debug(f"Setting up {meta_path}")

    return data_path, meta_path


def _load_metadata(meta_path: Path) -> dict:
    """Loads the metadata information from a given Path to a Python native object.

    Args:
        meta_path (Path):
            A Path object that points to a metadata file for a given economic data series.
    
    Returns:
        dict: a Python dictionary from the json.loads() function call, or an empty dictionary if the meta_path object does not exist.
    """

    if not meta_path.exists():
        util_logger.debug(f"Unable to find {meta_path.name}, metadata is empty.")
        return {}
    util_logger.debug(f"Found file {meta_path.name}, returning contents as dict.")
    return json.loads(meta_path.read_text())

def save_atomic(df: pd.DataFrame, data_path: Path, meta: dict, fmt: str = "parquet") -> Path:
    """Implements an atomic save design pattern that will prevent users from seeing partially written cache files.

    Performs this using the OS-specific .replace() function on a temporary file that will fully overwrite the old file, without leaving it partially completed for users who open the file in the middle of the write operation.

    The default for writing the cache file to disk is Parquet (https://parquet.apache.org/docs/file-format/). This is because this file format preserves type information while saving space. It can save space because it is a binary file format that is able to be efficiently compressed.

    Args:
        df (pd.DataFrame):
            The Pandas DataFrame to write to the cache.
        data_path (Path):
            The original path of the data file to overwrite.
        meta (dict):
            The metadata dictionary that will become a sidecar file to the data file
        fmt (str):
            The format to use to write the cache file to disk. Defaults to 'parquet'

    Returns:
        Path, the data path for logging in artifact trackers.
    """

    # create a temporary file that will replace the cached file
    tmp = data_path.with_suffix(data_path.suffix + ".tmp")
    util_logger.debug(f"Created temporary file {tmp.name}")
    # save in various formats depending on the supplied format
    match fmt:
        case "parquet":
            # preserves type information, not the index
            df.to_parquet(tmp)
        case "feather":
            # does not preserve type information, smaller file format for most simple use cases
            df.to_feather(tmp)
        case _:
            # does not preserve type information, plain text file format for simple use cases
            df.to_csv(tmp)
    
    util_logger.info(f"Saved content to {tmp.name} successfully, performing atomic swap.")
    
    # swap tmp with data_path using replace()
    tmp.replace(data_path)
    # take in the metadata object and write to the new meta_path location
    # this should overwrite that path, presumably
    if len(meta) > 0:
        meta_path = data_path.with_suffix(CACHE_META_SUFFIX)
        meta_path.write_text(json.dumps(meta))
        util_logger.info(f"Saved metadata to {meta_path.name}")
    util_logger.info(f"{data_path.name} is now the new version.")

    return data_path


def fetch_with_cache(series_id: str, request_uri: str, dest="data/orig", max_age_days=30, fmt="parquet", to_wandb: bool = False):
    """Loads a FRED data series from an API call or locally if data is not stale.

    This function is meant to reduce network bandwidth and calls to the FRED API by using local caching to the dest_path directory. It also uses a metadata sidecar file in order to track when the last actual update was from the API side. If the data wasn't actually updated from the API side, the cached data will be loaded instead.
    
    Local caching can be performed with Parquet, Feather, or CSV files as needed. Stick with one of these three formats.

    Args:
        series_id (str):
            The FRED series Identifier. See https://fred.stlouisfed.org/docs/api/fred/ for more information.
        request_uri (str):
            The fully formed FRED request URI, complete with all parameter values. See https://fred.stlouisfed.org/docs/api/fred/ for more information.
        logger (str):
            A fully formed logging.Logger object, like that returned from src.utilities.new_logger.
        dest_path (str):
            The destination path for the raw (unaltered) DataFrame, either as a Feather or CSV file (or both). Defaults to 'data/orig'
        max_days_old (int):
            The maximum number of days the local file can be cached before needing to be renewed. Renewal occurs from a call to the FRED API. Data is at most daily, but this parameter can be 0 to force an API call as needed. Defaults to 30.
        fmt (str):
            The file type used in the local data cache. Defaults to 'parquet', for the Parquet columnar file type. Options are 'parquet', 'feather', or 'csv'.
    
    Returns:
        A Pandas DataFrame containing at most 2 columns: a Date and series value column. Can potentially return an empty DataFrame if errors are encountered.

    Raises:
        HTTPError: Raised if there was an HTTPError from the requests response.
    """
    util_logger.info(f"Starting to fetch {series_id} from the FRED API...")
    # Step 0. create your dest_path and meta_path, load meta to variable
    dest = Path(dest)
    util_logger.debug(f"Checking for creation of {str(dest)}...")
    dest.mkdir(parents=True, exist_ok=True)  # create this directory if not exists, create parents as needed, OK if already exists.
    util_logger.debug(f"{str(dest)} has been verified: {dest.exists()}")

    data_path, meta_path = _cache_paths(dest, series_id, fmt)

    # load metadata file if exists, otherwise get back an empty dict
    meta = _load_metadata(meta_path)
    
    # Step 1. Check freshness of cached data, if exists
    if data_path.exists():
        age = (time.time() - data_path.stat().st_mtime) / 86400 # get age in days
        if age <= max_age_days:
            util_logger.debug(f"{data_path.name} age ({age}) is below threshold ({max_age_days}). Using cached version.")

            match fmt:
                case "parquet":
                    return pd.read_parquet(data_path)
                case "feather":
                    return pd.read_feather(data_path)
                case _:
                    try:
                        return pd.read_csv(data_path)
                    except Exception as err:
                        util_logger.error("Unable to read in CSV file.")
                        util_logger.error(err)

    # Step 2. Call API with conditional headers and using a session
    session = _make_session()
    headers = {}
    # form headers, forces a 304 if the ETag is the same or if the data hasn't been updated.
    # we should receive the ETag and Last-Modified fields from the HTTP response object to compare with
        # this is how the metadata sidecar is going to be loaded.
    if meta.get("etag"):
        headers["If-None-Match"] = meta["etag"]
    if meta.get("last_modified"):
        headers['If-Modified-Since'] = meta["last_modified"]

    resp = session.get(request_uri, timeout=(5,30), headers=headers)

    # if there has been no update since the cached file was last modified
    if resp.status_code == 304 and data_path.exists():
        util_logger.debug(f"Received 304 Not Modified; returning cached file {data_path.name}")
        match fmt:
            case "parquet":
                return pd.read_parquet(data_path)
            case "feather":
                return pd.read_feather(data_path)
            case _:
                try:
                    return pd.read_csv(data_path)
                except Exception as err:
                    util_logger.error("Unable to read in CSV file.")
                    util_logger.error(err)

    resp.raise_for_status()  # raises HTTP error if occurred

    payload = resp.json() # get JSON from FRED API call

    # validate payload keys
    observations = payload['observations']

    if observations is None:
        raise FetchError("Missing 'observations' in FRED API response")
    
    util_logger.info(f"Received {payload['count']} records from {series_id} call. Creating DataFrame...")
    
    series_df = pd.DataFrame(
        {
            'date': [obs['date'] for obs in observations],
            series_id: [obs['value'] for obs in observations]
        }
    )

    util_logger.debug(f"Performing type conversions: date --> np.datetime64[ns], {series_id} --> numeric (as appropriate).")
    series_df["date"] = pd.to_datetime(series_df["date"])
    series_df[series_id] = pd.to_numeric(series_df[series_id], errors="coerce")

    util_logger.info(f"The DataFrame for {series_id} was created with {series_df.shape[0]} rows and {series_df.shape[1]} columns.")

    # write atomically, save metadata
    meta = {
        "fetched_at": time.time(),
        "etag": resp.headers.get("ETag"),
        "last_modified": resp.headers.get("Last-Modified")
    }
    util_logger.debug(f"Saving metadata: {str(meta)}")

    save_atomic(series_df, data_path, meta, fmt)

    return series_df