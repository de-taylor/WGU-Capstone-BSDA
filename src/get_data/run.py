"""The get_data step parses all required FRED Data Series names, and returns a DataFrame with all Series combined, including the Series of interest (the response variable.)
"""

# Imports
# Standard Library Modules
import argparse
import json
import os
from pathlib import Path

# Pip Modules
import pandas as pd
from requests import HTTPError
import wandb

# Custom Modules
from src.utilities import new_logger, fetch_with_cache, save_atomic


# Start the logging object
logger = new_logger("etl.get_data", 'logs/etl_clean')


def go(args):
    logger.info("Starting the WANDB run...")
    run = wandb.init(job_type="get_data")

    logger.info(f"Looking for FRED series in {args.series_config_path}")
    abs_fred_config = Path(args.series_config_path).resolve()

    if os.path.exists(abs_fred_config):
        with open(abs_fred_config, 'r') as json_fp:
            logger.info(f"Opened {abs_fred_config}")
            fred_series = json.load(json_fp)
            logger.info(f"Loaded {abs_fred_config}, moving to data fetch operation.")
    else:
        fred_series = None
        logger.error(f"Could not find {abs_fred_config}, make sure it exists before continuing.")

    if fred_series is not None:
        # fred_series MUST first exist before trying to grab things from it
        # start with monthly frequency
        monthly_dfs = []
        for series in fred_series['monthly_series']:
            request_uri = f"{args.api_base_url}?series_id={series}&api_key={args.fred_api_key}&file_type=json"
            logger.info(f"Starting fetch process for {series}...")
            try:
                monthly_dfs.append(fetch_with_cache(series_id=series, request_uri=request_uri, dest="data/orig"))
                logger.debug(f"Setting the {series} DataFrame's index to `date`...")
                monthly_dfs[-1] = monthly_dfs[-1].set_index('date', drop=True).sort_index()
            except HTTPError as err:
                logger.error(err)
            logger.info(f"Fetch process for {series} ({monthly_dfs[-1].shape}) is complete.")

        # high frequency
        hf_data_frames = []
        for series in fred_series['hf_series']:
            request_uri = f"{args.api_base_url}?series_id={series}&api_key={args.fred_api_key}&file_type=json&frequency=m&aggregation_method=eop"
            logger.info(f"Starting fetch process for {series}...")
            try:
                hf_data_frames.append(fetch_with_cache(series_id=series, request_uri=request_uri, dest="data/orig"))
                logger.debug(f"Setting the {series} DataFrame's index to `date`...")
                hf_data_frames[-1] = hf_data_frames[-1].set_index('date', drop=True).sort_index()
            except HTTPError as err:
                logger.error(err)
            logger.info(f"Fetch process for {series} ({hf_data_frames[-1].shape}) is complete.")

        # low frequency
        lf_data_frames = []
        for series in fred_series['lf_series']:
            request_uri = f"{args.api_base_url}?series_id={series}&api_key={args.fred_api_key}&file_type=json"
            logger.info(f"Starting fetch process for {series}...")
            try:
                tmp_df = fetch_with_cache(series_id=series, request_uri=request_uri, dest="data/orig")
                logger.debug(f"Setting the {series} DataFrame's index to `date`...")
                tmp_df = tmp_df.set_index('date', drop=True).sort_index()
                # resample to monthly frequency
                tmp_monthly = tmp_df.resample('MS').asfreq()

                # distribute quarterly values to monthly, naively assuming it was the same value in all three months
                tmp_monthly[series] = tmp_df.resample('MS').ffill()[series]

                # add monthly df to dataframes list
                lf_data_frames.append(tmp_monthly)
            except HTTPError as err:
                logger.error(err)
            logger.info(f"Fetch process for {series} ({lf_data_frames[-1].shape}) is complete.")

        logger.info("Combining intermediate DataFrames into a single DataFrame...")
        # pull into intermediate dataframes
        ms_df = pd.concat(monthly_dfs, axis=1, join='outer', verify_integrity=True)
        logger.debug(f"Month-Start DataFrame Created ({ms_df.shape}): {ms_df.columns.values}")
        hf_df = pd.concat(hf_data_frames, axis=1, join='outer', verify_integrity=True)
        logger.debug(f"High-Frequency DataFrame Created ({hf_df.shape}): {hf_df.columns.values}")
        lf_df = pd.concat(lf_data_frames, axis=1, join='outer', verify_integrity=True)
        logger.debug(f"Low-Frequency DataFrame Created ({lf_df.shape}): {lf_df.columns.values}")

        # pull into final dataframe
        comb_df = pd.concat([ms_df, hf_df, lf_df], axis=1, join='outer')
        logger.info(f"Combined DataFrame created ({comb_df.shape}) and ready to upload.")

        # commit Parquet file to disk
        # make sure intermediate path exists
        wip_dest = Path(args.output_path)
        wip_dest.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created/verified destination path for clean DataFrame: {str(wip_dest.resolve())}")

        saved_path = save_atomic(comb_df, Path(f"{args.output_path}/{args.artifact_name}"), {})
        logger.info(f"Saved DataFrame to {saved_path}")

        # commit raw dataset now, cleaning will come later
        artifact = wandb.Artifact(
            args.artifact_name,
            args.artifact_type,
            "Combined FRED data series, monthly, not seasonally adjusted.",
            metadata={"stage":"raw"}
        )

        artifact.add_file(str(Path(saved_path).resolve()))

        run.log_artifact(artifact)
        logger.info(f"Uploaded {args.artifact_name} to Weights & Biases successfully.")

        run.finish()

if __name__ == "__main__":
    # create main parser object
    parser = argparse.ArgumentParser(description="Download all FRED data series locally and commit to remote location")

    # add parser args
    parser.add_argument("--series_config_path", type=str, help="The string Path of the FRED series names, by series frequency.")
    parser.add_argument("--api_base_url", type=str, help="The base FRED API URL to use to gather the economic series data")
    parser.add_argument("--fred_api_key", type=str, help="The secret FRED API key to use to gather data, when the cache has expired")
    parser.add_argument("--output_path", type=str, help="The local directory where the original DataFrame should be kept")
    parser.add_argument("--artifact_name", type=str, help="Name for the output artifact")
    parser.add_argument("--artifact_type", type=str, help="Type of the output artifact. This will be used to categorize the artifact in the W&B interface")

    args = parser.parse_args()

    go(args)