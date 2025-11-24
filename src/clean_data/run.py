"""The clean_data step implements the cleaning steps that became apparent during EDA.

These steps are relatively simple since this data is already relatively clean.
"""

# Imports
# Standard Library Modules
import argparse
from pathlib import Path

# Pip Modules
import pandas as pd
import wandb

# Custom Modules
from src.utilities import new_logger, save_atomic


# Start the logging object
logger = new_logger("etl.clean_data", 'logs/etl_clean')


def go(args):
    logger.info("Starting the WANDB run...")
    run = wandb.init(job_type="clean_data")

    # grab input artifact and log that the clean_data step is using it
    logger.info(f"Fetching WIP artifact: {args.input_artifact}")
    artifact_local_path = run.use_artifact(args.input_artifact).file(f"data/wip/")

    logger.debug(f"Attempting to read {args.input_artifact} to a DataFrame")
    orig_df = pd.read_parquet(artifact_local_path)
    logger.debug(f"Successfully read in {args.input_artifact} {orig_df.shape}: {orig_df.columns.values}")

    # restrict time scale
    wip_df = orig_df.query('index >= "2017-01-01" & index <= "2024-12-01"')
    logger.info(f"Restricted index (2017-01-01 to 2024-12-01).")
    logger.debug(f"DataFrame shape is now: {wip_df.shape}")
    
    # expand out annual values for 2024 from January through December
    logger.info("Filling in all 2024 values for annual columns: MEHOINUSA646N, MEPAINUSA646N, SPPOPGROWUSA, POPTOTUSA647NWDB")
    wip_df.loc['2024-02-01':'2024-12-01', ['MEHOINUSA646N']] = wip_df.loc['2024-01-01', 'MEHOINUSA646N']
    wip_df.loc['2024-02-01':'2024-12-01', ['MEPAINUSA646N']]  = wip_df.loc['2024-01-01', 'MEPAINUSA646N']
    wip_df.loc['2024-02-01':'2024-12-01', ['SPPOPGROWUSA']]  = wip_df.loc['2024-01-01', 'SPPOPGROWUSA']
    wip_df.loc['2024-02-01':'2024-12-01', ['POPTOTUSA647NWDB']]  = wip_df.loc['2024-01-01', 'POPTOTUSA647NWDB']

    logger.info("Making a clean copy of the clean DataFrame.")
    clean_df = wip_df.copy()

    # commit Parquet file to disk
    # make sure intermediate path exists
    clean_dest = Path(args.output_path)
    clean_dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created/verified destination path for clean DataFrame: {str(clean_dest.resolve())}")

    saved_path = save_atomic(clean_df, Path(f"{args.output_path}/{args.artifact_name}"), {})
    logger.info(f"Saved DataFrame to {saved_path}")

    # commit raw dataset now, cleaning will come later
    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description="Cleaned FRED data series data, ready for modeling.",
        metadata={"stage":"cleaned"}
    )

    artifact.add_file(str(Path(saved_path).resolve()))

    run.log_artifact(artifact)
    logger.info(f"Uploaded {args.artifact_name} to Weights & Biases successfully.")

    run.finish()

if __name__ == "__main__":
    # create main parser object
    parser = argparse.ArgumentParser(description="Download all FRED data series locally and commit to remote location")

    # add parser args
    parser.add_argument("--input_artifact", type=str, help="The specific artifact to use for cleaning")
    parser.add_argument("--output_path", type=str, help="The local directory where the original DataFrame should be kept")
    parser.add_argument("--artifact_name", type=str, help="Name for the output artifact")
    parser.add_argument("--artifact_type", type=str, help="Type of the output artifact. This will be used to categorize the artifact in the W&B interface")

    args = parser.parse_args()

    go(args)