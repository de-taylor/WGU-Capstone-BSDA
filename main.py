import hydra
import mlflow
import os
from omegaconf import DictConfig, OmegaConf

_steps = [
    "get_data",
    "clean_data",
    "check_data",
    "split_data",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]

# read in the hydra configuration
@hydra.main(config_name='config', version_base=None, config_path=".")
def go(config: DictConfig):
    # set up WANDB project and experiment variables
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # get the path at root of MLFlow project
    root_path = hydra.utils.get_original_cwd()

    # Determine which steps to execute
    steps_or = config['main']['steps']
    active_steps = steps_or.split(",") if steps_or != "all" else ''

    # run each step in turn
    if "get_data" in active_steps:
        # grab all data and load up to W&B
        _ = mlflow.run(
            uri=".",
            entry_point="get_data",
            parameters={
                "series_config_path": config["etl"]["series_config_path"],
                "api_base_url": config["etl"]["api_base_url"],
                "fred_api_key": config["etl"]["fred_api_key"],
                "output_path": config["etl"]["output_path"],
                "artifact_name": config["etl"]["artifact_name"],
                "artifact_type": "dataset"
            }
        )
    
    if "clean_data" in active_steps:
        # clean data, returning new artifact to W&B
        _ = mlflow.run(
            uri=".",
            entry_point="clean_data",
            parameters={
                "input_artifact": config["cleaning"]["input_artifact"],
                "output_path": config["cleaning"]["output_path"],
                "artifact_name": config["cleaning"]["artifact_name"],
                "artifact_type": "dataset"
            }
        )


if __name__ == "__main__":
    go()