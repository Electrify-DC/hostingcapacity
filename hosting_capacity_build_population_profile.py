"""This is the hosting capacity analysis main runner file."""

import logging
import os
import sys

import pandas as pd
import yaml
from dacite import from_dict

from DssFileProcessor import DssFileProcessor
from DssCircuitSimulator import DssCircuitSimulator
from TransmissionDistribution.config import Config
from TransmissionDistribution.time_series_data_processor import TimeSeriesDataProcessor

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s: %(levelname)s: %(message)s",
)


def load_config() -> Config:
    """
    Loads locally stored config to be used in simulation.
    :return: config containing parameters
    """
    with open("simulation_config.yaml", "r") as file:
        data = yaml.safe_load(file)
        logging.info("Loading local config")
    return from_dict(data_class=Config, data=data)


if __name__ == "__main__":
    config = load_config()
    df_list = []
    folder_path = config.file_config.path

    # Loop through all CSV files in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):  # Ensure it's a CSV file
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)  # Read CSV
            df_list.append(df)  # Add to list

    # Concatenate all DataFrames into one
    population_ami_df = pd.concat(df_list, ignore_index=True)
    # save population_ami_df
    population_ami_df.to_csv(f"scaled_load_data_processed/all_samples_ami_data.csv")
