"""This is the hosting capacity analysis main runner file."""

import logging
import sys

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
    data_processor = TimeSeriesDataProcessor(config=config)
    df = data_processor.get_processed_load_data()
    df.to_csv(
        f"scaled_load_data_processed/load_data_scaled_{config.circuit_config.feeder_name}.csv"
    )
