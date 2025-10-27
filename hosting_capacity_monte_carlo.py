"""This is the hosting capacity analysis main runner file."""

import logging
import sys

import pandas as pd
import yaml
from dacite import from_dict

from DssFileProcessor import DssFileProcessor
from DssCircuitSimulator import DssCircuitSimulator
from TransmissionDistribution.config import Config

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
    if not config.file_config.processed:
        dss_reader = DssFileProcessor(
            f"{config.file_config.path}/{config.file_config.name}"
        )
        dss_reader.process_master_files()
    hc = [0] * config.analysis_config.monte_carlo_sims
    for i in range(config.analysis_config.monte_carlo_sims):
        circuit_simulator = DssCircuitSimulator(config)
        circuit_simulator.start_engine()
        circuit_simulator.initialize_profiler_classes()
        hc[i] = circuit_simulator.maximize_solar_allocation_monte_carlo()
    data = {config.circuit_config.feeder_name: hc}
    df = pd.DataFrame(data)
    df.to_csv(
        f"{config.file_config.path}/monte_carlo_hc_{config.circuit_config.feeder_name}.csv"
    )
