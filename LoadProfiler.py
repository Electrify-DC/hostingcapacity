import math
import random
from typing import Optional

import numpy as np
import pandas as pd
from dss.ICircuit import ICircuit

from TransmissionDistribution.constants import data_constants
from TransmissionDistribution.config import Config
from TransmissionDistribution.time_series_data_processor import TimeSeriesDataProcessor


class LoadProfiler:
    """
    This class contains methods to generate and manipulate load profiles.
    """

    def __init__(
        self,
        dss_circuit: ICircuit,
        config: Config,
        num_days: int,
        time_steps: int,
        load_names: list[str],
        time_series_data_processor: Optional[TimeSeriesDataProcessor] = None,
        load_data: Optional[pd.DataFrame] = None,
        load_cols: Optional[list] = None,
    ):
        self.dss_circuit = dss_circuit
        self.time_series_data_processor = time_series_data_processor
        self.load_names = load_names
        self.analysis_config = config.analysis_config
        self.config = config
        self.data_config = config.data_config
        self.num_days = num_days
        self.time_steps = time_steps
        self.load_data_df = load_data
        self.load_cols = load_cols

    @property
    def load_ami_data(self):
        """
        Load data from file path or return passed in dataframe.
        """
        if self.load_data_df is not None:
            return self.load_data_df
        else:
            ami_data_name = self.data_config.customer_load_data
            path = self.config.file_config.path
            file = f"{path}/{ami_data_name}"
            ami_data = None
            if "npy" in ami_data_name:
                ami_data = np.load(file)
                ami_data = ami_data[:, self.analysis_config.day_of_year, :]
            elif "csv" in ami_data_name:
                ami_data = pd.read_csv(file)
            return ami_data

    @property
    def num_profiles(self):
        return len(self.load_ami_data)

    def generate_initial_load_profile(self, load_idx: int, load_name: str):
        """
        Generate initial load profile based on frequency of data and length of simulation
        :param load_idx: index of load
        :param load_name: name of load from load file
        :return: initial load profile array
        """
        load_profile = np.zeros(self.time_steps)
        self.dss_circuit.LoadShapes.New(f"customer_profile_{load_idx}")
        self.dss_circuit.LoadShapes.Npts = self.time_steps
        self.dss_circuit.LoadShapes.MinInterval = self.analysis_config.solve_frequency
        self.dss_circuit.LoadShapes.UseActual = True
        self.dss_circuit.LoadShapes.Qmult = (
            load_profile
            * math.tan(math.acos(self.analysis_config.initial_power_factor))
        ).tolist()
        self.set_load_profile_for_active_element(
            load_idx=load_idx, load_name=load_name, load_profile=load_profile
        )

    def generate_mapped_load_profile(self, load_name: str) -> np.ndarray:
        """
        Generates a mapped load profile and generates a random one if missing.
        :return: load profile
        """
        try:
            load_profile = self.mapping_load_profile_to_load(load_name=load_name)
            df = load_profile.copy()
            df.reset_index(inplace=True)
            num_customers_on_load = df[data_constants.PHASES].iloc[0]
            profile = df[self.load_cols].iloc[0].to_numpy() * num_customers_on_load
        except Exception as e:
            # load profile empty
            profile = self.time_series_data_processor.generate_random_load(
                load_name=load_name
            )
        return profile

    def generate_random_load_profile(self, load_name: str) -> np.ndarray:
        """
        Generates a mapped load profile and generates a random one if missing.
        :return: load profile
        """
        try:
            df = self.mapping_load_profile_to_load(load_name=load_name)
            df.reset_index(inplace=True)
            num_customers_on_load = df[data_constants.PHASES].iloc[0]
        except Exception as e:
            num_customers_on_load = 1
        load_profile = self.time_series_data_processor.generate_random_load_population(
            load_name=load_name
        )
        profile = load_profile * num_customers_on_load
        return profile

    def generate_mapped_load_profile_during_sim(
        self, load_idx: int, load_name: str, power_factor_list: list
    ) -> tuple:
        """
        Generates a load profile for use during the simulation.
        :param load_idx: index of load
        :param load_name: name of load from load file
        :param power_factor_list: power factor list used to produce reactive power profiles
        :return: load_profile, reactive_profile, random_profile
        """
        random.seed(42)
        load_profile = self.generate_mapped_load_profile(load_name=load_name)
        reactive_profile = []
        for j in range(self.time_steps):
            reactive_profile.append(
                (load_profile[j]) * math.tan(math.acos(power_factor_list[load_idx][j]))
            )
        reactive_profile = np.array(reactive_profile)
        self.dss_circuit.LoadShapes.Qmult = reactive_profile.tolist()
        self.set_load_profile_for_active_element(
            load_idx=load_idx, load_name=load_name, load_profile=load_profile
        )
        return load_profile, reactive_profile

    def generate_random_load_profile_during_sim(
        self, load_idx: int, load_name: str, power_factor_list: list
    ) -> tuple:
        """
        Generates a load profile for use during the simulation.
        :param load_idx: index of load
        :param load_name: name of load from load file
        :param power_factor_list: power factor list used to produce reactive power profiles
        :return: load_profile, reactive_profile, random_profile
        """
        load_profile = self.generate_random_load_profile(load_name=load_name)
        reactive_profile = []
        for j in range(self.time_steps):
            reactive_profile.append(
                (load_profile[j]) * math.tan(math.acos(power_factor_list[load_idx][j]))
            )
        reactive_profile = np.array(reactive_profile)
        self.dss_circuit.LoadShapes.Qmult = reactive_profile.tolist()
        self.set_load_profile_for_active_element(
            load_idx=load_idx, load_name=load_name, load_profile=load_profile
        )
        return load_profile, reactive_profile

    def set_load_profile_for_active_element(
        self, load_idx: int, load_name: str, load_profile: np.ndarray
    ) -> None:
        """
        Set the load profile for a given load that is set as an active element.
        :param load_idx: index of load
        :param load_name: name of load from load file
        :param load_profile: power profile of load usage over some time steps
        """
        profile_name = f"customer_profile_{load_idx}"
        self.dss_circuit.LoadShapes.Name = profile_name
        self.dss_circuit.LoadShapes.Pmult = load_profile.tolist()
        self.dss_circuit.SetActiveElement(f"load.{load_name}")
        self.dss_circuit.ActiveElement.Properties("daily").Val = profile_name

    def generate_power_factor(self) -> list:
        """
        Generates a power factor list over number of solve time steps and load.
        :return: power factor list of randomly generated power factors between min and max
        """
        power_factor_list = []
        customers = len(self.load_names)
        for i in range(self.num_days * customers):
            pf_list = []
            for j in range(self.time_steps):
                random_pf = random.uniform(
                    self.analysis_config.min_power_factor,
                    self.analysis_config.max_power_factor,
                )
                pf_list.append(random_pf)
            power_factor_list.append(pf_list)
        return power_factor_list

    def mapping_load_profile_to_load(self, load_name: str) -> pd.DataFrame:
        """
        Matches the load profile to the load based on the name.
        :return: matching load profile
        """
        prefix, id_1, id_2, id_phase = load_name.split("-")
        mapped_load_profile = self.load_ami_data[
            self.load_ami_data[data_constants.TRANSFORMER_ID].apply(
                lambda x: any(part in x for part in [id_1, id_2])
            )
        ]
        return mapped_load_profile
