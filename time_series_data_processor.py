import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from TransmissionDistribution.config import Config
from TransmissionDistribution.constants import data_constants, solar_constants


class TimeSeriesDataProcessor:
    """This class processes time data series and scales with appropriate customer meter to meter mapping"""

    def __init__(self, config: Config):
        self.path = config.file_config.path
        self.data_config = config.data_config
        self.feeder_name = config.circuit_config.feeder_name
        self.feeder_to_home_df = None
        self.load_data_df = None
        self.load_data_scaled_df = None
        self.solar_data_df = None
        self.load_cols = None
        self.total_load_data_df = None

    def process_customer_mapping(self):
        if "csv" in self.data_config.feeder_to_home_mapping:
            df = pd.read_csv(f"{self.path}/{self.data_config.feeder_to_home_mapping}")
        elif "xlsx" in self.data_config.feeder_to_home_mapping:
            df = pd.read_excel(f"{self.path}/{self.data_config.feeder_to_home_mapping}")
        else:
            raise TypeError(
                f"Invalid File Format Provided for {self.data_config.feeder_to_home_mapping}"
            )
        df = df.rename(columns=str.upper)
        self.feeder_to_home_df = df[df[data_constants.FEEDER_ID] == self.feeder_name]

    def process_load_mapping(self):
        df = pd.read_csv(f"{self.path}/{self.data_config.customer_load_data}")
        df = df.rename(columns=str.upper)
        df.columns = df.columns.str.replace("-", "", regex=True)
        # phase columns
        df[data_constants.PHASES] = (
            (df[data_constants.A_PHASE] != 0)
            & (df[data_constants.B_PHASE] != 0)
            & (df[data_constants.C_PHASE] != 0)
        )
        df[data_constants.PHASES] = df[data_constants.PHASES].replace(
            {True: 3, False: 1}
        )
        self.load_data_df = df.rename(
            columns={data_constants.FACILITY_ID: data_constants.TRANSFORMER_ID}
        )
        self.load_cols = self.load_data_df.filter(
            regex=data_constants.TIME_DATA_COL
        ).columns

    def process_customer_population_load_data(self):
        df = pd.read_csv(f"{self.path}/{self.data_config.customer_load_population}")
        self.total_load_data_df = df

    def process_missing_load_data(self):
        df = self.load_data_df.copy()
        for phase, group in df.groupby(data_constants.PHASES):
            # Iterate only over the specified columns
            for col in self.load_cols:
                missing_indices = group[col].isna()
                if missing_indices.any():
                    mu, std = norm.fit(group[col].dropna())
                    # Generate random Gaussian samples for the missing values in this group
                    samples = np.random.normal(mu, std, missing_indices.sum())
                    # Fill missing values in the DataFrame for the current group and column
                    df.loc[group.index[missing_indices], col] = np.abs(samples)
        self.load_data_df[self.load_cols] = df[self.load_cols]

    def scale_load_according_to_number_customers(self):
        num_customers_per_aggregate_load = (
            self.feeder_to_home_df.groupby(data_constants.TRANSFORMER_ID)
            .size()
            .reset_index(name=data_constants.NUM_CUSTOMERS)
        )
        self.load_data_df = self.load_data_df.merge(
            num_customers_per_aggregate_load,
            on=data_constants.TRANSFORMER_ID,
            how="left",
        )
        # drop loads with nan values
        self.load_data_df = self.load_data_df.dropna(
            subset=[data_constants.NUM_CUSTOMERS]
        )
        # scale only load data and avoid missing data
        self.load_data_df[self.load_cols] = self.load_data_df[self.load_cols].div(
            self.load_data_df[data_constants.NUM_CUSTOMERS], axis=0
        )

    def get_processed_load_data(self) -> pd.DataFrame:
        """
        Gets processed load data from customer mapping and load data.
        :return: processed load dataframe
        """
        self.process_customer_mapping()
        self.process_load_mapping()
        self.scale_load_according_to_number_customers()
        self.process_missing_load_data()
        if self.data_config.customer_load_population is not None:
            self.process_customer_population_load_data()
        return self.load_data_df

    def generate_random_load(self, load_name: str, phase: int = 1) -> np.ndarray:
        """
        Generates a random load based on phase.
        :param load_name:
        :param phase:
        :return:
        """
        df = self.load_data_df.copy()
        df = df.dropna()
        # filter by phase
        if "a" in load_name and "b" in load_name and "c" in load_name:
            phase = data_constants.THREE_PHASE
        df_phase = df[df[data_constants.PHASES] == phase]
        sample = {}
        for col in self.load_cols:
            mu, std = norm.fit(df_phase[col])
            # generate gaussian sample
            sample[col] = np.abs(np.random.normal(mu, std, size=1))
        return np.concatenate(list(sample.values()))

    def generate_random_load_population(
        self, load_name: str, phase: int = 1
    ) -> np.ndarray:
        """
        Generates a random load based on phase.
        :param load_name:
        :param phase:
        :return:
        """
        df = self.total_load_data_df.copy()
        df = df.dropna()
        # filter by phase
        if "a" in load_name and "b" in load_name and "c" in load_name:
            phase = data_constants.THREE_PHASE
        df_phase = df[df[data_constants.PHASES] == phase]
        sample = {}
        for col in self.load_cols:
            mu, std = norm.fit(df_phase[col])
            # generate gaussian sample
            sample[col] = np.abs(np.random.normal(mu, std, size=1))
        return np.concatenate(list(sample.values()))

    def process_peak_solar_data(self):
        """
        Process solar peak profile data. Format specific for data from NOAA.
        :return:
        """
        if "csv" in self.data_config.solar_peak_profile:
            pv_df = pd.read_csv(f"{self.path}/{self.data_config.solar_peak_profile}")
        elif "xlsx" in self.data_config.solar_peak_profile:
            pv_df = pd.read_excel(f"{self.path}/{self.data_config.solar_peak_profile}")
        else:
            raise TypeError(
                f"Invalid File Format Provided for {self.data_config.feeder_to_home_mapping}"
            )
        # dataframe is one column need to break out
        pv_df[["Date", "Time", "AirMass", "SolarZen", "Elev", "Azim"]] = pv_df[
            pv_df.columns[0]
        ].str.split(expand=True)
        pv_df_elevation = pv_df["Elev"]
        radiation_df = self.calc_solar_radiation(pv_df_elevation)
        # data includes 11 pm from night before and 12 am from next night
        self.solar_data_df = radiation_df["Scaled_Radiation"][1:25]

    def get_processed_peak_solar_data(self):
        """
        Process solar peak profile data. Format specific for data from NOAA.
        :return:
        """
        self.process_peak_solar_data()
        return self.solar_data_df

    def calc_solar_radiation(
        self, elevation: pd.DataFrame, cloud_cover_percentage: float = 0.0
    ) -> pd.DataFrame:
        # data starts one time stamp before 12:00
        df = pd.DataFrame()
        time_stamps = len(elevation)
        solar_radiation = [0.0]
        scaled_solar_radiation = [0.0]
        for idx, elev in enumerate(elevation):
            if idx == time_stamps - 1:
                break
            phi_prev = math.radians(float(elev))
            phi_now = math.radians(float(elevation[idx + 1]))
            phi = (phi_prev + phi_now) / 2
            R_knot = (
                solar_constants.SOLAR_RADIATION_FACTOR * np.sin(phi)
                - solar_constants.SOLAR_RADIATION_OFFSET
            )
            solar_rad = R_knot * (
                1
                - solar_constants.CLOUD_COVER_FACTOR
                * math.pow(
                    cloud_cover_percentage, solar_constants.CLOUD_COVER_MULTIPLIER
                )
            )
            solar_rad = max(solar_rad, 0.0)
            solar_radiation.append(solar_rad)
            scaled_solar_radiation.append(
                solar_rad / solar_constants.MAX_SOLAR_RADIATION
            )
        df["Radiation"] = solar_radiation
        df["Scaled_Radiation"] = scaled_solar_radiation
        return df
