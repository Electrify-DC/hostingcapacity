import logging
import random
from typing import Any, Union

import dss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dss import plot as dss_plt

from TransmissionDistribution.constants import data_constants
from TransmissionDistribution.LoadProfiler import LoadProfiler
from TransmissionDistribution.PlottingSimulation import PlottingSimulation
from TransmissionDistribution.SolarProfiler import SolarProfiler
from TransmissionDistribution.config import Config
from TransmissionDistribution.time_series_data_processor import TimeSeriesDataProcessor


class DssCircuitSimulator:
    def __init__(self, config: Config):
        self.config = config
        self.master = f"{config.file_config.path}/{config.file_config.name}"
        # init basic parameters for analysis
        self.frequency = config.circuit_config.frequency
        self.num_iterations = config.analysis_config.max_iterations
        self.num_phases = config.circuit_config.phases
        self.solve_frequency = self.config.analysis_config.solve_frequency
        self.selected_day = self.config.analysis_config.day_of_year
        # init dss related variables
        self.dss_engine = dss.DSS
        self.dss_text = self.dss_engine.Text
        self.dss_circuit = self.dss_engine.ActiveCircuit
        self.dss_circuit_solution = self.dss_engine.ActiveCircuit.Solution
        self.dss_circuit_ctrl_queue = self.dss_engine.ActiveCircuit.CtrlQueue
        # init objects that are other classes
        self.logger = logging.getLogger()
        self.plotter = PlottingSimulation()
        self.time_series_data_processor = TimeSeriesDataProcessor(config=config)
        self.solar_profiler = None
        self.load_profiler = None
        self.load_names = None
        self.num_customers = None
        self.load_cols = None
        self.pv_cols = None

    @property
    def add_pv(self) -> bool:
        """
        Is true if pv config is present - meaning add pv to simulation.
        """
        if getattr(self.config, "pv_config"):
            return True
        return False

    @property
    def num_days(self):
        return self.config.analysis_config.num_of_days

    @property
    def time_steps(self):
        """The time steps for the whole simulation"""
        num_pts = pd.Timedelta(hours=1) // pd.Timedelta(
            minutes=self.config.analysis_config.solve_frequency
        )
        return num_pts * self.config.analysis_config.simulation_length

    @property
    def report_thermal_overload(self):
        return self.config.analysis_config.report_thermal_overload

    @property
    def report_voltage_exception(self):
        return self.config.analysis_config.report_voltage_exception

    def dss_text_command(
        self,
        action: Union[list[str], str],
        variable: Union[list[Any], Any, None] = None,
    ) -> None:
        """
        Performs dss text commands by parsing actions and variables.
        :param action: Action(s) to perform. Can be a single string or a list of strings.
        :param variable: Optional variable(s) used for setting, can be any type.
        """
        if not isinstance(action, list):
            action = [action]  # Convert single action to list for uniform processing
        if variable is None:
            variable = []
        elif not isinstance(variable, list):
            variable = [variable]  # Convert single variable to list
        command_parts = []
        for a, v in zip(action, variable):
            command_parts.append(f"{a} {v}")

        # If one list is longer, append remaining elements
        longer_list = action if len(action) > len(variable) else variable
        command_parts.extend(longer_list[len(command_parts) :])

        # Construct and execute the command
        command = " ".join(map(str, command_parts)).strip()
        self.dss_text.Command = command

    def start_engine(self) -> None:
        """
        Starts the dss engine and performs the initial compilation and base settings.
        """
        self.logger.info("Starting dss engine and activating circuit")
        self.dss_engine.AllowForms = 0
        self.dss_text_command(action="Clear")
        self.dss_text_command(action="Compile", variable=self.master)
        self.dss_text_command(action="Calcvoltagebases")
        self.dss_text_command(
            action="Set DefaultBaseFrequency=", variable=self.frequency
        )
        self.dss_text_command(
            action="Set maxcontroliter=", variable=self.num_iterations
        )

    def plot_single_line_diagram_with_local_hosting_capacity(self) -> None:
        """
        Plot single line diagram of circuit with various elements.
        """
        dss_plt.enable()
        self.dss_text_command(action="Buscoords", variable="Buscoords.dss")
        self.dss_text_command(
            action="plot General quantity=1 dots=y labels=n subs=y object=solar_power_allocation.csv "
            "C1=$0000FF00 C2=$00FF0000"
        )

    def initialize_profiler_classes(self) -> None:
        load_data = self.time_series_data_processor.get_processed_load_data()
        self.load_cols = self.time_series_data_processor.load_cols
        self.load_names = self.dss_circuit.Loads.AllNames
        self.num_customers = len(self.load_names)
        self.load_profiler = LoadProfiler(
            dss_circuit=self.dss_circuit,
            config=self.config,
            num_days=self.num_days,
            time_steps=self.time_steps,
            load_names=self.load_names,
            time_series_data_processor=self.time_series_data_processor,
            load_data=load_data,
            load_cols=self.load_cols,
        )
        if self.add_pv:
            peak_pv_data = (
                self.time_series_data_processor.get_processed_peak_solar_data()
            )
            self.solar_profiler = SolarProfiler(
                dss_circuit=self.dss_circuit,
                dss_text=self.dss_text,
                config=self.config,
                num_days=self.num_days,
                time_steps=self.time_steps,
                load_names=self.load_names,
                peak_pv_data=peak_pv_data,
            )

    def initialize_circuit_with_profiles(self) -> None:
        self.logger.info("Initializing circuit with load and pv profiles")
        for load_idx, load_name in enumerate(self.load_names):
            self.load_profiler.generate_initial_load_profile(
                load_idx=load_idx, load_name=load_name
            )
            if self.add_pv:
                self.solar_profiler.setup_pv_system(
                    load_idx=load_idx, load_name=load_name
                )

    def modify_circuit_with_adjusted_pv_profiles(
        self, pv_profiles: pd.DataFrame
    ) -> None:
        self.logger.info("Initializing circuit with load and pv profiles")
        for load_idx, load_name in enumerate(self.load_names):
            modified_power = pv_profiles.loc[load_name, "modified_power"]
            self.solar_profiler.setup_pv_system(
                load_idx=load_idx, load_name=load_name, modified_power=modified_power
            )

        # Uncomment if you wish to plot profiles
        # plot the single phase load profiles being used

        df = self.load_profiler.load_ami_data[
            self.load_profiler.load_ami_data[data_constants.PHASES]
            == data_constants.SINGLE_PHASE
        ]
        # self.plotter.plot_profiles(
        #     profiles=[row for _, row in df[self.load_cols].iterrows()],
        #     title=f"Residential Load Profiles Scaled by Number of Customers for Single Phase Loads",
        #     ylabel="Active Power (kW)",
        # )
        df = self.load_profiler.load_ami_data[
            self.load_profiler.load_ami_data[data_constants.PHASES]
            == data_constants.THREE_PHASE
        ]
        # self.plotter.plot_profiles(
        #     profiles=[row for _, row in df[self.load_cols].iterrows()],
        #     title=f"Residential Load Profiles Scaled by Number of Customers for Three Phase Loads",
        #     ylabel="Active Power (kW)",
        # )

        if self.add_pv:
            df = self.solar_profiler.load_peak_pv_data.to_frame().T
            # plot the solar profiles being used
            # self.plotter.plot_profiles(
            #     profiles=[row for _, row in df.iterrows()],
            #     title="Residential Daily PV Generation Profiles in a Year",
            #     ylabel="Normalised PV Generation",
            # )

    def initialize_data_variables(self):
        self.kW_monitors = np.zeros((self.num_customers, self.time_steps))
        self.kvar_monitors = np.zeros((self.num_customers, self.time_steps))
        self.voltages_monitors = np.zeros((self.num_customers, self.time_steps))
        self.df_kw = pd.DataFrame()
        self.df_kvar = pd.DataFrame()
        self.df_volt = pd.DataFrame()
        self.load_profiles_all = []
        self.pv_profiles_all = []
        self.rand_cust = []
        self.reactive_profiles_all = []

    def monitor_data(self, step: int):
        for load_idx, load_name in enumerate(self.load_names):
            self.dss_circuit.SetActiveElement(f"Load.{load_name}")
            self.kW_monitors[load_idx, step] = self.dss_circuit.ActiveElement.Powers[0]
            self.kvar_monitors[load_idx, step] = self.dss_circuit.ActiveElement.Powers[
                1
            ]
            bus_name = self.dss_circuit.ActiveElement.Properties("bus1").Val
            self.dss_circuit.SetActiveBus(bus_name)
            self.voltages_monitors[load_idx, step] = (
                self.dss_circuit.ActiveBus.puVmagAngle[0]
            )

    def solve_model(self) -> tuple:
        """
        Basic commands to solve model that already has all feeder elements and associated profiles over solve horizon.
        """
        self.dss_text_command(action="Reset")
        self.dss_text_command(
            action=["Set Mode=", "number=", "stepsize="],
            variable=["daily", "1", f"{self.solve_frequency}m"],
        )
        self.dss_text_command(
            action="Set overloadreport=", variable=f"{self.report_thermal_overload}"
        )
        self.dss_text_command(
            action="Set voltexcept=", variable=f"{self.report_voltage_exception}"
        )
        for step in range(self.time_steps):
            self.dss_circuit_solution.Solve()
            self.monitor_data(step=step)
        df_volt = pd.DataFrame(self.voltages_monitors)
        if self.dss_circuit_solution.Converged:
            print("Message: The Solution Converged Successfully\n")
        else:
            print("Message: The Solution Did Not Converge\n")

        modified_power = []
        bus_name = []
        load_name_data = []
        for load_idx, load_name in enumerate(self.load_names):
            self.dss_circuit.SetActiveElement(f"PVSystem.{load_idx}")
            modified_power.append(self.dss_circuit.ActiveElement.Properties("pmpp").Val)
            bus_name.append(self.dss_circuit.ActiveElement.Properties("bus1").Val)
            load_name_data.append(load_name)
        data = {
            "bus_name": bus_name,
            "modified_power": modified_power,
            "load_name": load_name_data,
        }
        df = pd.DataFrame(
            data=data,
        )
        df = df.set_index("load_name")

        # self.plotter.plot_profiles(
        #     profiles=[row for _, row in df_volt[0 : self.num_customers].iterrows()],
        #     title=f"Voltage Profile at each Bus",
        #     ylabel="Voltage (p.u.)",
        #     y_lim=[0.00, 1.10],
        #     x_upper_lim=(1 + self.config.analysis_config.voltage_deviation),
        # )
        return df, self.dss_circuit_solution.Converged

    def simulate_power_flow_analysis(self) -> tuple:
        """
        Simulate a power flow analysis for number of time steps.
        :return dataframe that contains current pv ratings which get modified when maximizing pv
        """
        # remove number if you want different load allocations for missing data for each run
        power_factor_list = self.load_profiler.generate_power_factor()
        self.initialize_data_variables()
        random.seed(42)
        for load_idx, load_name in enumerate(self.load_names):
            load_profile, reactive_profile = (
                self.load_profiler.generate_mapped_load_profile_during_sim(
                    load_idx=load_idx,
                    load_name=load_name,
                    power_factor_list=power_factor_list,
                )
            )
            self.load_profiles_all.append(load_profile)
            self.reactive_profiles_all.append(reactive_profile)
        if self.add_pv:
            for load_idx, load_name in enumerate(self.load_names):
                self.solar_profiler.set_peak_pv_profile_during_sim(
                    load_idx=load_idx, load_name=load_name
                )
        return self.solve_model()

    def simulate_monte_carlo_power_flow_analysis(self) -> tuple:
        """
        Simulate a power flow analysis for number of time steps.
        :return: results from solving model
        """
        power_factor_list = self.load_profiler.generate_power_factor()
        self.initialize_data_variables()
        random.seed(42)  # remove to allow for random paths
        for load_idx, load_name in enumerate(self.load_names):
            load_profile, reactive_profile = (
                self.load_profiler.generate_random_load_profile_during_sim(
                    load_idx=load_idx,
                    load_name=load_name,
                    power_factor_list=power_factor_list,
                )
            )
            self.load_profiles_all.append(load_profile)
            self.reactive_profiles_all.append(reactive_profile)
        if self.add_pv:
            for load_idx, load_name in enumerate(self.load_names):
                self.solar_profiler.set_peak_pv_profile_during_sim(
                    load_idx=load_idx, load_name=load_name
                )
        return self.solve_model()

    def bounded_simulation(self) -> bool:
        """
        Check if pv input resulted in a bounded solution.
        :param pv_df: pv for each bus
        :return: bool True or False
        """
        voltage_deviation = self.config.analysis_config.voltage_deviation
        min_voltage, max_voltage = 0.5, 1 + voltage_deviation
        bounded = np.all(
            (self.voltages_monitors >= min_voltage)
            & (self.voltages_monitors <= max_voltage)
        )
        return bounded

    def adjust_pv_profiles(self, pv_range: int, pv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust pv profiles according to state of voltage monitor with random value.
        :return: updated pv dataframe
        """
        df = pv_df.copy()
        voltage_deviation = self.config.analysis_config.voltage_deviation
        min_voltage, max_voltage = 0.5, 1 + voltage_deviation
        for load_idx, load_name in enumerate(self.load_names):
            row = self.voltages_monitors[load_idx]
            current_pv = int(df.loc[load_name, "modified_power"])
            max_range = max(1, int(pv_range))
            if np.all((row > min_voltage) & (row < max_voltage)):  # within bounds
                new_pv = max(0, current_pv + np.random.randint(low=0, high=max_range))
            else:
                new_pv = max(0, current_pv - np.random.randint(low=0, high=max_range))
            df.loc[load_name, "modified_power"] = str(new_pv)
        return df

    def pv_powers_bounded(self, pv_df: pd.DataFrame):
        """
        Get pv powers that were bounded
        :return:
        """
        df = pv_df.copy()
        voltage_deviation = self.config.analysis_config.voltage_deviation
        min_voltage, max_voltage = 0.5, 1 + voltage_deviation
        for load_idx, load_name in enumerate(self.load_names):
            row = self.voltages_monitors[load_idx]
            if np.all((row > min_voltage) & (row < max_voltage)):  # within bounds
                out_of_bounds_bus = int(df.loc[load_name, "modified_power"])
            else:
                out_of_bounds_bus = 0
            df.loc[load_name, "modified_power"] = str(out_of_bounds_bus)
        return df

    def update_max_pv(
        self,
        converged: bool,
        bounded: bool,
        pv_powers_df: pd.DataFrame,
        max_pv: int,
        max_pv_power_df: pd.DataFrame,
    ):
        if converged and bounded:
            acceptable_pv = pv_powers_df["modified_power"].astype(int).sum()
            if acceptable_pv > max_pv:
                max_pv_power_df["modified_power"] = pv_powers_df["modified_power"]
                max_pv = max_pv_power_df["modified_power"].astype(int).sum()
        elif converged:
            new_df = self.pv_powers_bounded(pv_df=pv_powers_df)
            if new_df["modified_power"].astype(int).sum() > max_pv:
                max_pv_power_df["modified_power"] = new_df["modified_power"]
                max_pv = max_pv_power_df["modified_power"].astype(int).sum()
        return max_pv_power_df, max_pv

    def maximize_solar_allocation_monte_carlo(self) -> float:
        """
        Maximize solar allocation
        :return total hosting capacity
        """
        self.initialize_circuit_with_profiles()
        pv_powers_df, converged = self.simulate_monte_carlo_power_flow_analysis()
        delta_pv = int(
            self.config.pv_config.max_power_output
            / self.config.analysis_config.maximize_pv_iterations
        )
        max_pv_power_df = self.pv_powers_bounded(pv_df=pv_powers_df)
        max_pv = max_pv_power_df["modified_power"].astype(int).sum()
        for i in range(self.config.analysis_config.maximize_pv_iterations):
            pv_range = self.config.pv_config.max_power_output - i * delta_pv
            adjusted_pv_powers_df = self.adjust_pv_profiles(
                pv_range=pv_range, pv_df=pv_powers_df
            )
            self.modify_circuit_with_adjusted_pv_profiles(
                pv_profiles=adjusted_pv_powers_df
            )
            pv_powers_df, converged = self.simulate_monte_carlo_power_flow_analysis()
            bounded = self.bounded_simulation()
            max_pv_power_df, max_pv = self.update_max_pv(
                converged=converged,
                bounded=bounded,
                pv_powers_df=pv_powers_df,
                max_pv=max_pv,
                max_pv_power_df=max_pv_power_df,
            )
        # save maximum pv power dataframe to csv
        max_pv_power_df.to_csv("solar_power_allocation.csv", index=False, header=False)
        total_hosting_cap = max_pv_power_df["modified_power"].astype(int).sum()
        self.logger.info(
            f"The total hosting capacity for feeder {self.config.circuit_config.feeder_name} is {total_hosting_cap}"
        )
        return total_hosting_cap

    def maximize_solar_allocation(self) -> float:
        """
        Maximize solar allocation
        :return total hosting capacity
        """
        self.initialize_circuit_with_profiles()
        pv_powers_df, converged = self.simulate_power_flow_analysis()
        delta_pv = int(
            self.config.pv_config.max_power_output
            / self.config.analysis_config.maximize_pv_iterations
        )
        max_pv_power_df = self.pv_powers_bounded(pv_df=pv_powers_df)
        max_pv = max_pv_power_df["modified_power"].astype(int).sum()
        for i in range(self.config.analysis_config.maximize_pv_iterations):
            pv_range = self.config.pv_config.max_power_output - i * delta_pv
            adjusted_pv_powers_df = self.adjust_pv_profiles(
                pv_range=pv_range, pv_df=pv_powers_df
            )
            self.modify_circuit_with_adjusted_pv_profiles(
                pv_profiles=adjusted_pv_powers_df
            )
            pv_powers_df, converged = self.simulate_power_flow_analysis()
            bounded = self.bounded_simulation()
            max_pv_power_df, max_pv = self.update_max_pv(
                converged=converged,
                bounded=bounded,
                pv_powers_df=pv_powers_df,
                max_pv=max_pv,
                max_pv_power_df=max_pv_power_df,
            )

        # make sure current pv powers is plotted
        max_pv_power_df.to_csv("solar_power_allocation.csv", index=False, header=False)
        total_hosting_cap = max_pv_power_df["modified_power"].astype(int).sum()
        self.logger.info(
            f"The total hosting capacity for feeder {self.config.circuit_config.feeder_name} is {total_hosting_cap}"
        )
        return total_hosting_cap
