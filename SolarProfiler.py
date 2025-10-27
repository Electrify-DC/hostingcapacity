from typing import Union, Any

import numpy as np
import pandas as pd
from dss.ICircuit import ICircuit
from dss.IText import IText

from TransmissionDistribution.config import Config


class SolarProfiler:
    """
    This class contains methods to generate and manipulate solar profiles.
    """

    def __init__(
        self,
        dss_circuit: ICircuit,
        dss_text: IText,
        config: Config,
        num_days: int,
        time_steps: int,
        load_names: list[str],
        peak_pv_data: pd.DataFrame,
    ):
        self.dss_circuit = dss_circuit
        self.dss_text = dss_text
        self.load_names = load_names
        self.analysis_config = config.analysis_config
        self.config = config
        self.data_config = config.data_config
        self.num_days = num_days
        self.time_steps = time_steps
        self.pv_config = self.config.pv_config
        self.peak_pv_data_df = peak_pv_data

    @property
    def pv_irradiance(self):
        return self.pv_config.irradiance

    @property
    def pv_cut_in(self):
        return self.pv_config.cut_in

    @property
    def pv_cut_out(self):
        return self.pv_config.cut_out

    @property
    def v_max_pu(self):
        return self.pv_config.v_max_pu

    @property
    def v_min_pu(self):
        return self.pv_config.v_min_pu

    @property
    def power_rating(self):
        return self.pv_config.power_rating

    @property
    def max_power_output(self):
        return self.pv_config.max_power_output

    @property
    def power_factor(self):
        return self.pv_config.power_factor

    @property
    def add_inverter(self) -> bool:
        """
        Is true if inverter config is present - meaning add inverter to simulation.
        """
        if getattr(self.config, "inverter_config"):
            return True
        return False

    @property
    def load_peak_pv_data(self):
        """
        Loads pv data from a processed dataframe or a file path.
        :return: pv data frame
        """
        if self.peak_pv_data_df is not None:
            return self.peak_pv_data_df
        else:
            pv_data_name = self.data_config.solar_peak_profile
            path = self.config.file_config.path
            file = f"{path}/{pv_data_name}"
            pv_data = None
            if "npy" in pv_data_name:
                pv_data = np.load(file)
            elif "csv" in pv_data_name:
                pv_data = pd.read_csv(file)
            return pv_data

    @property
    def num_profiles(self):
        return len(self.load_pv_data)

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
            command_parts.append(f"{a}{v}")

        # If one list is longer, append remaining elements
        longer_list = action if len(action) > len(variable) else variable
        command_parts.extend(longer_list[len(command_parts) :])

        # Construct and execute the command
        command = " ".join(map(str, command_parts)).strip()
        self.dss_text.Command = command

    def setup_pv_system(
        self, load_idx: int, load_name: str, modified_power: float = None
    ) -> None:
        """
        Generate initial pv profile for pv systems.
        :param modified_power: power rating inputted (used during maximization of solar)
        :param load_idx: index of load
        :param load_name: name of load
        """
        if modified_power is None:
            command = "new"
            kva = self.power_rating
            pmpp = self.max_power_output
        else:
            command = "edit"
            kva = modified_power
            pmpp = modified_power
        pv_profile = np.zeros(self.time_steps)
        self.dss_circuit.LoadShapes.New(f"pv_profile_{load_idx}")
        self.dss_circuit.LoadShapes.Npts = self.time_steps
        self.dss_circuit.LoadShapes.MinInterval = self.analysis_config.solve_frequency
        self.dss_circuit.LoadShapes.UseActual = True
        self.dss_circuit.LoadShapes.Pmult = pv_profile.tolist()
        self.dss_circuit.SetActiveElement(f"load.{load_name}")
        bus_name = self.dss_circuit.ActiveElement.Properties("bus1").Val
        num_phases = self.dss_circuit.ActiveElement.Properties("phases").Val
        base_voltage = self.dss_circuit.ActiveElement.Properties("kV").Val
        self.dss_text_command(
            action=[
                f"{command} PVSystem.",
                "phases=",
                "irradiance=",
                "%cutin=",
                "%cutout=",
                "vmaxpu=",
                "vminpu=",
                "kva=",
                "pmpp=",
                "bus1=",
                "pf=",
                "kv=",
                "varfollowinverter=",
                "enabled=",
            ],
            variable=[
                load_idx,
                num_phases,
                self.pv_irradiance,
                self.pv_cut_in,
                self.pv_cut_out,
                self.v_max_pu,
                self.v_min_pu,
                kva,
                pmpp,
                bus_name,
                self.power_factor,
                base_voltage,
                True,
                True,
            ],
        )
        if self.add_inverter:
            self.set_inverter(load_idx=load_idx, command=command)

    def generate_pv_profile_during_sim(self, load_idx: int, load_name: str):
        """
        Generate pv profile during simulation.
        :param day: day of the year
        :param load_idx: index of load
        :param load_name: name of load
        :return: pv profile as list
        """
        pv_profile_name = f"pv_profile_{load_idx}"
        if len(self.load_pv_data.shape) == 1:
            pv_profile = self.load_pv_data
        else:
            pv_profile = self.load_pv_data[0, :]  # replace 0 with random number
        self.dss_circuit.LoadShapes.Name = pv_profile_name
        self.dss_circuit.LoadShapes.Pmult = pv_profile.tolist()
        self.dss_circuit.SetActiveElement(f"load.{load_name}")
        bus_name = self.dss_circuit.ActiveElement.Properties("bus1").Val
        self.dss_circuit.SetActiveElement(f"PVSystem.{load_idx}")
        self.dss_circuit.ActiveElement.Properties("bus1").Val = str(bus_name)
        self.dss_circuit.ActiveElement.Properties("daily").Val = pv_profile_name
        return pv_profile

    def set_peak_pv_profile_during_sim(self, load_idx: int, load_name: str):
        """
        Set peak pv profile during simulation.
        :param load_idx: index of load
        :param load_name: name of load
        """
        pv_profile_name = f"pv_profile_{load_idx}"
        pv_profile = self.load_peak_pv_data.to_numpy()
        self.dss_circuit.LoadShapes.Name = pv_profile_name
        self.dss_circuit.LoadShapes.Pmult = pv_profile.tolist()
        self.dss_circuit.SetActiveElement(f"load.{load_name}")
        bus_name = self.dss_circuit.ActiveElement.Properties("bus1").Val
        self.dss_circuit.SetActiveElement(f"PVSystem.{load_idx}")
        self.dss_circuit.ActiveElement.Properties("bus1").Val = str(bus_name)
        self.dss_circuit.ActiveElement.Properties("daily").Val = pv_profile_name

    def set_inverter(self, load_idx: int, command: str):
        npts = len(self.config.inverter_config.xcurve)
        self.dss_text_command(
            action=[f"{command} XYCurve.", "npts=", "Yarray=", "Xarray="],
            variable=[
                f"vw_curve_{load_idx}",
                npts,
                self.config.inverter_config.ycurve,
                self.config.inverter_config.xcurve,
            ],
        )
        self.dss_text_command(
            action=[
                f"{command} InvControl.",
                "mode=",
                "voltage_curvex_ref=",
                "vvc_curve1=",
                "varchangetolerance=",
                "voltagechangetolerance=",
                "refreactivepower=",
            ],
            variable=[
                f"InvPVCtrl_{load_idx}",
                self.config.inverter_config.setting,
                "rated",
                f"vw_curve_{load_idx}",
                self.config.inverter_config.deltavar_tolerance,
                self.config.inverter_config.deltav_tolerance,
                "varmax_watts",
            ],
        )
