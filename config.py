from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FileConfig:
    """
    Parameters related to file structure.
    """

    path: str
    name: str
    processed: bool  # if processed for open dss analysis: if first time running after conversion to dss set to True
    converted: bool  # if converted to open dss


@dataclass
class CircuitConfig:
    """
    Parameters related to the circuit under analysis.
    """

    feeder_name: int
    secondary_circuit_present: bool
    frequency: int  # hertz
    phases: int  # max number of phases present
    balanced: bool


@dataclass
class AnalysisConfig:
    """
    Parameters related to the analysis.
    """

    voltage_deviation: float  # per unit
    report_thermal_overload: bool
    report_voltage_exception: bool
    load_randomization: bool
    solar_randomization: bool
    max_iterations: int
    solve_frequency: int  # minutes
    simulation_length: int  # hour
    initial_power_factor: float  # max 1
    min_power_factor: float
    max_power_factor: float
    day_of_year: int
    num_of_days: int
    maximize_pv_iterations: int
    monte_carlo_sims: int


@dataclass
class DataConfig:
    """
    Parameters related to names of data.
    """

    customer_load_data: str
    customer_solar_data: str
    solar_peak_profile: str
    feeder_to_home_mapping: str
    customer_load_population: str


@dataclass
class PvConfig:
    """
    Parameters related to configuration of each pv system.
    """

    power_rating: int
    max_power_output: int
    irradiance: float
    power_factor: float  # unity = 1
    v_max_pu: float
    v_min_pu: float
    cut_in: float
    cut_out: float


@dataclass
class InverterConfig:
    """
    Parameters related to configuration of each inverter
    """

    xcurve: list[float]
    ycurve: list[float]
    setting: str
    delta_p: float
    deltav_tolerance: float
    deltavar_tolerance: float
    enabled: bool


@dataclass
class Config:
    """
    The main configurations used to simulate a power flow analysis.
    """

    file_config: FileConfig = field(default_factory=FileConfig)
    circuit_config: CircuitConfig = field(default_factory=CircuitConfig)
    analysis_config: AnalysisConfig = field(default_factory=AnalysisConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    pv_config: Optional[PvConfig] = None
    inverter_config: Optional[InverterConfig] = None
