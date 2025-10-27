"""This file contains constants used to perform a hosting capacity study"""


class SolarConstants:
    CLOUD_COVER_FACTOR: float = 0.75
    CLOUD_COVER_MULTIPLIER: float = 3.4
    SOLAR_RADIATION_FACTOR: float = 990
    SOLAR_RADIATION_OFFSET: float = 30
    MAX_SOLAR_RADIATION: float = SOLAR_RADIATION_FACTOR - SOLAR_RADIATION_OFFSET


class LoadDemandConstants:
    POWER_FACTOR_MIN: float = 0.87
    POWER_FACTOR_MAX: float = 0.98


class DataNameConstants:
    FEEDER_ID: str = "FEEDERID"
    TRANSFORMER_ID: str = "TRANSFORMERID"
    FACILITY_ID: str = "FACILITYID"
    NUM_CUSTOMERS: str = "CUSTOMERS"
    TIME_DATA_COL: str = r"^H\d+$"
    PHASES: str = "PHASES"
    A_PHASE: str = "APHASE"
    B_PHASE: str = "BPHASE"
    C_PHASE: str = "CPHASE"
    SINGLE_PHASE: int = 1
    THREE_PHASE: int = 3


# the below are imported
solar_constants = SolarConstants()
load_constants = LoadDemandConstants()
data_constants = DataNameConstants()
