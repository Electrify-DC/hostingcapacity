# Hosting Capacity Analysis

This repository contains scripts to process time series data provided with customer mapping and load profiles, perform hosting capacity analysis with mapped load allocation, and perform a series of monte carlo simulations with random load allocation. 

Please read the below for details around each script and warnings for how to assess results. 

## Required Libraries
The required libraries are contained in requirements.txt ensure you have a working virtual environment set up with the libraries listed in the requirements. Ensure your python version is at least 3.11. This repo has not been built or tested with any lower versions of Python. 

## Config
The config is the place where you as a user can modify any of the parameters to get various simulation results. The user must provide valid inputs otherwise the simulation can provide invalid results. Ensure that all model files are in opendss format and that relevant time series data has been provided in the paths you set in the config. 

The path should be where your feeder data is. 

## Scripts
- hosting_capacity_build_profiles()
  - Use this to build a load profile for a specified feeder by providing a customer mapping and load data 

- hosting_capacity_build_population_profiles
  - Use this to concatenate various load profiles to create a "population" of load profiles which is used in the monte carlo simulations

- hosting_capacity_single_run
  - Use this to perform a single simulation to analyze hosting capacity and determine max pv penetration over the number of iterations defined in the config

- hosting_capacity_monte_carlo
  - Use this to perform a series of monte carlo simulations to analyze hosting capacity for various load profiles chosen from the population of load profiles
  - Performs number of simulations defined in config and maximizes over number of iterations defined in config

## Debugging
If your initial parameters for pv are wrong you may have results that don't make sense. For example a good gut check is running a simulation with and without an inverter config. If the results for without an inverter are greater than with an inverter. It is likely you need to change your initial settings. 

Again if the solution fails to converge for all runs your initial conditions may be incorrect or you may not be simulating over enough time to reach an acceptable result.


## Acknowledgement

#### University of Melbourne Team Nando's Pages:
This code could not have been written without learning from these resources first.

* https://sites.google.com/view/luisfochoa/research/research-team
* https://sites.google.com/view/luisfochoa/research/past-team-members


## Licenses

Since this repository uses DSS-Python which is based on OpenDSS, both licenses have been included. Check all corresponding files (`LICENSE-OpenDSS`, `LICENSE-DSS-Python`, `LICENSE`).
