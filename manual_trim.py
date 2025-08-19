"""Trim solver for glider
For a given flight speed and altitude, computes AoA, control for p=q=r=hddot
(Bascially steady glider for that speed)"""

from pathlib import Path
from typing import Tuple, Optional, Dict
import importlib.util
import json


import numpy as np
from scipy.optimize import least_squares

from plav.vehicle_models.generic_aircraft_config import AircraftConfig, init_aircraft, init_dummy_aircraft, alpha_beta_to_velocity
from plav.atmosphere_models.ussa1976 import get_pressure_density_temp
from plav.simulator import get_gravity
from plav.plav import Plav

from plav.trim_solver import trim_glider_hddot0

scenario_file = "scenarios/brgrDroneDrop.json"

if not Path(scenario_file).exists():
    print(f"Scenario file {scenario_file} does not exist")

with open(scenario_file, 'r') as file:
    modelparam = json.load(file)
file.close()
# needs to change


control_unit = None
if modelparam['use_generic_aircraft']:
    print('Using Generic Model')
    aircraft = init_aircraft(modelparam)
else:
    custom_fdm = Path("./vehicle_models/" + modelparam['aircraft_file'])
    spec_aircraft = importlib.util.spec_from_file_location(custom_fdm.stem, str(custom_fdm))
    aircraft_plugin: AircraftConfig = importlib.util.module_from_spec(spec_aircraft)
    spec_aircraft.loader.exec_module(aircraft_plugin)
    aircraft = aircraft_plugin.init_aircraft(modelparam)

    if 'control_file' in modelparam:
        custom_control = Path("./control_modules/" + modelparam['control_file'])
        spec_control = importlib.util.spec_from_file_location(custom_control.stem, str(custom_control))
        control_plugin = importlib.util.module_from_spec(spec_control)
        spec_control.loader.exec_module(control_plugin)
        control_unit = control_plugin.init_control(modelparam)



result = trim_glider_hddot0(16, 0, aircraft)

print(result)
