"""
Implenting physics simulation by using an RungeKutta integrator at a fixed timestep
This one uses WGS84 and keeps tract of long lat in NED

Refactored 3 to OOO style

"""
import sys
import json
import time
import math
from pathlib import Path

import numpy as np
#from scipy.integrate import solve_ivp
from numba import jit, float64
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg
import pandas as pd
from flightgear_python.fg_if import FDMConnection
from serial.serialutil import SerialException

from plav.quaternion_math import from_euler
from plav.simulator import Simulator
from plav.vehicle_models.generic_aircraft_config import AircraftConfig, init_aircraft, init_dummy_aircraft
from plav.vehicle_models.brgr_model import init_brgr, BRGRConfig
from plav.vehicle_models.f16_model import F16_aircraft
from plav.atmosphere_models.ussa1976 import Atmosphere
from plav.step_logging import SimDataLogger
from src.plav.f16_control import F16Control, tas_to_eas
from src.plav.f16_control_HITL import F16ControlHITL
from plav.joystick_reader import JoystickReader
from plav.plotter import Plotter

import plav.conversions as conv

#from pyqtgraph.Qt import QtWidgets


def init_position(long, lat, alt, velocity, bearing, elevation, roll, init_omega):
    """initalize the state"""

    init_pos = np.array([long,lat,alt])

    init_vel = velocity

    #first apply bearing stuff
    #roll pitch yaw
    init_ori_ned = from_euler(roll*math.pi/180,elevation*math.pi/180,bearing*math.pi/180)


    y0 = np.append(np.append(init_ori_ned, init_omega), np.append(init_pos, init_vel))
    return y0

def init_sim():
    """initialize the simulation
    if given a config file, it will load the config
    if not, it will compile the simulation
    """

    sim_comp_start_time = time.perf_counter()

    dummy_atmosphere = Atmosphere()
    dummy_aircraft = init_dummy_aircraft()
    dummy_control = None

    dummy_y0 = init_position(long = 0.0, lat = 0.0, alt = 0.0, velocity = 0.0,
                    bearing = 0.0, elevation = 0.0, roll = 0.0,
                    init_omega = np.array([0.0, 0.0, 0.0, 0.0], 'd'))

    dummy_t_span = np.array([0.0, 0.01],'d')
    sim_object = Simulator(init_state = dummy_y0, time_span = dummy_t_span, aircraft=dummy_aircraft,
                            sim_atmosphere=dummy_atmosphere, control_sys=dummy_control, t_step=0.01)

    try:
        sim_object.pump_sim()
    except KeyboardInterrupt:
        print("Simulation compilation aborted")
        exit(0)

    sim_comp_end_time = time.perf_counter()
    #sim_comp_time = sim_comp_end_time - sim_comp_start_time

def load_scenario(scenario_file):
    """load a scenario file"""
    if not Path(scenario_file).exists():
        print(f"Scenario file {scenario_file} does not exist")
        return None

    with open(scenario_file, 'r') as file:
        modelparam = json.load(file)
    file.close()

    if modelparam['useF16'] and modelparam['hasgridfins']:
        print("Cannot use F16 and grid fins at the same time")
        return None

    # needs to change 
    return modelparam

def load_aircraft_config(modelparam):
    """load the aircraft config and control unit if relevant from the modelparam"""
    control_unit = None
    if modelparam['useF16']:
        print('Using F16')
        control_vector = np.array(modelparam['init_control'], 'd')
        aircraft = F16_aircraft(control_vector)

        if modelparam["useSAS"] and not modelparam['hitl_active']:
            print('Using Software Autopilot')
            control_unit = F16Control(np.array(modelparam['commands'], 'd'))
            stability_augmentation_on_disc, autopilot_on_disc = 1.0, 1.0
            control_unit.update_switches(stability_augmentation_on_disc, autopilot_on_disc)

        if modelparam["useSAS"] and modelparam['hitl_active']:
            print('Using HITL Autopilot')
            try:
                control_unit = F16ControlHITL(np.array(modelparam['commands'], 'd'), 'COM5')
            except SerialException:
                print("Serial port error, check if the arduino is connected and available")
                sys.exit(1)
            stability_augmentation_on_disc, autopilot_on_disc = 1.0, 1.0
            control_unit.update_switches(stability_augmentation_on_disc, autopilot_on_disc)
    elif modelparam['hasgridfins']:
        print('Using Grid Fin Model')
        aircraft = init_brgr(modelparam)
    else:
        print('Using Generic Model')
        aircraft = init_aircraft(modelparam)

    return aircraft, control_unit

def load_atmosphere(modelparam, use_file_atmosphere:bool = True):
    """load the atmosphere config from the modelparam and return an Atmosphere object"""
    if use_file_atmosphere:
        wind_alt_profile       = np.array(modelparam['wind_alt_profile'], dtype='d')
        wind_speed_profile     = np.array(modelparam['wind_speed_profile'], dtype='d')
        wind_direction_profile = np.array(modelparam['wind_direction_profile'], dtype='d')
    else:
        wind_alt_profile = np.array([0, 0], dtype='d')
        wind_speed_profile = np.array([0, 0], dtype='d')
        wind_direction_profile = np.array([0, 0], dtype='d')
    #init atmosphere config
    return Atmosphere(wind_alt_profile,wind_speed_profile,wind_direction_profile)

def load_init_position(modelparam):
    """load the initial position from the modelparam and return y0"""
    init_long = modelparam['init_lon']
    init_lat = modelparam['init_lat']
    inital_alt = modelparam['init_alt']
    init_velocity = modelparam['init_vel']
    init_rte = modelparam['init_rot']
    init_ori = np.array(modelparam['init_ori'], 'd')

    init_x = init_position(init_long, init_lat, inital_alt, init_velocity,
                    bearing=init_ori[2], elevation=init_ori[1],
                    roll=init_ori[0], init_omega=init_rte)

    return init_x

def export_data(trimmed_sim_data):
    """Export the simulation data to a CSV file"""
    csv_data = {'time': trimmed_sim_data[0],
        'altitudeMsl_ft': trimmed_sim_data[10]*conv.M_TO_FT,
        'longitude_deg': trimmed_sim_data[8]*conv.RAD_TO_DEG,
        'latitude_deg': trimmed_sim_data[9]*conv.RAD_TO_DEG,
        'localGravity_ft_s2': trimmed_sim_data[23] *conv.M_TO_FT,
        'eulerAngle_deg_Yaw':  trimmed_sim_data[16] *conv.RAD_TO_DEG,
        'eulerAngle_deg_Pitch': trimmed_sim_data[15] *conv.RAD_TO_DEG,
        'eulerAngle_deg_Roll' : trimmed_sim_data[14] *conv.RAD_TO_DEG,
        'aero_bodyForce_lbf_X': trimmed_sim_data[17] *conv.N_TO_LBF,
        'aero_bodyForce_lbf_Y': trimmed_sim_data[18] *conv.N_TO_LBF,
        'aero_bodyForce_lbf_Z': trimmed_sim_data[19] *conv.N_TO_LBF,
        'aero_bodyMoment_ftlbf_L': trimmed_sim_data[20] *conv.NM_TO_LBF_FT,
        'aero_bodyMoment_ftlbf_M': trimmed_sim_data[21] *conv.NM_TO_LBF_FT,
        'aero_bodyMoment_ftlbf_N': trimmed_sim_data[22] *conv.NM_TO_LBF_FT,
        'trueAirspeed_nmi_h': trimmed_sim_data[30]*conv.MPS_TO_KTS,
        'airDensity_slug_ft3': trimmed_sim_data[27] *conv.M_TO_FT,
        'downrageDistance_m': trimmed_sim_data[35],
        }

    df = pd.DataFrame(csv_data)
    filename = "output.csv"
    df.to_csv(filename, index=False)
    print(f"Data exported to {filename}")


def fdm_callback(fdm_data, current_pos):
    """updates flight data for Flightgear"""

    fdm_data.lon_rad = current_pos[0]
    fdm_data.lat_rad = current_pos[1]
    fdm_data.alt_m = current_pos[2]
    fdm_data.phi_rad = current_pos[3]
    fdm_data.theta_rad = current_pos[4]
    fdm_data.psi_rad = current_pos[5]
    #fdm_data.alpha_rad = sim_data[31, -1]
    #fdm_data.beta_rad = sim_data[31, -1]
    #fdm_data.phidot_rad_per_s = sim_data[5,-1]
    #fdm_data.thetadot_rad_per_s = sim_data[6,-1]
    #fdm_data.psidot_rad_per_s = sim_data[7,-1]
    #fdm_data.v_north_ft_per_s = sim_data[11,-1]*39.37/12
    #fdm_data.v_east_ft_per_s = sim_data[12,-1]*39.37/12
    #fdm_data.v_down_ft_per_s = sim_data[13,-1]*39.37/12
    return fdm_data  # return the whole structure


def start_simulation(scenario_file: str, timespan, real_time=False, export_to_csv=True):
    """Starts the simulation for the given scenario file"""
    use_flight_gear = False
    timespan = np.array(timespan,'d')
    #test code to make sure refactor still works

    #load the scenario file
    modelparam = load_scenario("scenarios/" + scenario_file)
    if modelparam is None:
        print("No valid scenario file found, exiting")
        sys.exit(1)

    #load the aircraft config and control unit
    aircraft, control_unit = load_aircraft_config(modelparam)

    #load the atmosphere config
    atmosphere = load_atmosphere(modelparam)

    #load the initial position
    y0 = load_init_position(modelparam)

    #setup the HITL system if active
    if control_unit is not None:
        hitl_active = control_unit.is_hitl()
    else:
        hitl_active = False

    #initialize the simulation object
    t_span = np.array([0.0, 120.0], 'd')

    sim_object = Simulator(y0, t_span, aircraft, atmosphere, control_sys = control_unit, t_step=0.01)


    # Create main Qt application
    app = QtWidgets.QApplication([])
    realtime_window = QtWidgets.QMainWindow()
    realtime_window.setWindowTitle('Real Time Flying')

    pg.setConfigOptions(antialias=True)

    # Create a central widget and layout
    central_widget = QtWidgets.QWidget()
    instrument_widget = QtWidgets.QWidget()
    main_layout = QtWidgets.QVBoxLayout()
    controls_layout = QtWidgets.QVBoxLayout()
    central_widget.setLayout(main_layout)
    instrument_widget.setLayout(controls_layout)
    realtime_window.setCentralWidget(instrument_widget)

    # Create plot area using pyqtgraph GraphicsLayoutWidget
    plot_widget = pg.GraphicsLayoutWidget()
    main_layout.addWidget(plot_widget)

    # Create control panel with Pause/Unpause buttons
    button_layout = QtWidgets.QHBoxLayout()
    pause_button = QtWidgets.QPushButton("Pause/Play")
    joystick = pg.JoystickButton()
    joystick.setFixedWidth(30)
    joystick.setFixedHeight(30)
    #unpause_button = QtWidgets.QPushButton("Unpause")
    button_layout.addWidget(pause_button)
    button_layout.addWidget(joystick)
    #button_layout.addWidget(unpause_button)
    controls_layout.addLayout(button_layout)

    # Connect buttons to simulation control
    pause_button.clicked.connect(sim_object.pause_or_unpause_sim)
    #unpause_button.clicked.connect(sim_object.unpause_sim)


    if real_time is False:
        sim_start_time = time.perf_counter()
        sim_object.run_sim()
        sim_end_time = time.perf_counter()

    sim_data = sim_object.return_results()

    plotter_object = Plotter(sim_data, modelparam['title'])


    def update():
        """Update loop for simulation"""
        x, y = joystick.getState()
        sim_object.update_manual_control(stick_x=x, stick_y=y)

        sim_data = sim_object.update_real_time()

        plotter_object.update_plots(sim_data)
        
    timer = QtCore.QTimer()
    timer.timeout.connect(update)

    if not real_time:
        #print("Sim took ", sim_end_time-sim_start_time)
        print('Data Rows: ', sim_data[0].size)

    fdm_event_pipe = None
    if __name__ == '__main__' and use_flight_gear:
        print("Starting FlightGear Connection")
        fdm_conn = FDMConnection()
        print('broke after fdm')
        fdm_event_pipe = fdm_conn.connect_rx('localhost', 5501, fdm_callback)
        print('broke at pipe')
        fdm_conn.connect_tx('localhost', 5502)
        print('broke at start')
        fdm_conn.start()  # Start the FDM RX/TX loop
        print("Started FlightGear Connection")

    update_loop_speed_ms = 10
    timer.start(update_loop_speed_ms)

    if export_to_csv and not real_time:
        print("Exporting data to CSV")
        export_data(sim_data)


    realtime_window.show()
    pg.exec()


    if hitl_active: #shut down the HIL system
        try:
            control_unit.shut_down_hil()
            print("HITL system shut down")
        except SerialException:
            print("HITL shut down undisgracefully")

start_simulation("brgrDroneDrop.json",[0,120])
