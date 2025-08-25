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
import importlib.util
import socket
import struct
import json
import typing

import numpy as np
#from scipy.integrate import solve_ivp
from numba import jit, float64
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg
import pandas as pd
from flightgear_python.fg_if import FDMConnection
from serial.serialutil import SerialException
import typer

from plav.quaternion_math import from_euler
from plav.simulator import Simulator
from plav.vehicle_models.generic_aircraft_config import AircraftConfig, init_aircraft, init_dummy_aircraft
from plav.atmosphere_models.ussa1976 import Atmosphere
from plav.step_logging import SimDataLogger, return_data_for_csv
from plav.joystick_reader import JoystickReader
from plav.plotter import Plotter
from plav.pilot_control import PilotJoystick

import plav.conversions as conv
import plav.step_logging as slog # for log data indices

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

def init_null_sim():
    """initialize the simulation
    if given a config file, it will load the config
    if not, it will compile the simulation
    """

    #sim_comp_start_time = time.perf_counter()

    dummy_atmosphere = Atmosphere()
    dummy_aircraft = init_dummy_aircraft()
    dummy_control = None

    dummy_y0 = init_position(long = 0.0, lat = 0.0, alt = 0.0, velocity = 0.0,
                    bearing = 0.0, elevation = 0.0, roll = 0.0,
                    init_omega = np.array([0.0, 0.0, 0.0, 0.0], 'd'))

    dummy_t_span = np.array([0.0, 0.01],'d')
    sim_object = Simulator(init_state = dummy_y0, time_span = dummy_t_span,
                            aircraft=dummy_aircraft, sim_atmosphere=dummy_atmosphere,
                            control_sys=dummy_control, t_step=0.01)

    try:
        sim_object.pump_sim()
    except KeyboardInterrupt:
        print("Simulation compilation aborted")
        exit(0)

    #sim_comp_end_time = time.perf_counter()
    #sim_comp_time = sim_comp_end_time - sim_comp_start_time

def load_scenario(scenario_file) -> dict:
    """load a scenario file"""
    if not Path(scenario_file).exists():
        print(f"Scenario file {scenario_file} does not exist")
        return None

    with open(scenario_file, 'r') as file:
        modelparam = json.load(file)
    file.close()
    # needs to change
    return modelparam

def load_aircraft_config(modelparam) -> tuple[AircraftConfig, typing.Any]:
    """load the aircraft config and control unit if relevant from the modelparam"""
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

def change_aircraft(sim_object: Simulator, modelparam, use_control_unit=True):
    """Changes the aircraft to what's described in modelparam"""
    aircraft, control_unit = self.load_aircraft_config(modelparam)

    sim_object.change_aircraft(aircraft)
    if use_control_unit:
        sim_object.change_control_sys(control_unit)
    return

def change_control(sim_object: Simulator, modelparam):
    """Changes the control unit to what's user specified"""
    custom_control = Path("./control_modules/" + modelparam['control_file'])
    spec_control = importlib.util.spec_from_file_location(custom_control.stem, str(custom_control))
    control_plugin = importlib.util.module_from_spec(spec_control)
    spec_control.loader.exec_module(control_plugin)
    control_unit = control_plugin.init_control(modelparam)

    sim_object.change_control(control_unit)
    return

def export_data(trimmed_sim_data):
    """Export the simulation data to a CSV file"""
    csv_data = return_data_for_csv(trimmed_sim_data)
    df = pd.DataFrame(csv_data)
    filename = "output.csv"
    df.to_csv(filename, index=False)
    print(f"Data exported to {filename}")


class Plav(object):
    """Plav Simulator Object. Instaniating launches a simulator thread"""
    def __init__(self, scenario_file: str, timespan,
                         real_time=False, no_gui = False, export_to_csv=True, runsim=True,
                         use_sitl=False, ardupilot_ip = "127.0.0.1"):
        self.no_gui = no_gui
        use_flight_gear = False
        self.real_time = real_time
        self.use_sitl = use_sitl
        self.ardupilot_ip = ardupilot_ip

        #load the scenario file
        modelparam = load_scenario("scenarios/" + scenario_file)
        if modelparam is None:
            print("No valid scenario file found, exiting")
            sys.exit(1)

        self.aircraft, self.control_unit = load_aircraft_config(modelparam)
        atmosphere = load_atmosphere(modelparam)
        y0 = load_init_position(modelparam)

        self.hitl_active = False
        if self.control_unit is not None:
            self.hitl_active = self.control_unit.is_hitl()
            

        t_span = np.array(timespan, 'd')
        self.sim_object = Simulator(y0, t_span, self.aircraft, atmosphere,
                            control_sys = self.control_unit, t_step=0.01)

        if real_time is False:
            sim_start_time = time.perf_counter()
            self.sim_object.run_sim()
            sim_end_time = time.perf_counter()

        self.window_title = modelparam['title']
        self.active = True
        self.address = None

        if self.use_sitl:
            # --- UDP communication setup ---
            print('Initalizing SITL UDP communication')
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((ardupilot_ip, 9002))
            self.sock.settimeout(0.1)

            self.last_sitl_frame = -1
            self.connected = False
            self.frame_count = 0
            self.frame_time = time.time()
            self.print_frame_count = 1000

        if runsim:
            self.run_simulation()


    def run_simulation(self):
        """Main thread for running the sim (blocking)"""

        sim_data = self.sim_object.return_results()
        if not self.no_gui:
            # Create main Qt application
            app = QtWidgets.QApplication([])
            pg.setConfigOptions(antialias=True)

            plotter_object = Plotter(sim_data, self.window_title)
            flight_stick = PilotJoystick(self.sim_object.pause_or_unpause_sim)

        def update():
            """Update loop for simulation"""
            if not self.no_gui:
                x, y = flight_stick.get_joystick_pos()
                self.sim_object.update_manual_control(stick_x=x, stick_y=y)


            #for SITL stuff
            if self.use_sitl:
                try:
                    data, self.address = self.sock.recvfrom(100)
                    parse_format = 'HHI16H'
                    if len(data) != struct.calcsize(parse_format):
                        print(f"Bad packet size: {len(data)}")
                    decoded = struct.unpack(parse_format, data)
                    magic = 18458
                    if decoded[0] != magic:
                        print(f"Incorrect magic: {decoded[0]}")
                    frame_rate_hz = decoded[1]
                    frame_number = decoded[2]
                    pwm = decoded[3:]
                    
                    #TODO: if frame_rate_hz != RATE_HZ: ... RATE_HZ = frame_rate_hz
                    #TODO: reset logic
                    self.frame_count += 1
                except Exception:
                    time.sleep(0.01)
                    

                 

            sim_data = self.sim_object.update_real_time()



            if self.use_sitl:
                latest_data = sim_data[:, -1]
                phys_time = latest_data[slog.SDI_TIME]
                
                gyro = [latest_data[slog.SDI_P], latest_data[slog.SDI_Q], latest_data[slog.SDI_R]]
                accel = [latest_data[slog.SDI_AX], latest_data[slog.SDI_AY], latest_data[slog.SDI_AZ]]
                quat = [latest_data[slog.SDI_Q2], latest_data[slog.SDI_Q3], latest_data[slog.SDI_Q4], latest_data[slog.SDI_Q1]]
                pos = [latest_data[slog.SDI_DELTA_N], latest_data[slog.SDI_DELTA_E], latest_data[slog.SDI_ALT]]
                velo = [latest_data[slog.SDI_VN], latest_data[slog.SDI_VE], latest_data[slog.SDI_VD]]

                

                json_data = {
                    "timestamp": phys_time,
                    "imu": {
                        "gyro": gyro,
                        "accel_body": accel
                    },
                    "position": pos,
                    "quaternion": quat,
                    "velocity": velo
                }

                self.sock.sendto((json.dumps(json_data, separators=(',', ':')) + "\n").encode("ascii"), self.address)

            if not self.no_gui:
                plotter_object.update_plots(sim_data)



        timer = QtCore.QTimer()
        timer.timeout.connect(update)

        if not self.no_gui or self.hitl_active:
            update_loop_speed_ms = 10
            timer.start(update_loop_speed_ms)

        #if export_to_csv and not self.real_time:
        #    print("Exporting data to CSV")
        #    self.export_data(sim_data)


        if not self.no_gui or self.hitl_active:
            pg.exec()
        #if self.hitl_active: #shut down the HIL system
        #    try:
        #        self.control_unit.shut_down_hil()
        #        print("HITL system shut down")
        #    except SerialException:
        #        print("HITL shut down undisgracefully")

    def get_aircraft(self):
        """Returns aircraft object"""
        return self.aircraft


    def toggle_pause(self):
        """pause passthrough"""
        self.sim_object.pause_or_unpause_sim()
