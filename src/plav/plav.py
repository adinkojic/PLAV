"""
Implenting physics simulation by using an RungeKutta integrator at a fixed timestep
This one uses WGS84 and keeps tract of long lat in NED

Refactored 3 to OOO style

"""
import sys
import json
import time
import math
import numpy as np
#from scipy.integrate import solve_ivp
from numba import jit, float64
from PyQt6 import QtWidgets, QtCore
from pyqtgraph.Qt import QtCore
from matplotlib import colormaps
import pyqtgraph as pg
import pandas as pd
from flightgear_python.fg_if import FDMConnection
from serial.serialutil import SerialException

from plav.quaternion_math import from_euler
from plav.simulator import Simulator
from plav.generic_aircraft_config import AircraftConfig, init_aircraft
from src.plav.brgr_model import init_brgr, BRGRConfig
from plav.f16_model import F16_aircraft
from plav.atmosphere import Atmosphere
from plav.step_logging import SimDataLogger
from src.plav.f16_control import F16Control, tas_to_eas
from src.plav.f16_control_HITL import F16ControlHITL
from plav.joystick_reader import JoystickReader

#from pyqtgraph.Qt import QtWidgets


def init_state(long, lat, alt, velocity, bearing, elevation, roll, init_omega):
    """initalize the state"""

    init_pos = np.array([long,lat,alt])

    init_vel = velocity

    #first apply bearing stuff
    #roll pitch yaw
    init_ori_ned = from_euler(roll*math.pi/180,elevation*math.pi/180,bearing*math.pi/180)


    y0 = np.append(np.append(init_ori_ned, init_omega), np.append(init_pos, init_vel))
    return y0


code_start_time = time.perf_counter()

#load aircraft config
with open('aircraftConfigs/brgrDroneDrop.json', 'r') as file:
    modelparam = json.load(file)
file.close()

real_time = True
hitl_active = False
use_flight_gear = False
export_to_csv = True
t_span = np.array([0.0, 720.0])

control_unit = None
if modelparam['useF16']:
    print('Using F16')
    control_vector = np.array(modelparam['init_control'],'d')
    aircraft = F16_aircraft(control_vector)

    if modelparam["useSAS"] and not hitl_active:
        print('Using Software Autopilot')
        control_unit = F16Control(np.array(modelparam['commands'],'d'))
        stability_augmentation_on_disc, autopilot_on_disc = 1.0, 1.0
        control_unit.update_switches(stability_augmentation_on_disc, autopilot_on_disc)

    if modelparam["useSAS"] and hitl_active:
        print('Using HITL Autopilot')
        try:
            control_unit = F16ControlHITL(np.array(modelparam['commands'],'d'), 'COM5')
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

#control_unit = JoystickReader('COM6')

use_file_atmosphere = True
if use_file_atmosphere:
    wind_alt_profile       = np.array(modelparam['wind_alt_profile'], dtype='d')
    wind_speed_profile     = np.array(modelparam['wind_speed_profile'], dtype='d')
    wind_direction_profile = np.array(modelparam['wind_direction_profile'], dtype='d')
else:
    wind_alt_profile = np.array([0, 0], dtype='d')
    wind_speed_profile = np.array([0, 0], dtype='d')
    wind_direction_profile = np.array([0, 0], dtype='d')
#init atmosphere config
atmosphere = Atmosphere(wind_alt_profile,wind_speed_profile,wind_direction_profile)

inital_alt    = modelparam['init_alt']
init_velocity = modelparam['init_vel']
init_rte      = modelparam['init_rot']
init_ori   = np.array(modelparam['init_ori'], 'd')

init_long = modelparam['init_lon']
init_lat = modelparam['init_lat']

y0 = init_state(init_long, init_lat, inital_alt, init_velocity, bearing=init_ori[2], elevation=init_ori[1], roll=init_ori[0], init_omega=init_rte)

sim_object = Simulator(y0, t_span, aircraft, atmosphere, control_sys = control_unit, t_step=0.01)

# Create main Qt application
app = QtWidgets.QApplication([])
main_window = QtWidgets.QMainWindow()
realtime_window = QtWidgets.QMainWindow()
main_window.setWindowTitle(modelparam['title'])
realtime_window.setWindowTitle('Real Time Flying')
main_window.resize(1600, 900)

pg.setConfigOptions(antialias=True)

# Create a central widget and layout
central_widget = QtWidgets.QWidget()
instrument_widget = QtWidgets.QWidget()
main_layout = QtWidgets.QVBoxLayout()
controls_layout = QtWidgets.QVBoxLayout()
central_widget.setLayout(main_layout)
instrument_widget.setLayout(controls_layout)
main_window.setCentralWidget(central_widget)
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

#meters to feet
mtf = 39.37/12


print("Pumping Sim...")
#this ensures it's all compiled
try:
    sim_object.pump_sim()
except KeyboardInterrupt:
    print("Simulation compilation aborted")
    exit(0)

print("Compilation and file load took ", time.perf_counter()-code_start_time)

if real_time is False:
    sim_start_time = time.perf_counter()
    sim_object.run_sim()
    sim_end_time = time.perf_counter()

sim_data = sim_object.return_results()

long_lat_plot = plot_widget.addPlot(title="Long, Lat [deg] vs Time")
long_lat_plot.addLegend()
lon = long_lat_plot.plot(sim_data[0], sim_data[8]*180/math.pi, pen=(130,20,130), name="Longitude")
lat = long_lat_plot.plot(sim_data[0], sim_data[9]*180/math.pi, pen=(40,40,240), name="Latitiude")
long_lat_plot.addLegend(frame=True, colCount=2)


cm = pg.colormap.get('CET-L17')
cm.reverse()
pen0 = cm.getPen( span=(0.0,3e-6), width=2 )
path_plot = plot_widget.addPlot(title="Flight Path [long,lat]")
path_plot.addLegend()
path = path_plot.plot(sim_data[8]*180/math.pi, sim_data[9]*180/math.pi,pen =pen0, name="Path")


altitude_plot = plot_widget.addPlot(title="Altitude [ft] vs Time")
#altitude_plot.addLegend()
alt = altitude_plot.plot(sim_data[0], sim_data[10]*mtf,pen=(240,20,20), name="Altitude")

downrange_plot = plot_widget.addPlot(title="Downrange Distance [m] vs Time")
downrange = downrange_plot.plot(sim_data[0], sim_data[35],pen=(240,240,255),name="Downrange")

plot_widget.nextRow()

airspeed = plot_widget.addPlot(title="Airspeed [TAS] vs Time")
#airspeed.addLegend()
speed = airspeed.plot(sim_data[0], sim_data[30]*1.943844,pen=(40, 40, 180), name="airspeed")

flight_path_plot = plot_widget.addPlot(title="Flight Path angle vs Time")
flight_path = flight_path_plot.plot(sim_data[0], sim_data[34] * 180/math.pi,pen=(240,240,255),name="Flight Path")

alpha_beta = plot_widget.addPlot(title="Alpha Beta [deg] vs Time")
alpha_beta.addLegend()
alpha = alpha_beta.plot(sim_data[0], sim_data[31]*180/math.pi,pen=(200, 30, 40), name="Alpha")
beta = alpha_beta.plot(sim_data[0], sim_data[32]*180/math.pi,pen=(40, 30, 200), name="Beta")

range_cross_section = plot_widget.addPlot(title="Downrange [m] vs Altitude [m]")
rcs = range_cross_section.plot(sim_data[35], sim_data[10],pen=(255,255,255),name="Flight Cross Section")

plot_widget.nextRow()

velocity_plot = plot_widget.addPlot(title="Velocity [ft/s] vs Time [s]")
velocity_plot.addLegend()
vn = velocity_plot.plot(sim_data[0], sim_data[11]*mtf,pen=(240,20,20), name="Vn")
ve = velocity_plot.plot(sim_data[0], sim_data[12]*mtf,pen=(20,240,20), name="Ve")
vd = velocity_plot.plot(sim_data[0], sim_data[13]*mtf,pen=(240,20,240), name="Vd")

body_forces = plot_widget.addPlot(title="Body force [lbf] vs Time")
body_forces.addLegend()
fx = body_forces.plot(sim_data[0], sim_data[17] *0.2248089431, pen=(40, 40, 255), name="X")
fy = body_forces.plot(sim_data[0], sim_data[18] *0.2248089431, pen=(40, 255, 40), name="Y")
fz = body_forces.plot(sim_data[0], sim_data[19] *0.2248089431, pen=(255, 40, 40), name="Z")

body_moment = plot_widget.addPlot(title="Body Moment [ft lbf] vs Time")
body_moment.addLegend()
mx = body_moment.plot(sim_data[0], sim_data[20] *0.7375621493, pen=(40, 40, 180), name="X")
my = body_moment.plot(sim_data[0], sim_data[21] *0.7375621493, pen=(40, 180, 40), name="Y")
mz = body_moment.plot(sim_data[0], sim_data[22] *0.7375621493, pen=(180, 40, 40), name="Z")


body_rate_plot = plot_widget.addPlot(title="Body Rate [deg/s] vs Time [s]")
body_rate_plot.addLegend()
p = body_rate_plot.plot(sim_data[0], sim_data[5]*180/math.pi,pen=(240,240,20), name="p")
q = body_rate_plot.plot(sim_data[0], sim_data[6]*180/math.pi,pen=(20,240,240), name="q")
r = body_rate_plot.plot(sim_data[0], sim_data[7]*180/math.pi,pen=(240,20,240), name="r")

plot_widget.nextRow()



local_gravity = plot_widget.addPlot(title="Local Gravity [ft/s^2] vs Time")
gravity = local_gravity.plot(sim_data[0], sim_data[23] *mtf,pen=(10,130,20), name="Gravity")

air_density_plot = plot_widget.addPlot(title="Air density [kg/m^3] vs Time")
rho = air_density_plot.plot(sim_data[0], sim_data[27],pen=(20,5,130),name="Air Density")

air_pressure = plot_widget.addPlot(title="Air Pressusre [Pa] vs Time")
pressure = air_pressure.plot(sim_data[0], sim_data[28],pen=(120,5,20),name="Air Pressure")

reynolds_plot = plot_widget.addPlot(title="Reynolds Number vs Time")
re = reynolds_plot.plot(sim_data[0], sim_data[33],pen=(240,240,255),name="Reynolds Number")

plot_widget.nextRow()


quat_plot = plot_widget.addPlot(title="Rotation Quaternion vs Time")
quat_plot.addLegend()
q1 = quat_plot.plot(sim_data[0], sim_data[1],pen=(255,255,255),name="1")
q2 = quat_plot.plot(sim_data[0], sim_data[2],pen=(255,10,10),name="i")
q3 = quat_plot.plot(sim_data[0], sim_data[3],pen=(10,255,10),name="j")
q4 = quat_plot.plot(sim_data[0], sim_data[4],pen=(10,10,255),name="k")

euler_plot = plot_widget.addPlot(title="Euler Angles [deg] vs Time")
euler_plot.addLegend()
roll = euler_plot.plot(sim_data[0], sim_data[14] *180/math.pi,pen=(240,20,20), name="roll")
pitch = euler_plot.plot(sim_data[0], sim_data[15] *180/math.pi,pen=(120,240,20), name="pitch")
yaw = euler_plot.plot(sim_data[0], sim_data[16] *180/math.pi,pen=(120,20,240), name="yaw")

thrust_plot = plot_widget.addPlot(title="Thrust vs Time")
thrust = thrust_plot.plot(sim_data[0], sim_data[36],pen=(255,255,255),name="Thrust")

control_plot = plot_widget.addPlot(title="Control Surface Deflection [-1, 1] vs Time")
control_plot.addLegend()
rudder = control_plot.plot(sim_data[0], sim_data[37],pen=(10,10,255),name="Rudder")
aileron = control_plot.plot(sim_data[0], sim_data[38],pen=(255,10,10),name="Aileron")
elevator = control_plot.plot(sim_data[0], sim_data[39],pen=(10,255,10),name="Elevator")
throttle = control_plot.plot(sim_data[0], sim_data[40],pen=(255,255,255),name="Throttle")


def fdm_callback(fdm_data, event_pipe):
    """updates flight data for Flightgear"""

    current_pos = sim_object.latest_state()

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


def update():
    """Update loop for simulation"""
    global long_lat_plot, altitude_plot, path_plot, velocity_plot, \
        body_rate_plot, euler_plot, local_gravity, body_forces, \
        body_moment, airspeed, alpha_beta, quat_plot, air_density_plot, \
        air_pressure, reynolds_plot, sim_data, \
        lat, lon, alt, path, vn, ve, vd, p, q, r, roll, pitch, yaw, gravity,\
        fx, fy, fz, mx, my, mz, speed, alpha, beta, q1, q2, q3, q4, rho, pressure,\
        re, fdm_event_pipe
    
    x, y = joystick.getState()
    sim_object.update_manual_control(stick_x=x, stick_y=y)

    sim_data = sim_object.update_real_time()


    lon.setData(sim_data[0], sim_data[8]*180/math.pi)
    lat.setData(sim_data[0], sim_data[9]*180/math.pi)
    alt.setData(sim_data[0], sim_data[10]*mtf)
    path.setData(sim_data[8]*180/math.pi, sim_data[9]*180/math.pi)

    vn.setData(sim_data[0], sim_data[11]*mtf)
    ve.setData(sim_data[0], sim_data[12]*mtf)
    vd.setData(sim_data[0], sim_data[13]*mtf)

    p.setData(sim_data[0], sim_data[5]*180/math.pi)
    q.setData(sim_data[0], sim_data[6]*180/math.pi)
    r.setData(sim_data[0], sim_data[7]*180/math.pi)

    roll.setData(sim_data[0], sim_data[14] *180/math.pi)
    pitch.setData(sim_data[0], sim_data[15] *180/math.pi)
    yaw.setData(sim_data[0], sim_data[16] *180/math.pi)

    gravity.setData(sim_data[0], sim_data[23] *mtf)

    fx.setData(sim_data[0], sim_data[17] *0.2248089431)
    fy.setData(sim_data[0], sim_data[18] *0.2248089431)
    fz.setData(sim_data[0], sim_data[19] *0.2248089431)

    mx.setData(sim_data[0], sim_data[20] *0.7375621493)
    my.setData(sim_data[0], sim_data[21] *0.7375621493)
    mz.setData(sim_data[0], sim_data[22] *0.7375621493)

    speed.setData(sim_data[0], sim_data[30]*1.943844)
    alpha.setData(sim_data[0], sim_data[31]*180/math.pi)
    beta.setData(sim_data[0], sim_data[32]*180/math.pi)

    q1.setData(sim_data[0], sim_data[1])
    q2.setData(sim_data[0], sim_data[2])
    q3.setData(sim_data[0], sim_data[3])
    q4.setData(sim_data[0], sim_data[4])

    rho.setData(sim_data[0], sim_data[27])
    pressure.setData(sim_data[0], sim_data[28])
    re.setData(sim_data[0], sim_data[33])

    flight_path.setData(sim_data[0], sim_data[34] * 180/math.pi)
    downrange.setData(sim_data[0], sim_data[35])

    rcs.setData(sim_data[35], sim_data[10])
    thrust.setData(sim_data[0], sim_data[36],pen=(255,255,255))
    rudder.setData(sim_data[0], sim_data[37])
    aileron.setData(sim_data[0], sim_data[38])
    elevator.setData(sim_data[0], sim_data[39])
    throttle.setData(sim_data[0], sim_data[40])


timer = QtCore.QTimer()
timer.timeout.connect(update)



if not real_time:
    print("Sim took ", sim_end_time-sim_start_time)
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
timer.start(10)

if export_to_csv and not real_time:
    csv_data = {'time': sim_data[0],
        'altitudeMsl_ft': sim_data[10]*mtf,
        'longitude_deg': sim_data[8]*180/math.pi,
        'latitude_deg': sim_data[9]*180/math.pi,
        'localGravity_ft_s2': sim_data[23] *mtf,
        'eulerAngle_deg_Yaw':  sim_data[16] *180/math.pi,
        'eulerAngle_deg_Pitch': sim_data[15] *180/math.pi,
        'eulerAngle_deg_Roll' : sim_data[14] *180/math.pi,
        'aero_bodyForce_lbf_X': sim_data[17] *0.2248089431,
        'aero_bodyForce_lbf_Y': sim_data[18] *0.2248089431,
        'aero_bodyForce_lbf_Z': sim_data[19] *0.2248089431,
        'aero_bodyMoment_ftlbf_L': sim_data[20] *0.7375621493,
        'aero_bodyMoment_ftlbf_M': sim_data[21] *0.7375621493,
        'aero_bodyMoment_ftlbf_N': sim_data[22] *0.7375621493,
        'trueAirspeed_nmi_h': sim_data[30]*1.943844,
        'airDensity_slug_ft3': sim_data[27] *0.00194032,
        'downrageDistance_m': sim_data[35],
        }
    
    df = pd.DataFrame(csv_data)
    filename = "output.csv"
    df.to_csv(filename, index=False)

    print(f"Data exported to {filename}")

main_window.show()
realtime_window.show()
pg.exec()


if hitl_active: #shut down the HIL system
    try:
        control_unit.shut_down_hil()
        print("HITL system shut down")
    except SerialException:
        print("HITL shut down undisgracefully")
