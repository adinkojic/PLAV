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

import quaternion_math as quat
from aircraftconfig import AircraftConfig, init_aircraft
from f16_model import F16_aircraft
#import ussa1976
from atmosphere import Atmosphere
from step_logging import SimDataLogger
from runge_kutta4 import basic_rk4
from f16Control import F16Control, tas_to_eas
from f16ControlHITL import F16ControlHITL

#from pyqtgraph.Qt import QtWidgets



@jit(float64(float64,float64))
def get_gravity(phi, h):
    """gets gravity accel from lat and altitude
    phi: latitude
    h: altitude"""
    gravity = 9.780327*(1 +5.3024e-3*np.sin(phi)**2 - 5.8e-6*np.sin(2*phi)**2) \
            - (3.0877e-6 - 4.4e-9*np.sin(phi)**2)*h + 7.2e-14*h**2
    return gravity

@jit
def x_dot(t, y, aircraft_config, atmosphere, log = None):
    """Implements standard NED equations
    [q1 q2 q3 q4], [p q r], (lambda) long, (phi)lat, alt, vn, ve, vd,
    q4 is the angle q13 is the axis """

    x_dot = np.zeros(13)
    q = np.array([y[0], y[1], y[2], y[3]])

    omega = np.array([ y[4], y[5], y[6]])

    long = y[7]
    lat = y[8]
    altitude = y[9]
    vn = y[10]
    ve = y[11]
    vd = y[12]

    mass = aircraft_config.get_mass()
    inertia_tensor = aircraft_config.get_inertia_matrix()

    a = 6378137.0 #earth semi-major axis
    e = 0.0818 #earth ecentricity
    omega_e = 7.292115e-5 #earth rotation rate
    R_phi = a*(1-e**2)/((1-e**2*np.sin(lat)))**(3/2)
    R_lamb = a/math.sqrt((1-e**2*np.sin(lat)))

    omega_NI = omega_e * np.array([np.cos(lat), 0, -np.sin(lat)]) + np.array( \
        [(ve)/(R_lamb + altitude), -(vn)/(R_phi + altitude), -(ve * np.tan(lat))/(R_lamb + altitude)])

    gravity = get_gravity(lat, altitude)

    atmosphere.update_conditions(altitude)

    air_density = atmosphere.get_density()
    air_temperature = atmosphere.get_temperature()
    static_pressure = atmosphere.get_pressure()
    speed_of_sound = atmosphere.get_speed_of_sound()

    #adds wind
    v_airspeed = quat.rotateVectorQ(q, np.array([vn, ve, vd], 'd') + atmosphere.get_wind_ned())
    #solving for acceleration, which is velocity_dot
    aircraft_config.update_conditions(altitude,  v_airspeed, omega, air_density, air_temperature, speed_of_sound)


    aero_forces_body, aero_moments = aircraft_config.get_forces()
    aircraft_thrust = aircraft_config.calculate_thrust()

    body_forces_body = np.array([
        aero_forces_body[0] + aircraft_thrust,
        aero_forces_body[1],
        aero_forces_body[2],
        ], 'd')

    accel_body = body_forces_body/mass

    accel_ned = quat.rotateFrameQ(q, accel_body)


    accel_north = accel_ned[0]
    accel_east  = accel_ned[1]
    accel_down  = accel_ned[2]

    omega = omega - quat.rotateFrameQ(q, omega_NI)

    #integrate state
    q1dot = 0.5*(-omega[0]*q[1] -omega[1]*q[2] -omega[2]*q[3])
    q2dot = 0.5*( omega[0]*q[0] +omega[2]*q[2] -omega[1]*q[3])
    q3dot = 0.5*( omega[1]*q[0] -omega[2]*q[1] +omega[0]*q[3])
    q4dot = 0.5*( omega[2]*q[0] +omega[1]*q[1] -omega[0]*q[2])

    #(11.27) in Engineeering Dyanmics (Kasdin and Paley)
    omega_dot = np.linalg.solve(inertia_tensor, aero_moments - np.cross(np.eye(3), omega) @ inertia_tensor @ omega)

    lat_dot = vn/(R_phi+altitude)
    long_dot = ve/((R_lamb+altitude)*np.cos(lat))
    altitude_dot = -vd

    #from book Optimal Estimation of Dynamic Systems
    vn_dot = accel_north-(long_dot + 2*omega_e)*ve*np.sin(lat) + vn*vd/(R_phi+altitude)
    ve_dot = accel_east -(long_dot + 2*omega_e)*vn*np.sin(lat) + ve*vd/(R_phi+altitude) + 2*omega_e*vd*np.cos(lat)
    vd_dot = accel_down + gravity-ve**2/(R_lamb+altitude)-vn**2/(R_phi+altitude) - 2*omega_e*ve*np.cos(lat)



    x_dot[0] = q1dot
    x_dot[1] = q2dot
    x_dot[2] = q3dot
    x_dot[3] = q4dot

    x_dot[4] = omega_dot[0]
    x_dot[5] = omega_dot[1]
    x_dot[6] = omega_dot[2]

    x_dot[7] = lat_dot
    x_dot[8] = long_dot
    x_dot[9] = altitude_dot

    x_dot[10] = vn_dot
    x_dot[11] = ve_dot
    x_dot[12] = vd_dot

    mach = aircraft_config.get_mach()
    dynamic_pressure = aircraft_config.get_qbar()
    true_airspeed = aircraft_config.get_airspeed()

    alpha = aircraft_config.get_alpha()
    beta  = aircraft_config.get_beta()
    reynolds = aircraft_config.get_reynolds()

    control_deflection = aircraft_config.get_control_deflection()

    if log is not None:
        log.load_line(t, y, aero_forces_body, \
                    aero_moments, gravity, speed_of_sound, mach ,dynamic_pressure, \
                    true_airspeed, air_density, static_pressure, air_temperature, \
                    alpha, beta, reynolds, aircraft_thrust, control_deflection)

    return x_dot

def init_state(lat, lon, alt, velocity, bearing, elevation, roll, init_omega):
    """initalize the state"""


    init_pos = np.array([lat,lon,alt])

    init_vel = velocity

    #first apply bearing stuff
    #roll pitch yaw
    init_ori_ned = quat.from_euler(roll*math.pi/180,elevation*math.pi/180,bearing*math.pi/180)


    y0 = np.append(np.append(init_ori_ned, init_omega), np.append(init_pos, init_vel))
    return y0

class Simulator(object):
    """A sim object is required to store all the required data nicely."""
    def __init__(self, init_state, time_span, aircraft, atmosphere, control_sys= None,t_step = 0.1):
        self.state = init_state
        self.t_span = time_span
        self.time = time_span[0]
        self.sim_log = SimDataLogger(preallocated=1.1*(time_span[1]-time_span[0]) / t_step)
        self.t_step = t_step
        self.aircraft = aircraft
        self.atmosphere = atmosphere
        self.start_time = None

        self.paused = True
        self.elapsed_time = 0.0
        self.time_at_last_pause = 0.0

        self.pilot_vec = np.zeros(4, 'd')
        self.control_sys = control_sys

        #log the inital state
        x_dot(self.time, self.state, aircraft, atmosphere, self.sim_log)
        self.sim_log.save_line()

    def advance_timestep(self):
        """advance timestep function, updates timestep and saves values"""

        if self.control_sys is not None:
            #by now the HIL should have a response ready
            #for HIL it might block a bit as the aurdino computes
            total_control_vector = self.control_sys_request_response()
            aircraft.update_control(total_control_vector)
                
        self.time, self.state = basic_rk4(x_dot, self.time, self.t_step, self.state,\
                                           args= (self.aircraft,self.atmosphere))
 
        #lon wrapparound
        if self.state[8] < -math.pi:
            self.state[8] = self.state[8] + 2.0*math.pi
        elif self.state[8] > math.pi:
            self.state[8] = self.state[8] - 2.0*math.pi

        #get stuff
        x_dot(self.time, self.state, self.aircraft, self.atmosphere, self.sim_log)
        self.sim_log.save_line()

        if self.control_sys is not None:
            #tell the control system to update now in case its a HIL system so it has time
            self.control_sys_update()

    def control_sys_update(self):
        """updates the control system with the latest data"""
        last_line = self.sim_log.get_lastest()

        if last_line is not None:
            sim_time = last_line[0]
            tas = last_line[30]*1.943844
            density = last_line[27]
            equivalent_airspeed = tas_to_eas(tas, density)


            altitude_msl = last_line[10] * 3.28084
            angle_of_attack = last_line[31] * 180/math.pi
            angle_of_sideslip = last_line[32] * 180/math.pi
            euler_angle_roll = last_line[14] * 180/math.pi
            euler_angle_pitch = last_line[15] * 180/math.pi
            euler_angle_yaw = last_line[16] * 180/math.pi
            body_angular_rate_roll = last_line[5]
            body_angular_rate_pitch = last_line[6]
            body_angular_rate_yaw = last_line[7]

            #print("altitude msl: ", altitude_msl)
            #print("equivalent airspeed: ", equivalent_airspeed)
            #print("angle of attack: ", angle_of_attack)
            #print("angle of sideslip: ", angle_of_sideslip)
            #print("euler angle roll: ", euler_angle_roll)
            #print("euler angle pitch: ", euler_angle_pitch)
            #print("euler angle yaw: ", euler_angle_yaw)
            #print("body angular rate roll: ", body_angular_rate_roll)
            #print("body angular rate pitch: ", body_angular_rate_pitch)
            #print("body angular rate yaw: ", body_angular_rate_yaw)
            #print("tas: ", tas)
            #print("density: ", density)

            self.control_sys.update_enviroment(altitude_msl, equivalent_airspeed, angle_of_attack, \
                    angle_of_sideslip, euler_angle_roll, euler_angle_pitch, \
                    euler_angle_yaw, body_angular_rate_roll ,\
                    body_angular_rate_pitch, body_angular_rate_yaw, sim_time)
 
            pilot_control_lat = 0.0 #self.pilot_vec[0]
            pilot_control_yaw = 0.0 #self.pilot_vec[1]
            pilot_control_long = 0.0 #self.pilot_vec[2]
            pilot_control_throttle = 0.0 #self.pilot_vec[3]
            self.control_sys.update_pilot_control(pilot_control_long, pilot_control_lat, \
                        pilot_control_yaw, pilot_control_throttle)

    def control_sys_request_response(self):
        """request a reponse from the control system, which should have a response ready"""
        control_vec = self.control_sys.get_control_output()
        return control_vec

    def latest_state(self):
        """returns the most recent state
        in lat [rad], lon [rad], alt [m], psi [rad], theta [rad], phi[rad]"""
        lat_lon_alt = self.state[7:10]
        psi_theta_phi = quat.to_euler(self.state[0:4])

        return np.concatenate((lat_lon_alt,psi_theta_phi))

    def update_real_time(self, time_warp = 1.0):
        """Updates the real time sim, try to call with a delay in between"""
        if self.start_time is None:
            self.start_time = time.time()

        if self.paused:
            pass
        else:
            self.elapsed_time = (time.time() - self.start_time) * time_warp +self.time_at_last_pause
            while self.time < self.elapsed_time:
                self.advance_timestep()
        return self.return_results()

    def update_manual_control(self, stick_x, stick_y):
        """pass in a control vector for the simulation"""
        command = np.array([0.0, stick_x, stick_y, 0.0],'d')
        self.pilot_vec = command

    def pause_sim(self):
        """Pauses the sim, saving time at stop"""
        if not self.paused:
            self.paused = True
            self.time_at_last_pause = self.elapsed_time

    def unpause_sim(self):
        """Unpauses sim, starts counting time again"""
        if self.paused:
            self.paused = False
            self.start_time = time.time()

    def pause_or_unpause_sim(self):
        """Flips state of sim"""
        if self.paused:
            self.unpause_sim()
        else:
            self.pause_sim()

    def run_sim(self):
        """runs the sim until t_span"""
        while self.time < self.t_span[1]:
            self.advance_timestep()
        self.time_at_last_pause = self.t_span[1]

    def return_results(self):
        """logger"""
        return self.sim_log.return_data()

    def return_time_steps(self):
        """returns number of timesteps saved"""
        return self.sim_log.return_data_size()


code_start_time = time.perf_counter()

#load aircraft config
with open('aircraftConfigs/case13.json', 'r') as file:
    modelparam = json.load(file)
file.close()

real_time = False
hitl_active = False
use_flight_gear = False
export_to_csv = True
t_span = np.array([0.0, 30.0])

control_unit = None
if modelparam['useF16']:
    control_vector = np.array(modelparam['init_control'],'d')
    aircraft = F16_aircraft(control_vector)

    if modelparam["useSAS"] and not hitl_active:
        print('using SAS')
        control_unit = F16Control(np.array(modelparam['commands'],'d'))
        stability_augmentation_on_disc, autopilot_on_disc = 1.0, 1.0
        control_unit.update_switches(stability_augmentation_on_disc, autopilot_on_disc)

    if modelparam["useSAS"] and hitl_active:
        print('using HITL')
        control_unit = F16ControlHITL(np.array(modelparam['commands'],'d'), 'COM3')
        stability_augmentation_on_disc, autopilot_on_disc = 1.0, 1.0
        control_unit.update_switches(stability_augmentation_on_disc, autopilot_on_disc)
else:
    aircraft = init_aircraft(modelparam)

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

use_file_init_conditions = True
if use_file_init_conditions:
    inital_alt    = modelparam['init_alt']
    init_velocity = modelparam['init_vel']
    init_rte      = modelparam['init_rot']
    init_ori   = np.array(modelparam['init_ori'], 'd')

    init_x = modelparam['init_lat']
    init_y = modelparam['init_lon']

else:
    inital_alt = 9144
    init_x = 0
    init_y = 0

    init_airspeed = 20 #meters per second
    init_alpha = 0 #degrees
    init_beta  = 0
    #init_velocity = aero.from_alpha_beta(init_airspeed, init_alpha, init_beta)
    init_velocity = [0.0, 0.0, 0.0]
    init_rte = np.array([0.0, 0.0, 0.0], dtype='d')
    init_ori = np.array([0.0, 0.0, 0.0], 'd')



y0 = init_state(init_x, init_y, inital_alt, init_velocity, bearing=init_ori[2], elevation=init_ori[1], roll=init_ori[0], init_omega=init_rte)

#pump sim once
basic_rk4(x_dot, 0.0, 0.01, y0, args= (aircraft,atmosphere, None))



sim_object = Simulator(y0, t_span, aircraft, atmosphere, control_sys = control_unit, t_step=0.1)

print("Sim started...")

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

if real_time is False:
    sim_start_time = time.perf_counter()
    sim_object.run_sim()
    sim_end_time = time.perf_counter()

sim_data = sim_object.return_results()
print(sim_object.return_time_steps())
print(np.size(sim_data))

long_lat_plot = plot_widget.addPlot(title="Long Lat [deg] vs Time")
long_lat_plot.addLegend()
lat = long_lat_plot.plot(sim_data[0], sim_data[8]*180/math.pi, pen=(40,40,240), name="Latitiude")
lon = long_lat_plot.plot(sim_data[0], sim_data[9]*180/math.pi, pen=(130,20,130), name="Longitude")
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


print("compilation took ", time.perf_counter()-code_start_time)

def fdm_callback(fdm_data, event_pipe):
    """updates flight data for Flightgear"""

    current_pos = sim_object.latest_state()

    fdm_data.lat_rad = current_pos[0]
    fdm_data.lon_rad = current_pos[1]
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

    lat.setData(sim_data[0], sim_data[8]*180/math.pi)
    lon.setData(sim_data[0], sim_data[9]*180/math.pi)
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
    print("sim took ", sim_end_time-sim_start_time)

    print('data size: ', sim_data[0].size)
    print('final alt', sim_data[10][-1]*mtf)
    print('final lon', sim_data[9][-1]* 180/math.pi)

print("size of sim_data: ", sys.getsizeof(sim_data))

fdm_event_pipe = None
if __name__ == '__main__' and use_flight_gear:  # NOTE: This is REQUIRED on Windows!
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
        'longitude_deg': sim_data[9]*180/math.pi,
        'latitude_deg': sim_data[8]*180/math.pi,
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
