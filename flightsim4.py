"""
Implenting physics simulation by using an RungeKutta integrator at a fixed timestep
This one uses WGS84 and keeps tract of long lat in NED

Refactored 3 to OOO style

"""

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

import quaternion_math as quat
from aircraftconfig import AircraftConfig, init_aircraft
from f16_model import F16_aircraft
#import ussa1976
from atmosphere import Atmosphere
from step_logging import SimDataLogger
from runge_kutta4 import basic_rk4

#from pyqtgraph.Qt import QtWidgets



@jit(float64(float64,float64))
def get_gravity(phi, h):
    """gets gravity accel from lat and altitude
    phi: latitude
    h: altitude"""
    graivty = 9.780327*(1 +5.3024e-3*np.sin(phi)**2 - 5.8e-6*np.sin(2*phi)**2) \
            - (3.0877e-6 - 4.4e-9*np.sin(phi)**2)*h + 7.2e-14*h**2
    return graivty

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

    atmosphere.update_conditions(altitude, time = t)

    air_density = atmosphere.get_density()
    air_temperature = atmosphere.get_temperature()
    static_pressure = atmosphere.get_pressure()
    speed_of_sound = atmosphere.get_speed_of_sound()

    #adds wind
    v_airspeed = quat.rotateVectorQ(q, np.array([vn, ve, vd], 'd') + atmosphere.get_wind_ned())
    #solving for acceleration, which is velocity_dot
    #aircraft_config.update_control(control_vect)
    aircraft_config.update_conditions(altitude,  v_airspeed, omega, air_density, air_temperature, speed_of_sound)


    aero_forces_body, aero_moments = aircraft_config.get_forces()
    thrust = aircraft_config.calculate_thrust()
    x_cp = aircraft_config.get_xcp()

    body_forces_body = np.array([
        aero_forces_body[0] + thrust,
        aero_forces_body[1],
        aero_forces_body[2],
        ], 'd')
    #torque from forces

    moments_with_torque = np.array([
        aero_moments[0] - x_cp[2]*body_forces_body[1] + x_cp[1]*body_forces_body[2],
        aero_moments[1] + x_cp[2]*body_forces_body[0] - x_cp[0]*body_forces_body[2],
        aero_moments[2] - x_cp[1]*body_forces_body[0] + x_cp[0]*body_forces_body[1],
    ], 'd')


    forces_ned = quat.rotateFrameQ(q, body_forces_body)

    ## figure out forces here
    #in NED frame
    accel_north = forces_ned[0]/mass
    accel_east  = forces_ned[1]/mass
    accel_down  = forces_ned[2]/mass

    omega = omega - quat.rotateFrameQ(q, omega_NI)

    #integrate state
    q1dot = 0.5*(-omega[0]*q[1] -omega[1]*q[2] -omega[2]*q[3])
    q2dot = 0.5*( omega[0]*q[0] +omega[2]*q[2] -omega[1]*q[3])
    q3dot = 0.5*( omega[1]*q[0] -omega[2]*q[1] +omega[0]*q[3])
    q4dot = 0.5*( omega[2]*q[0] +omega[1]*q[1] -omega[0]*q[2])

    #(11.27) in Engineeering Dyanmics (Kasdin and Paley)
    omega_dot = np.linalg.solve(inertia_tensor, moments_with_torque - np.cross(np.eye(3), omega) @ inertia_tensor @ omega)

    lat_dot = vn/(R_phi+altitude)
    long_dot = ve/((R_lamb+altitude)*np.cos(lat))
    altitude_dot = -vd

    #from book Optimal Estimation of Dynamic Systems
    vn_dot = -(long_dot + 2*omega_e)*ve*np.sin(lat) + vn*vd/(R_phi+altitude) + accel_north
    ve_dot = -(long_dot + 2*omega_e)*vn*np.sin(lat) + ve*vd/(R_phi+altitude) + 2*omega_e*vd*np.cos(lat)+accel_east
    vd_dot = -ve**2/(R_lamb+altitude)-vn**2/(R_phi+altitude) - 2*omega_e*ve*np.cos(lat) + gravity + accel_down



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

    rollpitchyaw=quat.to_euler(q)


    if log is not None:
        log.load_line(t, y, rollpitchyaw, aero_forces_body, \
                    aero_moments, gravity, speed_of_sound, mach ,dynamic_pressure, \
                    true_airspeed, air_density, static_pressure, air_temperature, \
                    alpha, beta, reynolds)

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
    def __init__(self, init_state, time_span, aircraft, atmosphere, t_step = 0.1):
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

        self.control_vec = None

        #log the inital state
        x_dot(self.time, self.state, aircraft, atmosphere, self.sim_log)
        self.sim_log.save_line()

    def advance_timestep(self):
        """advance timestep function, updates timestep and saves values"""

        self.time, self.state = basic_rk4(x_dot, self.time, self.t_step, self.state, args= (self.aircraft,self.atmosphere))
 
        #lat wrapparound probably doesnt work
        #if self.state[7] < -math.pi:
        #    self.state[7] = self.state[7] + math.pi
        #elif self.state[7] > math.pi:
        #    self.state[7] =self.state[7] - math.pi

        #lon wrapparound
        if self.state[8] < -math.pi:
            self.state[8] = self.state[8] + 2.0*math.pi
        elif self.state[8] > math.pi:
            self.state[8] = self.state[8] - 2.0*math.pi

        #get stuff
        x_dot(self.time, self.state, self.aircraft, self.atmosphere, self.sim_log)
        self.sim_log.save_line()

    def update_real_time(self, time_warp = 1.0):
        """Updates the real time sim, try to call with a delay in between"""
        if self.start_time is None:
            self.start_time = time.time()

        if self.paused:
            pass
        else:
            self.elapsed_time = (time.time() - self.start_time) * time_warp + self.time_at_last_pause
            while self.time < self.elapsed_time:
                self.advance_timestep()

        return self.return_results()
    
    def update_control_manual(self, pitch, roll):
        """pass in a control vector for the simulation"""
        joystick_command = np.array([0, roll, pitch, 0],'d')
        self.aircraft.update_control(joystick_command)
    
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
with open('aircraftConfigs/case16IDL.json', 'r') as file:
    modelparam = json.load(file)
file.close()

if modelparam['useF16']:
    control_vector = np.array(modelparam['init_control'],'d')
    aircraft = F16_aircraft(control_vector)
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
    init_ori = np.array([0.0, 0.0, 0.0])



y0 = init_state(init_x, init_y, inital_alt, init_velocity, bearing=init_ori[2], elevation=init_ori[1], roll=init_ori[0], init_omega=init_rte)

#pump sim once
basic_rk4(x_dot, 0.0, 0.01, y0, args= (aircraft,atmosphere))

real_time = False
t_span = np.array([0.0, 180.0])

sim_object = Simulator(y0, t_span, aircraft, atmosphere, t_step=0.01)

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

altitude_plot = plot_widget.addPlot(title="Altitude [ft] vs Time")
#altitude_plot.addLegend()
alt = altitude_plot.plot(sim_data[0], sim_data[10]*mtf,pen=(240,20,20), name="Altitude")

cm = pg.colormap.get('CET-L17')
cm.reverse()
pen0 = cm.getPen( span=(0.0,3e-6), width=2 )
path_plot = plot_widget.addPlot(title="Flight Path [long,lat]")
path_plot.addLegend()
path = path_plot.plot(sim_data[8]*180/math.pi, sim_data[9]*180/math.pi,pen =pen0, name="Path")

plot_widget.nextRow()

velocity_plot = plot_widget.addPlot(title="Velocity [ft/s] vs Time [s]")
velocity_plot.addLegend()
vn = velocity_plot.plot(sim_data[0], sim_data[11]*mtf,pen=(240,20,20), name="Vn")
ve = velocity_plot.plot(sim_data[0], sim_data[12]*mtf,pen=(20,240,20), name="Ve")
vd = velocity_plot.plot(sim_data[0], sim_data[13]*mtf,pen=(240,20,240), name="Vd")

body_rate_plot = plot_widget.addPlot(title="Body Rate [deg/s] vs Time [s]")
body_rate_plot.addLegend()
p = body_rate_plot.plot(sim_data[0], sim_data[5]*180/math.pi,pen=(240,240,20), name="p")
q = body_rate_plot.plot(sim_data[0], sim_data[6]*180/math.pi,pen=(20,240,240), name="q")
r = body_rate_plot.plot(sim_data[0], sim_data[7]*180/math.pi,pen=(240,20,240), name="r")

euler_plot = plot_widget.addPlot(title="Euler Angles [deg] vs Time")
euler_plot.addLegend()
roll = euler_plot.plot(sim_data[0], sim_data[14] *180/math.pi,pen=(240,20,20), name="roll")
pitch = euler_plot.plot(sim_data[0], sim_data[15] *180/math.pi,pen=(120,240,20), name="pitch")
yaw = euler_plot.plot(sim_data[0], sim_data[16] *180/math.pi,pen=(120,20,240), name="yaw")

plot_widget.nextRow()


local_gravity = plot_widget.addPlot(title="Local Gravity [ft/s^2] vs Time")
gravity = local_gravity.plot(sim_data[0], sim_data[23] *mtf,pen=(10,130,20), name="Gravity")

body_forces = plot_widget.addPlot(title="Body force [lbf] vs Time")
body_forces.addLegend()
fx = body_forces.plot(sim_data[0], sim_data[17] / 4.448, pen=(40, 40, 255), name="X")
fy = body_forces.plot(sim_data[0], sim_data[18] / 4.448, pen=(40, 255, 40), name="Y")
fz = body_forces.plot(sim_data[0], sim_data[19] / 4.448, pen=(255, 40, 40), name="Z")

body_moment = plot_widget.addPlot(title="Body Moment [ft lbf] vs Time")
body_moment.addLegend()
mx = body_moment.plot(sim_data[0], sim_data[20] / 1.356, pen=(40, 40, 180), name="X")
my = body_moment.plot(sim_data[0], sim_data[21] / 1.356, pen=(40, 180, 40), name="Y")
mz = body_moment.plot(sim_data[0], sim_data[22] / 1.356, pen=(180, 40, 40), name="Z")

plot_widget.nextRow()

airspeed = plot_widget.addPlot(title="Airspeed [TAS] vs Time")
#airspeed.addLegend()
speed = airspeed.plot(sim_data[0], sim_data[30]*1.943844,pen=(40, 40, 180), name="airspeed")

alpha_beta = plot_widget.addPlot(title="Alpha Beta [deg] vs Time")
alpha_beta.addLegend()
alpha = alpha_beta.plot(sim_data[0], sim_data[31]*180/math.pi,pen=(200, 30, 40), name="Alpha")
beta = alpha_beta.plot(sim_data[0], sim_data[32]*180/math.pi,pen=(40, 30, 200), name="Beta")

quat_plot = plot_widget.addPlot(title="Rotation Quaternion vs Time")
quat_plot.addLegend()
q1 = quat_plot.plot(sim_data[0], sim_data[1],pen=(255,255,255),name="1")
q2 = quat_plot.plot(sim_data[0], sim_data[2],pen=(255,10,10),name="i")
q3 = quat_plot.plot(sim_data[0], sim_data[3],pen=(10,255,10),name="j")
q4 = quat_plot.plot(sim_data[0], sim_data[4],pen=(10,10,255),name="k")

plot_widget.nextRow()

air_density_plot = plot_widget.addPlot(title="Air density [kg/m^3] vs Time")
rho = air_density_plot.plot(sim_data[0], sim_data[27],pen=(20,5,130),name="Air Density")

air_pressure = plot_widget.addPlot(title="Air Pressusre [Pa] vs Time")
pressure = air_pressure.plot(sim_data[0], sim_data[28],pen=(120,5,20),name="Air Pressure")

reynolds_plot = plot_widget.addPlot(title="Reynolds Number vs Time")
re = reynolds_plot.plot(sim_data[0], sim_data[33],pen=(240,240,255),name="Reynolds Number")

print("compilation took ", time.perf_counter()-code_start_time)

def update():
    global long_lat_plot, altitude_plot, path_plot, velocity_plot, \
        body_rate_plot, euler_plot, local_gravity, body_forces, \
        body_moment, airspeed, alpha_beta, quat_plot, air_density_plot, \
        air_pressure, reynolds_plot, sim_data, \
        lat, lon, alt, path, vn, ve, vd, p, q, r, roll, pitch, yaw, gravity,\
        fx, fy, fz, mx, my, mz, speed, alpha, beta, q1, q2, q3, q4, rho, pressure,\
        re
    
    x, y = joystick.getState()
    #sim_object.update_control_manual(roll=x, pitch=y)

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

    #unoptimized af gotta fix
    roll.setData(sim_data[0], sim_data[14] *180/math.pi)
    pitch.setData(sim_data[0], sim_data[15] *180/math.pi)
    yaw.setData(sim_data[0], sim_data[16] *180/math.pi)

    gravity.setData(sim_data[0], sim_data[23] *mtf)

    fx.setData(sim_data[0], sim_data[14] / 4.448)
    fy.setData(sim_data[0], sim_data[15] / 4.448)
    fz.setData(sim_data[0], sim_data[16] / 4.448)

    mx.setData(sim_data[0], sim_data[20] / 1.356)
    my.setData(sim_data[0], sim_data[21] / 1.356)
    mz.setData(sim_data[0], sim_data[22] / 1.356)

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


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10)
if not real_time:
    
    print("sim took ", sim_end_time-sim_start_time)

    print('data size: ', sim_data[0].size)
    print('final alt', sim_data[10][-1]*mtf)
    print('final lon', sim_data[9][-1]* 180/math.pi)


main_window.show()
realtime_window.show()
pg.exec()
