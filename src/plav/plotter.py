"""Plotter module for PLAV simulator.
Based on PyQT"""

import math

from PyQt6 import QtWidgets, QtCore
from pyqtgraph.Qt import QtCore
from matplotlib import colormaps
import pyqtgraph as pg
from scipy import signal

import plav.conversions as conv
import plav.step_logging as slog # for log data indices


class Plotter(QtWidgets.QMainWindow):
    """Plotter class for PLAV simulator."""

    def __init__(self, sim_data, title="PLAV Simulator Plotter"):
        super().__init__()
        self.sim_data = sim_data

        # Initialize all plot variables to None so they can be updated later
        self.lon = None
        self.lat = None
        self.alt = None
        self.path = None
        self.vn = None
        self.ve = None
        self.vd = None
        self.p = None
        self.q = None
        self.r = None
        self.roll = None
        self.pitch = None
        self.yaw = None
        self.gravity = None
        self.fx = None
        self.fy = None
        self.fz = None
        self.mx = None
        self.my = None
        self.mz = None
        self.speed = None
        self.alpha = None
        self.beta = None
        self.rcs = None
        self.q1 = None
        self.q2 = None
        self.q3 = None
        self.q4 = None
        self.rho = None
        self.pressure = None
        self.re = None
        self.thrust = None
        self.rudder = None
        self.aileron = None
        self.elevator = None
        self.throttle = None
        self.downrange = None
        self.flight_path = None

        self.init_UI(title)

    def init_UI(self, title):
        """Initialize the UI components."""
        self.setWindowTitle(title.strip())
        self.resize(1600, 900)

        # Create a central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # Create a layout
        layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(layout)

        pg.setConfigOptions(antialias=True)

        # Create plot area using pyqtgraph GraphicsLayoutWidget
        plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(plot_widget)

        self.add_plots(plot_widget)
        self.update_plots(self.sim_data)
        # Add more UI components as needed

        self.show()

    def add_plots(self, plot_widget):
        """Add the inital plots to the plot widget."""
        long_lat_plot = plot_widget.addPlot(title="Long, Lat [deg] vs Time")
        long_lat_plot.addLegend()
        self.lon = long_lat_plot.plot(self.sim_data[slog.SDI_TIME],
                                 self.sim_data[slog.SDI_LONG]*conv.RAD_TO_DEG,
                                 pen=(130,20,130), name="Longitude")
        self.lat = long_lat_plot.plot(self.sim_data[slog.SDI_TIME],
                                 self.sim_data[slog.SDI_LAT]*conv.RAD_TO_DEG,
                                 pen=(40,40,240), name="Latitiude")
        long_lat_plot.addLegend(frame=True, colCount=2)


        cm = pg.colormap.get('CET-L17')
        cm.reverse()
        pen0 = cm.getPen( span=(0.0,3e-6), width=2 )
        path_plot = plot_widget.addPlot(title="Flight Path [long,lat]")
        path_plot.addLegend()
        self.path = path_plot.plot(self.sim_data[slog.SDI_LONG]*conv.RAD_TO_DEG,
                              self.sim_data[slog.SDI_LAT]*conv.RAD_TO_DEG,
                              pen =pen0, name="Path")


        altitude_plot = plot_widget.addPlot(title="Altitude [m] vs Time")
        #altitude_plot.addLegend()
        self.alt = altitude_plot.plot(self.sim_data[slog.SDI_TIME],
                                 self.sim_data[slog.SDI_ALT],
                                 pen=(240,20,20), name="Altitude")

        downrange_plot = plot_widget.addPlot(title="Downrange Distance [m] vs Time")
        self.downrange = downrange_plot.plot(self.sim_data[slog.SDI_TIME],
                                        self.sim_data[slog.SDI_DOWNRANGE],
                                        pen=(240,240,255),name="Downrange")

        plot_widget.nextRow()

        airspeed = plot_widget.addPlot(title="Airspeed [TAS] vs Time")
        #airspeed.addLegend()
        self.speed = airspeed.plot(self.sim_data[slog.SDI_TIME],
                              self.sim_data[slog.SDI_TAS]*conv.MPS_TO_KTS,
                              pen=(40, 40, 180), name="airspeed")

        flight_path_plot = plot_widget.addPlot(title="Flight Path angle vs Time")
        self.flight_path = flight_path_plot.plot(self.sim_data[slog.SDI_TIME],
                                             self.sim_data[slog.SDI_FLIGHT_PATH]*conv.RAD_TO_DEG,
                                             pen=(240,240,255),name="Flight Path")

        alpha_beta = plot_widget.addPlot(title="Alpha Beta [deg] vs Time")
        alpha_beta.addLegend()
        self.alpha = alpha_beta.plot(self.sim_data[slog.SDI_TIME],
                                self.sim_data[slog.SDI_ALPHA]*conv.RAD_TO_DEG,
                                pen=(200, 30, 40), name="Alpha")
        self.beta = alpha_beta.plot(self.sim_data[slog.SDI_TIME],
                               self.sim_data[slog.SDI_BETA]*conv.RAD_TO_DEG,
                               pen=(40, 30, 200), name="Beta")

        range_cross_section = plot_widget.addPlot(title="Downrange [m] vs Altitude [m]")
        self.rcs = range_cross_section.plot(self.sim_data[slog.SDI_DOWNRANGE],
                                       self.sim_data[slog.SDI_ALT],
                                       pen=(255,255,255),name="Flight Cross Section")

        plot_widget.nextRow()

        velocity_plot = plot_widget.addPlot(title="Velocity [m/s] vs Time [s]")
        velocity_plot.addLegend()
        self.vn = velocity_plot.plot(self.sim_data[slog.SDI_TIME],
                                self.sim_data[slog.SDI_VN],
                                pen=(240,20,20), name="Vn")
        self.ve = velocity_plot.plot(self.sim_data[slog.SDI_TIME],
                                self.sim_data[slog.SDI_VE],
                                pen=(20,240,20), name="Ve")
        self.vd = velocity_plot.plot(self.sim_data[slog.SDI_TIME],
                                self.sim_data[slog.SDI_VD],
                                pen=(240,20,240), name="Vd")

        body_forces = plot_widget.addPlot(title="Body force [N] vs Time")
        body_forces.addLegend()
        self.fx = body_forces.plot(self.sim_data[slog.SDI_TIME],
                              self.sim_data[slog.SDI_FX],
                              pen=(40, 40, 255), name="X")
        self.fy = body_forces.plot(self.sim_data[slog.SDI_TIME],
                              self.sim_data[slog.SDI_FY],
                              pen=(40, 255, 40), name="Y")
        self.fz = body_forces.plot(self.sim_data[slog.SDI_TIME],
                              self.sim_data[slog.SDI_FZ],
                              pen=(255, 40, 40), name="Z")

        body_moment = plot_widget.addPlot(title="Body Moment [N m] vs Time")
        body_moment.addLegend()
        self.mx = body_moment.plot(self.sim_data[slog.SDI_TIME],
                              self.sim_data[slog.SDI_MX],
                              pen=(40, 40, 180), name="X")
        self.my = body_moment.plot(self.sim_data[slog.SDI_TIME],
                              self.sim_data[slog.SDI_MY],
                              pen=(40, 180, 40), name="Y")
        self.mz = body_moment.plot(self.sim_data[slog.SDI_TIME],
                              self.sim_data[slog.SDI_MZ],
                              pen=(180, 40, 40), name="Z")


        body_rate_plot = plot_widget.addPlot(title="Body Rate [deg/s] vs Time [s]")
        body_rate_plot.addLegend()
        self.p = body_rate_plot.plot(self.sim_data[slog.SDI_TIME],
                                self.sim_data[slog.SDI_P]*conv.RAD_TO_DEG,
                                pen=(240,240,20), name="p")
        self.q = body_rate_plot.plot(self.sim_data[slog.SDI_TIME],
                                self.sim_data[slog.SDI_Q]*conv.RAD_TO_DEG,
                                pen=(20,240,240), name="q")
        self.r = body_rate_plot.plot(self.sim_data[slog.SDI_TIME],
                                self.sim_data[slog.SDI_R]*conv.RAD_TO_DEG,
                                pen=(240,20,240), name="r")

        plot_widget.nextRow()

        local_gravity = plot_widget.addPlot(title="Local Gravity [m/s^2] vs Time")
        self.gravity = local_gravity.plot(self.sim_data[slog.SDI_TIME],
                                     self.sim_data[slog.SDI_GRAVITY],
                                     pen=(10,130,20), name="Gravity")

        air_density_plot = plot_widget.addPlot(title="Air density [kg/m^3] vs Time")
        self.rho = air_density_plot.plot(self.sim_data[slog.SDI_TIME],
                                    self.sim_data[slog.SDI_AIR_DENSITY],
                                    pen=(20,5,130),name="Air Density")

        air_pressure = plot_widget.addPlot(title="Air Pressure [Pa] vs Time")
        self.pressure = air_pressure.plot(self.sim_data[slog.SDI_TIME],
                                     self.sim_data[slog.SDI_AIR_PRESSURE],
                                     pen=(120,5,20),name="Air Pressure")

        reynolds_plot = plot_widget.addPlot(title="Reynolds Number vs Time")
        self.re = reynolds_plot.plot(self.sim_data[slog.SDI_TIME],
                                self.sim_data[slog.SDI_REYNOLDS],
                                pen=(240,240,255),name="Reynolds Number")

        plot_widget.nextRow()


        quat_plot = plot_widget.addPlot(title="Rotation Quaternion vs Time")
        quat_plot.addLegend()
        self.q1 = quat_plot.plot(self.sim_data[slog.SDI_TIME],
                            self.sim_data[slog.SDI_Q1],
                            pen=(255,255,255),name="1")
        self.q2 = quat_plot.plot(self.sim_data[slog.SDI_TIME],
                            self.sim_data[slog.SDI_Q2],
                            pen=(255,10,10),name="i")
        self.q3 = quat_plot.plot(self.sim_data[slog.SDI_TIME],
                            self.sim_data[slog.SDI_Q3],
                            pen=(10,255,10),name="j")
        self.q4 = quat_plot.plot(self.sim_data[slog.SDI_TIME],
                            self.sim_data[slog.SDI_Q4],
                            pen=(10,10,255),name="k")

        euler_plot = plot_widget.addPlot(title="Euler Angles [deg] vs Time")
        euler_plot.addLegend()
        self.roll = euler_plot.plot(self.sim_data[slog.SDI_TIME],
                               self.sim_data[slog.SDI_ROLL] *conv.RAD_TO_DEG,
                               pen=(240,20,20), name="roll")
        self.pitch = euler_plot.plot(self.sim_data[slog.SDI_TIME],
                                self.sim_data[slog.SDI_PITCH] *conv.RAD_TO_DEG,
                                pen=(120,240,20), name="pitch")
        self.yaw = euler_plot.plot(self.sim_data[slog.SDI_TIME],
                              self.sim_data[slog.SDI_YAW] *conv.RAD_TO_DEG,
                              pen=(120,20,240), name="yaw")

        thrust_plot = plot_widget.addPlot(title="Thrust vs Time")
        self.thrust = thrust_plot.plot(self.sim_data[slog.SDI_TIME],
                                  self.sim_data[slog.SDI_THRUST],
                                  pen=(255,255,255),name="Thrust")

        control_plot = plot_widget.addPlot(title="Control Surface Deflection [-1, 1] vs Time")
        control_plot.addLegend()
        self.rudder = control_plot.plot(self.sim_data[slog.SDI_TIME],
                                   self.sim_data[slog.SDI_RUDDER_CMD],
                                   pen=(10,10,255),name="Rudder")
        self.aileron = control_plot.plot(self.sim_data[slog.SDI_TIME],
                                    self.sim_data[slog.SDI_AILERON_CMD],
                                    pen=(255,10,10),name="Aileron")
        self.elevator = control_plot.plot(self.sim_data[slog.SDI_TIME],
                                     self.sim_data[slog.SDI_ELEVATOR_CMD],
                                     pen=(10,255,10),name="Elevator")
        self.throttle = control_plot.plot(self.sim_data[slog.SDI_TIME],
                                     self.sim_data[slog.SDI_THRUST_CMD],
                                     pen=(255,255,255),name="Throttle")

    def update_plots(self, new_sim_data):
        """Set new simulation data and update plots."""
        self.sim_data = new_sim_data


        # Update all plot data
        self.lon.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_LONG]*conv.RAD_TO_DEG)
        self.lat.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_LAT]*conv.RAD_TO_DEG)
        self.alt.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_ALT])
        self.path.setData(self.sim_data[slog.SDI_LONG]*conv.RAD_TO_DEG,
                          self.sim_data[slog.SDI_LAT]*conv.RAD_TO_DEG)

        self.vn.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_VN])
        self.ve.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_VE])
        self.vd.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_VD])

        self.p.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_P]*conv.RAD_TO_DEG)
        self.q.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_Q]*conv.RAD_TO_DEG)
        self.r.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_R]*conv.RAD_TO_DEG)

        self.roll.setData(self.sim_data[slog.SDI_TIME],
                          self.sim_data[slog.SDI_ROLL]*conv.RAD_TO_DEG)
        self.pitch.setData(self.sim_data[slog.SDI_TIME],
                           self.sim_data[slog.SDI_PITCH]*conv.RAD_TO_DEG)
        self.yaw.setData(self.sim_data[slog.SDI_TIME],
                         self.sim_data[slog.SDI_YAW]*conv.RAD_TO_DEG)

        self.gravity.setData(self.sim_data[slog.SDI_TIME],
                             self.sim_data[slog.SDI_GRAVITY])

        self.fx.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_FX])
        self.fy.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_FY])
        self.fz.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_FZ])

        self.mx.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_MX])
        self.my.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_MY])
        self.mz.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_MZ])

        self.speed.setData(self.sim_data[slog.SDI_TIME],
                           self.sim_data[slog.SDI_TAS]*conv.MPS_TO_KTS)
        self.alpha.setData(self.sim_data[slog.SDI_TIME],
                           self.sim_data[slog.SDI_ALPHA]*conv.RAD_TO_DEG)
        self.beta.setData(self.sim_data[slog.SDI_TIME],
                          self.sim_data[slog.SDI_BETA]*conv.RAD_TO_DEG)

        self.q1.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_Q1])
        self.q2.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_Q2])
        self.q3.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_Q3])
        self.q4.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_Q4])

        self.rho.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_AIR_DENSITY])
        self.pressure.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_AIR_PRESSURE])
        self.re.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_REYNOLDS])

        self.flight_path.setData(self.sim_data[slog.SDI_TIME],
                                 self.sim_data[slog.SDI_FLIGHT_PATH]*conv.RAD_TO_DEG)
        self.downrange.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_DOWNRANGE])

        self.rcs.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_DOWNRANGE])
        self.thrust.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_THRUST],
                            pen=(255,255,255))
        self.rudder.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_RUDDER_CMD])
        self.aileron.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_AILERON_CMD])
        self.elevator.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_ELEVATOR_CMD])
        self.throttle.setData(self.sim_data[slog.SDI_TIME], self.sim_data[slog.SDI_THRUST_CMD])
