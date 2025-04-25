"""F-16 Control Scheme, implementing F16_control.dml"""

import numpy as np
from numba import jit, float64
from numba.experimental import jitclass

spec = [
    ('trim_rdr', float64),
    ('trim_ail', float64),
    ('trim_el', float64),
    ('trim_power', float64),

    ('rdr', float64),
    ('ail', float64),
    ('el', float64),
    ('power', float64),

    ('pilot_control_throttle', float64),
    ('pilot_control_long', float64),
    ('pilot_control_lat', float64),
    ('pilot_control_yaw', float64),

    ('stability_augmentation_on_disc', float64),
    ('autopilot_on_disc', float64),

    ('equivalent_airspeed_command', float64),
    ('altitude_msl_command', float64),
    ('lateral_deviation_error', float64),
    ('true_base_course_command', float64),

    ('altitude_msl', float64),
    ('equivalent_airspeed', float64),
    ('angle_of_attack', float64),
    ('angle_of_sideslip', float64),
    ('euler_angle_roll', float64),
    ('euler_angle_pitch', float64),
    ('euler_angle_yaw', float64),
    ('body_angular_rate_roll', float64),
    ('body_angular_rate_pitch', float64),
    ('body_angular_rate_yaw', float64),

    ('time', float64),

    ('commands_list', float64[:,:]),
]

#@jitclass(spec)
class F16Control(object):
    """F-16 Control Sysem"""
    def __init__(self, commands):
        self.trim_rdr   = 0.0
        self.trim_ail   = 0.0
        self.trim_el    = -3.2410
        self.trim_power = 13.9019

        self.rdr   = 0.0
        self.ail   = 0.0
        self.el    = 0.0
        self.power = 0.0

        self.pilot_control_throttle = 0.0
        self.pilot_control_long = 0.0
        self.pilot_control_lat = 0.0
        self.pilot_control_yaw = 0.0

        self.stability_augmentation_on_disc = 0.0
        self.autopilot_on_disc = 0.0

        self.equivalent_airspeed_command = 0.0
        self.altitude_msl_command =0.0
        self.lateral_deviation_error = 0.0
        self.true_base_course_command = 0.0

        self.altitude_msl = 0.0
        self.equivalent_airspeed = 0.0
        self.angle_of_attack = 0.0
        self.angle_of_sideslip = 0.0
        self.euler_angle_roll = 0.0
        self.euler_angle_pitch = 0.0
        self.euler_angle_yaw = 0.0
        self.body_angular_rate_roll = 0.0
        self.body_angular_rate_pitch = 0.0
        self.body_angular_rate_yaw = 0.0

        self.time = 0.0

        self.commands_list = commands

    def get_control_output(self):
        """computes and gets the control output in -1 to 1 range"""

        #this is relevant to the nasa cases
        self.check_commands()

        control = self.compute_control()
        self.ail = control[0] / 20.0 - self.trim_ail / 20.0
        self.rdr = control[1] / 30.0 - self.trim_rdr / 30.0
        self.el = control[2] / 25.0 -  self.trim_el / 25.0
        self.power = control[3] / 100.0 - self.trim_power / 100.0

        return np.array([self.rdr, self.ail, self.el, self.power], 'd')

    def check_commands(self):
        """for the NASA check cases, execute the latest command when it's its time"""
        for i in range(self.commands_list.shape[0]):
            if self.time >= self.commands_list[i, 0]:
                eas_command = self.commands_list[i, 1]
                alt_command = self.commands_list[i, 2]
                lat_command = self.commands_list[i, 3]
                yaw_command = self.commands_list[i, 4]

                self.update_commands(eas_command, alt_command, lat_command, yaw_command)

    def update_enviroment(self, altitude_msl, equivalent_airspeed, angle_of_attack, \
                        angle_of_sideslip, euler_angle_roll, euler_angle_pitch, \
                        euler_angle_yaw, body_angular_rate_roll ,\
                        body_angular_rate_pitch, body_angular_rate_yaw, time):
        """Update the enviroment variables"""
        self.altitude_msl = altitude_msl
        self.equivalent_airspeed = equivalent_airspeed
        self.angle_of_attack = angle_of_attack
        self.angle_of_sideslip = angle_of_sideslip
        self.euler_angle_roll = euler_angle_roll
        self.euler_angle_pitch = euler_angle_pitch
        self.euler_angle_yaw = euler_angle_yaw
        self.body_angular_rate_roll = body_angular_rate_roll
        self.body_angular_rate_pitch = body_angular_rate_pitch
        self.body_angular_rate_yaw = body_angular_rate_yaw
        self.time = time

        



    def update_pilot_control(self, pilot_control_long, pilot_control_lat, pilot_control_yaw, \
                        pilot_control_throttle):
        """Update the pilot control variables"""
        self.pilot_control_long = pilot_control_long
        self.pilot_control_lat = pilot_control_lat
        self.pilot_control_yaw = pilot_control_yaw
        self.pilot_control_throttle = pilot_control_throttle

    def update_switches(self, stability_augmentation_on_disc, autopilot_on_disc):
        """Update the switches"""
        self.stability_augmentation_on_disc = stability_augmentation_on_disc
        self.autopilot_on_disc = autopilot_on_disc

    def update_commands(self, equivalent_airspeed_command, altitude_msl_command, \
                        lateral_deviation_error, true_base_course_command):
        """Update the commands"""
        self.equivalent_airspeed_command = equivalent_airspeed_command
        self.altitude_msl_command = altitude_msl_command
        self.lateral_deviation_error = lateral_deviation_error
        self.true_base_course_command = true_base_course_command

        """Compute the control output given the current state"""
        # Constants
        design_equivalent_airspeed = 2.878088596053291e+02
        design_angle_of_attack = 2.653813535191715
        design_euler_angle_pitch = 2.653813535191715
        autopilot_alt_error_feedback_gain = -0.05
        autopilot_lat_offset_error_feedback_gain = -0.01
        autopilot_track_error_feedback_gain = -10.0


        # Derived variables
        ap_on = self.autopilot_on_disc
        sas_on = self.stability_augmentation_on_disc
        fsas_on = sas_on + ap_on

        if ap_on > 0.5:
            switched_keas_cmd = self.equivalent_airspeed_command
        else:
            switched_keas_cmd = design_equivalent_airspeed

        autopilot_altitude_error = self.altitude_msl - self.altitude_msl_command
        autopilot_delta_pitch_cmd =autopilot_alt_error_feedback_gain * autopilot_altitude_error

        #clip autopilotDelatPitchCmd to -5, 5
        if autopilot_delta_pitch_cmd > 5.0:
            autopilot_delta_pitch_cmd = 5.0
        if autopilot_delta_pitch_cmd < -5.0:
            autopilot_delta_pitch_cmd = -5.0

        autopilot_pitch_cmd = autopilot_delta_pitch_cmd + design_euler_angle_pitch
        if ap_on > 0.5:
            switched_theta_cmd = autopilot_pitch_cmd
        else:
            switched_theta_cmd = design_euler_angle_pitch

        disturbed_equivalent_airspeed = self.equivalent_airspeed - switched_keas_cmd
        disturbed_angle_of_attack = self.angle_of_attack - design_angle_of_attack
        disturbed_euler_angle_pitch = self.euler_angle_pitch - switched_theta_cmd

        autopilot_course_correction = self.lateral_deviation_error * autopilot_lat_offset_error_feedback_gain
        if autopilot_course_correction > 30.0:
            autopilot_course_correction = 30.0
        if autopilot_course_correction < -30.0:
            autopilot_course_correction = -30.0

        autopilot_course_command = self.true_base_course_command + autopilot_course_correction
        track_angle_est =self.euler_angle_yaw  +self.angle_of_sideslip
        autopilot_track_error = wrap(track_angle_est - autopilot_course_command, -180.0, 180.0)

        autopilot_commanded_bank_angle = autopilot_track_error_feedback_gain * autopilot_track_error
        if autopilot_delta_pitch_cmd > 30.0:
            autopilot_delta_pitch_cmd = 30.0
        if autopilot_delta_pitch_cmd < -30.0:
            autopilot_delta_pitch_cmd = -30.0

        if ap_on > 0.5:
            switched_phi_cmd = autopilot_commanded_bank_angle
        else:
            switched_phi_cmd = 0.0

        disturbed_euler_angle_roll = self.euler_angle_roll - switched_phi_cmd

        long_lqr_gain_matrix = np.array([
            [-0.063009074230494, 0.113230403179271, 10.113432224566077, 3.154983341632913],
            [0.997260602961658, -0.025467711176391, 1.213308488207827, 0.208744369535208]
        ],'d')

        latd_lqr_gain_matrix = np.array([
            [3.078043941515770, 0.032365863044163, 4.557858908828332, 0.589443156647647],
            [-0.705817452754520, -0.256362860634868, -1.073666149713151, 0.822114635953878]
        ],'d')

        long_vector = np.array([
            disturbed_equivalent_airspeed,
            disturbed_angle_of_attack,
            self.body_angular_rate_pitch,
            disturbed_euler_angle_pitch
        ],'d')

        lat_vector = np.array([
            disturbed_euler_angle_roll,
            self.angle_of_sideslip,
            self.body_angular_rate_roll,
            self.body_angular_rate_yaw,
        ],'d')

        long_lqr_command_vec = long_lqr_gain_matrix @ long_vector
        latd_lqr_command_vec = latd_lqr_gain_matrix @ lat_vector

        switched_pilot_control_lat = np.array([self.pilot_control_lat, self.pilot_control_yaw], 'd')
        if ap_on > 0.5:
            switched_pilot_control_lat = np.array([0.0, 0.0], 'd')

        switched_pilot_control_long = np.array([self.pilot_control_long, self.pilot_control_throttle],'d')
        if ap_on > 0.5:
            switched_pilot_control_long = np.array([0.0, 0.0], 'd')

        trimed_long_throttle = np.array([self.trim_el/25.0, self.trim_power/100.0],'d')

        if fsas_on > 0.5:
            lon_command = -long_lqr_command_vec + switched_pilot_control_long + trimed_long_throttle
            lat_command = -latd_lqr_command_vec + switched_pilot_control_lat
        else:
            lon_command = switched_pilot_control_long + trimed_long_throttle
            lat_command = switched_pilot_control_lat

        #clippings
        np.clip(lon_command, -1.0, +1.0, out=lon_command)
        np.clip(lat_command, -1.0, +1.0, out=lat_command)
        if lon_command[1] < 0.0:
            lon_command[1] = 0.0

        aileron_deflection = lat_command[0] * -21.5
        rudder_deflection = lat_command[1]*-30.0 + aileron_deflection*0.008
        elevator_deflection = lon_command[0] * -25.0
        power_lever_angle = lon_command[1] * 100.0


        return np.array([aileron_deflection, rudder_deflection, elevator_deflection, power_lever_angle],'d')

@jit(float64(float64, float64, float64))
def wrap(value, minus_lim, plus_lim):
    """Wraparound function for angles"""
    period = plus_lim - minus_lim
    result = value
    if value > plus_lim:
        result = value- period
    if value < minus_lim:
        result = value+ period
    if minus_lim <= result <= plus_lim:
        return result
    else:
        return wrap(result, minus_lim, plus_lim)

@jit(float64(float64, float64))
def tas_to_eas(tas, density):
    """Convert True Airspeed to Equivalent Airspeed"""
    return tas * np.sqrt(density / 1.225)
