"""F-16 Control Scheme, implementing F16_control.dml"""

import numpy as np
from numba import jit, float64
from numba.experimental import jitclass
import serial

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
class F16ControlHITL(object):
    """F-16 Control Sysem"""
    def __init__(self, commands, serial_port= 'COM3'): 
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

        self.ser = serial.Serial(serial_port)



    def get_control_output(self):
        """computes and gets the control output in -1 to 1 range"""

        #this is relevant to the nasa cases
        self.check_commands()

        control = self.get_device_response()
        self.ail = control[1] / 20.0 - self.trim_ail / 20.0
        self.rdr = control[0] / 30.0 - self.trim_rdr / 30.0
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

        self.update_device_env()

    def update_device_env(self):
        """Updates the HITL device with the current enviroment"""
        packet = make_enviroment_packet(self.altitude_msl, self.equivalent_airspeed, \
                                            self.angle_of_attack, self.angle_of_sideslip, \
                                            self.euler_angle_roll, self.euler_angle_pitch, \
                                            self.euler_angle_yaw, self.body_angular_rate_roll ,\
                                            self.body_angular_rate_pitch, self.body_angular_rate_yaw, \
                                            self.time)
        self.ser.write(packet)

    def update_device_commands(self):
        """Updates the HITL device with the current commands"""
        packet = make_command_packet(self.equivalent_airspeed_command, self.altitude_msl_command, \
                                        self.lateral_deviation_error, self.true_base_course_command, \
                                        self.stability_augmentation_on_disc, self.autopilot_on_disc)
        self.ser.write(packet)

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

        self.update_device_commands()

    def update_commands(self, equivalent_airspeed_command, altitude_msl_command, \
                        lateral_deviation_error, true_base_course_command):
        """Update the commands"""
        self.equivalent_airspeed_command = equivalent_airspeed_command
        self.altitude_msl_command = altitude_msl_command
        self.lateral_deviation_error = lateral_deviation_error
        self.true_base_course_command = true_base_course_command

        self.update_device_commands()

    def get_device_response(self):
        """Get the control response from the device"""

        reponse_header = b'+++c'
        self.ser.write(reponse_header)  # Send the response header
        #time.sleep(0.01) # Wait for the device to process the command
        reponse = self.ser.read(16)
        return process_control_response(reponse)
    
    def shut_down_hil(self):
        """Shut down the HITL device"""
        self.ser.close()



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

def make_enviroment_packet(altitude_msl, equivalent_airspeed, angle_of_attack, \
                        angle_of_sideslip, euler_angle_roll, euler_angle_pitch, \
                        euler_angle_yaw, body_angular_rate_roll ,\
                        body_angular_rate_pitch, body_angular_rate_yaw, sim_time):
    """Update the enviroment variables to the device"""
    bit_message = b'+++e'

    float_scaling = 21474.83648

    alt_as_int = np.int32(altitude_msl * float_scaling)
    eas_as_int = np.int32(equivalent_airspeed * float_scaling)
    aoa_as_int = np.int32(angle_of_attack * float_scaling)
    aos_as_int = np.int32(angle_of_sideslip * float_scaling)
    euler_roll_as_int = np.int32(euler_angle_roll * float_scaling)
    euler_pitch_as_int = np.int32(euler_angle_pitch * float_scaling)
    euler_yaw_as_int = np.int32(euler_angle_yaw * float_scaling)
    body_angular_rate_roll_as_int = np.int32(body_angular_rate_roll * float_scaling)
    body_angular_rate_pitch_as_int = np.int32(body_angular_rate_pitch * float_scaling)
    body_angular_rate_yaw_as_int = np.int32(body_angular_rate_yaw * float_scaling)
    sim_time_as_int = np.int32(sim_time * float_scaling)

    bit_message += alt_as_int.tobytes()
    bit_message += eas_as_int.tobytes()
    bit_message += aoa_as_int.tobytes()
    bit_message += aos_as_int.tobytes()
    bit_message += euler_roll_as_int.tobytes()
    bit_message += euler_pitch_as_int.tobytes()
    bit_message += euler_yaw_as_int.tobytes()
    bit_message += body_angular_rate_roll_as_int.tobytes()
    bit_message += body_angular_rate_pitch_as_int.tobytes()
    bit_message += body_angular_rate_yaw_as_int.tobytes()
    bit_message += sim_time_as_int.tobytes()

    return bit_message

def make_command_packet(equivalent_airspeed_command, altitude_msl_command, \
                        lateral_deviation_error, true_base_course_command, \
                        stability_augmentation_on_disc, autopilot_on_disc):
    """Update the commands to the device"""
    float_scaling = 21474.83648

    bit_message = b'+++i'
    eas_as_int = np.int32(equivalent_airspeed_command * float_scaling)
    alt_as_int = np.int32(altitude_msl_command * float_scaling)
    lde_as_int = np.int32(lateral_deviation_error * float_scaling)
    tbc_as_int = np.int32(true_base_course_command * float_scaling)
    
    bit_message += eas_as_int.tobytes()
    bit_message += alt_as_int.tobytes()
    bit_message += lde_as_int.tobytes()
    bit_message += tbc_as_int.tobytes()

    last_byte = np.int32(0)
    if stability_augmentation_on_disc > 0.0:
        last_byte += 1
    if autopilot_on_disc > 0.0:
        last_byte += 2
    bit_message += last_byte.tobytes()

    return bit_message

def process_control_response(byte_response):
    """Process the control response from the device."""

    float_scaling = 21474.83648

    rudder = byte_response[0] + (byte_response[1] << 8 ) + (byte_response[2] << 16 ) + (byte_response[3] << 24)
    if byte_response[3] > 127: #dont forget to check the sign bit
        rudder = rudder - 2**32
    rudder = float(rudder / float_scaling)
    aileron = byte_response[4] + (byte_response[5] << 8 ) + (byte_response[6] << 16 ) + (byte_response[7] << 24)
    if byte_response[7] > 127: #dont forget to check the sign bit
        aileron = aileron - 2**32
    aileron = float(aileron / float_scaling)
    elevator = byte_response[8] + (byte_response[9] << 8 ) + (byte_response[10] << 16 ) + (byte_response[11] << 24)
    if byte_response[11] > 127: #dont forget to check the sign bit
        elevator = elevator - 2**32
    elevator = float(elevator / float_scaling)
    throttle = byte_response[12] + (byte_response[13] << 8 ) + (byte_response[14] << 16 ) + (byte_response[15] << 24)
    if byte_response[15] > 127: #dont forget to check the sign bit
        throttle = throttle - 2**32
    throttle = float(throttle / float_scaling)

    return np.array([rudder, aileron, elevator, throttle],'d')