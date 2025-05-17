"""reads the joystick and prints the values to the console"""

import serial
import numpy as np


def process_response(byte_response):
    """Process the byte response from the joystick."""
    valid_data = bool(byte_response[20] & 1)
    timed_out = bool((byte_response[20] >> 1) & 1)

    channel_values = [0] * 10
    for i in range(10):
        channel_values[i] = (byte_response[2*i] << 8) + byte_response[2*i + 1]

    return channel_values, valid_data, timed_out

class JoystickReader(object):
    """Class to read controller data."""
    def __init__(self, serial_port='COM6', baudrate=115200):
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.ser = serial.Serial(self.serial_port, baudrate=self.baudrate)

    def get_control_output(self):
        """Request and response for control system."""
        channels, valid, timed_out = self.read_joystick()
        if timed_out:
            print("Timed out, joystick not connected")
            return np.array([0.0,0.0,0.0,0.0,0.0],'d')

        throttle = -(channels[2] - 1000) / 350.0
        ailerons1 = (channels[0] - 1000) / 350.0
        ailerons2 = (channels[5] - 1000) / 350.0
        elevator = -(channels[1] - 1000) / 350.0
        rudder = -(channels[3] - 1000) / 350.0

        return np.array([rudder, ailerons1, elevator, throttle, ailerons2], dtype='d')


    def read_joystick(self):
        """Read joystick data."""
        response_header = b'c'
        self.ser.write(response_header)  # Send the response header
        response = self.ser.read(21)
        channels, valid, timed_out = process_response(response)
        return channels, valid, timed_out

    def update_enviroment(self, *dummy_args):
        """nothing its a joystick"""

    def update_pilot_control(self, *dummy_args):
        """nothing its a joystick"""

    def close(self):
        """Close the serial port."""
        self.ser.close()
