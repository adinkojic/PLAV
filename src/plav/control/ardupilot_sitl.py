"""Control scheme that interfaces with the ArduPilot SITL (Software In The Loop) simulation."""

import time, socket, struct, json, threading

import numpy as np

from plav.imu_noise import IMUNoise, deg_to_rad, ug_to_mps2_per_sqrtHz, dph_to_radps
import plav.step_logging as slog


class ArduPilotSITL:
    """ArduPilot interface for Control
    IP is the UDP address to the ArduPilot SITL
    0.0.0.0 if PLAV runs in Windows (And SITL runs in WSL)
    127.0.0.1 if running both in Linux
    """

    def __init__(self, ardupilot_ip = "0.0.0.0", ardupilot_port = 9002, add_noise = False):
        self.ardupilot_ip = ardupilot_ip
        # --- UDP communication setup ---
        print('Initalizing SITL UDP communication')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ardupilot_ip, ardupilot_port))
        self.sock.settimeout(0.1)

        self.last_sitl_frame = -1
        self.connected = False
        self.frame_count = 0
        self.frame_time = time.time()
        self.print_frame_count = 1000

        self.ardupilot_aileron = 0.0
        self.ardupilot_elevator = 0.0
        self.ardupilot_throttle = 0.0
        self.ardupilot_rudder = 0.0

        self.address = None

        self.phys_time = 0.0
        self.gyro = np.zeros(3,'d')
        self.accel = np.zeros(3,'d')
        self.quat = [1.0, 0.0, 0.0, 0.0]
        self.pos = [0.0, 0.0, 0.0]
        self.velo = [0.0, 0.0, 0.0]
        self.airspeed = 0.0
        self.wind = [0.0, 0.0, 0.0]

        self.frame_rate_hz = 1
        self.fresh_data = False

        self.sim_paused = False

        if add_noise:
            self.imu_noise = IMUNoise( #values taken from ICM-45686 datasheet
                fs=1000.0,
                gyro_nd=deg_to_rad(0.0038),
                accel_nd=ug_to_mps2_per_sqrtHz(70.0),
                gyro_bias_sigma=dph_to_radps(10.0),
                gyro_bias_tau=500.0,
                accel_bias_sigma=1.0e-2*9.80665,
                accel_bias_tau=500.0,
            )
        else:
            self.imu_noise = None

        self.transmitted = time.time()

        receiver_thread = threading.Thread(target=self.servo_receiver, daemon=True)
        self.servo_receiver(True) #run at least once to get the addy
        receiver_thread.start()
        self.sitl_sender()
        self.sender_thread = None
        print('SITL UDP communication initialized')

    def servo_receiver(self, oneshot= False):
        """Gets the servo commands from the SITL"""
        while not oneshot:
            try:
                data, self.address = self.sock.recvfrom(100)
                parse_format = 'HHI16H'
                if len(data) != struct.calcsize(parse_format):
                    print(f"Bad packet size: {len(data)}")
                decoded = struct.unpack(parse_format, data)
                magic = 18458
                if decoded[0] != magic:
                    print(f"Incorrect magic: {decoded[0]}")
                self.frame_rate_hz = decoded[1]
                frame_number = decoded[2]
                pwm = decoded[3:]
                #print(self.frame_rate_hz)

                #print(pwm)

                if 1000 <= pwm[0] <= 2000:
                    self.ardupilot_aileron = (pwm[0] -1500) / 500.0#pwm pulse our servo deflection
                #else:
                    #print(f"pwm out of bounds: {pwm[0]}")
                if 1000 <= pwm[1] <= 2000:
                    self.ardupilot_elevator = -(pwm[1] -1500) / 500.0
                if 1000 <= pwm[2] <= 2000:
                    self.ardupilot_throttle = (pwm[2] -1500) / 500.0
                if 1000 <= pwm[3] <= 2000:
                    self.ardupilot_rudder   = -(pwm[3] -1500) / 500.0

                #TODO: if frame_rate_hz != RATE_HZ: ... RATE_HZ = frame_rate_hz
                #TODO: reset logic
                self.frame_count += 1

            except socket.timeout:
                time.sleep(0.01)

    def paused(self):
        """call when pausing"""
        self.sim_paused = True
        if self.sender_thread is None:
            self.sender_thread = threading.Thread(target=self.paused_daemon, daemon=True)
            self.sender_thread.start()

    def unpaused(self):
        """call when unpausing"""
        self.sim_paused = False
        if self.sender_thread is not None:
            self.sender_thread.join()
            self.sender_thread = None

    def paused_daemon(self):
        """Daemon thread that runs when the simulation is paused to keep SITL updated"""
        while self.sim_paused:
            self.sitl_sender()
            time.sleep(0.1)


    def sitl_sender(self):
        """daemon that the data to SITL
        transmits every frame after fresh data
        1 s if otherwise"""

        #if not self.fresh_data:
        #    if time.time() - self.transmitted < 1.0:
        #        continue

        

        if self.imu_noise is not None:
            gyro_out, accel_out = self.imu_noise.step(self.gyro,self.accel)
        else:
            gyro_out = self.gyro
            accel_out = self.accel

        gyro_out = gyro_out.tolist()
        accel_out = accel_out.tolist()



        json_data = {
            "timestamp": self.phys_time,
            "imu": {
                "gyro": gyro_out,
                "accel_body": accel_out
            },
            "position": self.pos,
            "quaternion": self.quat,
            #"attitude": rpy,
            "velocity": self.velo,
            "airspeed": self.airspeed,
            "velocity_wind": self.wind,
        }
        #print(self.pos[2])

        try:
            self.sock.sendto((json.dumps(json_data, separators=(',', ':')) + "\n").encode("ascii"), self.address)
        except socket.timeout:
            pass
        except TypeError:
            pass
    
    def get_control_output(self):
        """Returns latest control output"""
        return self.ardupilot_rudder, self.ardupilot_aileron, self.ardupilot_elevator, self.ardupilot_throttle

    def update_environment(self, latest_data):
        """Sends the environment data to the SITL"""
        self.phys_time = latest_data[slog.SDI_TIME]
        self.gyro = np.array([latest_data[slog.SDI_P], latest_data[slog.SDI_Q], latest_data[slog.SDI_R]],'d')

        self.accel = np.array([latest_data[slog.SDI_AX], latest_data[slog.SDI_AY], latest_data[slog.SDI_AZ]],'d')

        self.quat = [latest_data[slog.SDI_Q1], latest_data[slog.SDI_Q2],
                latest_data[slog.SDI_Q3], latest_data[slog.SDI_Q4]]

        self.pos = [latest_data[slog.SDI_DELTA_N], latest_data[slog.SDI_DELTA_E],
               latest_data[slog.SDI_DELTA_D]]

        self.velo = [latest_data[slog.SDI_VN], latest_data[slog.SDI_VE], latest_data[slog.SDI_VD]]

        self.airspeed = latest_data[slog.SDI_TAS]

        self.wind = [-latest_data[slog.SDI_WIND_N], -latest_data[slog.SDI_WIND_E], -latest_data[slog.SDI_WIND_D]]

        #rpy = [latest_data[slog.SDI_ROLL], latest_data[slog.SDI_PITCH], latest_data[slog.SDI_YAW]]

        #self.fresh_data = True
        self.sitl_sender()

        #sender_thread = threading.Thread(target=self.sitl_sender, daemon=True)
        #sender_thread.start()

    def is_hitl(self):
        """Check if the control system is HITL/SITL"""
        return True

    def update_pilot_control(self, pilot_control_long, pilot_control_lat, pilot_control_yaw, \
                        pilot_control_throttle):
        """TODO: Allow manual control input to ArduPilot"""
