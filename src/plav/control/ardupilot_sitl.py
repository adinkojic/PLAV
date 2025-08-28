"""Control scheme that interfaces with the ArduPilot SITL (Software In The Loop) simulation."""

import time, socket, struct, json, threading

import plav.step_logging as slog

class ArduPilotSITL:
    """ArduPilot interface for Control
    IP is the UDP address to the ArduPilot SITL
    0.0.0.0 if PLAV runs in Windows (And SITL runs in WSL)
    127.0.0.1 if running both in Linux
    """

    def __init__(self, ardupilot_ip = "0.0.0.0", ardupilot_port = 9002):
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
        self.gyro = [0.0, 0.0, 0.0]
        self.accel = [0.0, 0.0, 0.0]
        self.quat = [1.0, 0.0, 0.0, 0.0]
        self.pos = [0.0, 0.0, 0.0]
        self.velo = [0.0, 0.0, 0.0]

        self.frame_rate_hz = 1
        self.fresh_data = False



        self.transmitted = time.time()

        receiver_thread = threading.Thread(target=self.servo_receiver, daemon=True)
        #sender_thread = threading.Thread(target=self.sitl_sender, daemon=True)
        self.servo_receiver(True) #run at least once to get the addy
        print(self.frame_rate_hz)
        receiver_thread.start()
        self.sitl_sender()
        #sender_thread.start()
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

                #print(frame_number)

                if 1000 <= pwm[0] <= 2000:
                    self.ardupilot_aileron  = (pwm[0] -1500) / 500.0#pwm pulse to our servo deflection
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

    def sitl_sender(self):
        """daemon that the data to SITL
        transmits every frame after fresh data
        1 s if otherwise"""

        #if not self.fresh_data:
        #    if time.time() - self.transmitted < 1.0:
        #        continue

        json_data = {
            "timestamp": self.phys_time,
            "imu": {
                "gyro": self.gyro,
                "accel_body": self.accel
            },
            "position": self.pos,
            "quaternion": self.quat,
            #"attitude": rpy,
            "velocity": self.velo
        }

        #print(json_data["timestamp"])

        try:
            print("trying to send")
            self.sock.sendto((json.dumps(json_data, separators=(',', ':')) + "\n").encode("ascii"), self.address)
            self.transmitted = time.time()
            print("Sent data to SITL" + time.time())
        except TypeError:
            pass
        
        
        #if self.fresh_data:
        #    self.fresh_data = False
        #time.sleep(1/self.frame_rate_hz)
        #time.sleep(0.001)
    
    def get_control_output(self):
        """Returns latest control output"""
        return self.ardupilot_rudder, self.ardupilot_aileron, self.ardupilot_elevator, self.ardupilot_throttle

    def update_environment(self, latest_data):
        """Sends the environment data to the SITL"""
        self.phys_time = latest_data[slog.SDI_TIME]
        self.gyro = [latest_data[slog.SDI_P], latest_data[slog.SDI_Q], latest_data[slog.SDI_R]]

        self.accel = [latest_data[slog.SDI_AX], latest_data[slog.SDI_AY], latest_data[slog.SDI_AZ]]

        self.quat = [latest_data[slog.SDI_Q1], latest_data[slog.SDI_Q2],
                latest_data[slog.SDI_Q3], latest_data[slog.SDI_Q4]]

        self.pos = [latest_data[slog.SDI_DELTA_N], latest_data[slog.SDI_DELTA_E],
               latest_data[slog.SDI_DELTA_D]]

        self.velo = [latest_data[slog.SDI_VN], latest_data[slog.SDI_VE], latest_data[slog.SDI_VD]]

        #rpy = [latest_data[slog.SDI_ROLL], latest_data[slog.SDI_PITCH], latest_data[slog.SDI_YAW]]

        self.fresh_data = True
        self.sitl_sender()

        #sender_thread = threading.Thread(target=self.sitl_sender, daemon=True)
        #sender_thread.start()

    def is_hitl(self):
        """Check if the control system is HITL/SITL"""
        return False

    def update_pilot_control(self, pilot_control_long, pilot_control_lat, pilot_control_yaw, \
                        pilot_control_throttle):
        """TODO: Allow manual control input to ArduPilot"""
