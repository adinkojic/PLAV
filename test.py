"""
Simple Flight Dynamics Model (FDM) example that makes the altitude increase and the plane roll in the air.
"""
import time
import math
from flightgear_python.fg_if import FDMConnection

def fdm_callback(fdm_data, event_pipe):

    fdm_data.alt_m = 500  # or just make a relative change
    fdm_data.phi_rad = fdm_data.phi_rad - 0.01
    return fdm_data  # return the whole structure

"""
Start FlightGear with `--native-fdm=socket,out,30,localhost,5501,udp --native-fdm=socket,in,30,localhost,5502,udp`
(you probably also want `--fdm=null` and `--max-fps=30` to stop the simulation fighting with
these external commands)
"""
if __name__ == '__main__':  # NOTE: This is REQUIRED on Windows!
    fdm_conn = FDMConnection()
    fdm_event_pipe = fdm_conn.connect_rx('localhost', 5501, fdm_callback)
    fdm_conn.connect_tx('localhost', 5502)
    fdm_conn.start()  # Start the FDM RX/TX loop


    while True:
        # could also do `fdm_conn.event_pipe.parent_send` so you just need to pass around `fdm_conn`
        fdm_event_pipe.parent_send()  # send tuple
        time.sleep(0.5)