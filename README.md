# Python Laptop Air Vehicles (PLAV)

### A 6 Degree-of-Freedom Flight Simulator written in Python with Real-Time Arduino Hardware in Loop Simulation

This software is designed for users to implement their own aircraft models and simulate it in a somewhat realistic model of the Earth. They can enter basic aerodynamic coefficients or make their own flight dynamics model from the ground up. It's really meant for engineering students trying to validate their own air vehicle (airplane, rocket) before flying without dealing with C++ or learning more developed programs like JSBSim.

The simulation models the World Geodetic System 1984 (WGS84) ellispoid model of the Earth with the US Standard Atmosphere 1976. A review of the theory behind this simulation is in my Senior Thesis, which is in this repo.

This software has been validated using the [NASA Engineering and Safety Center Atmospheric check cases](https://nescacademy.nasa.gov/flightsim/2015) 1 to 3 and 6 to 13.3. These cases are avaible for the user to run. 

To run the check cases youself, download the software and install the dependencies from the requirements.txt file. 

`pip install -r requirements.txt`

To test the HITL mode, install the sim proof-of-concept Arduino code in `\arduinoCode` with the Arduino IDE. Make sure you enter the correct COM port and close the Arduino IDE. The systems supports both offline and real-time simulation modes.

### ArduPilot SITL

Works with ArduPilot's JSON with SITL (https://ardupilot.org/dev/docs/sitl-with-JSON.html)
(Ardupilot control not fully implemented)

Start PLAV first and make sure it's running. Run
`plav sitl-sim --noise --live`
`--noise` adds sensor noise, `--live` adds live wind data
Remember to click the play button.

Then, start ArduPilot SITL. (It works if you start ArduPilot first but SITL boots much quicker if PLAV is already running).

If using WSL, run PLAV's sim_with_sitl with ip 0.0.0.0.
To figure out the IP to use for ArduPilot, run `ip route | awk '/^default/ {print $3}'`
Then use that ip in `../Tools/autotest/sim_vehicle.py -f JSON:[WINDOWSIP] --console --map` from the ArduPlane directory.

Add 
`BRGRBalloon=40.269712,-73.769042,34000.0,0
BRGRDrone=40.3320972,-74.6733131, 120.0,0`
to your locations.txt in ArduPilot too

If using Linux, use default values
