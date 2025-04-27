# Python Laptop Air Vehicles (PLAV)

### A 6 Degree-of-Freedom Flight Simulator written in Python with Real-Time Arduino Hardware in Loop Simulation

This software is designed for users to implement their own aircraft models and simulate it in a somewhat realistic model of the Earth. They can enter basic aerodynamic coefficients or make their own flight dynamics model from the ground up.

The simulation models the World Geodetic System 1984 (WGS84) ellispoid model of the Earth with the US Standard Atmosphere 1976.

This software has been validated using the [NASA Engineering and Safety Center Atmospheric check cases](https://nescacademy.nasa.gov/flightsim/2015) 1 to 3 and 6 to 13.3. These cases are avaible for the user to run. 

To run the check cases youself, download the software and install the dependencies from the requirements.txt file. 

`pip install -r requirements.txt`

There's no UI yet, so find the json file load input in `plav.py` and enter the approriate case. They are avaible in the `\aircraftConfigs` folder. There are options to use to real time and hardware-in-the-loop simulation and the sim duration. Run `plav.py` it will run the sim. The sim can be unpaused with the Pause/Play button.

To test the HITL mode, install the sim proof-of-concept Arduino code in `\arduinoCode` with the Arduino IDE. Make sure you enter the correct COM port and close the Arduino IDE. The systems supports both offline and real-time simulation modes.

I'll probably add one of the open source licenses later.
