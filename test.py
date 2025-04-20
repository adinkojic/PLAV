
import json
import math
from numba import jit, float64
import numpy as np

import aircraftconfig

#load aircraft config
with open('aircraftConfigs/openvspmodel.json', 'r', encoding=None) as file:
    modelparam = json.load(file)
file.close()

aircraft = aircraftconfig.init_aircraft(modelparam)

velocity = np.array([33, 0, 1.5],'d')

omega = np.array([0., 0., 0.], 'd')

pitch = 2.4
control = np.array([0., 0., pitch* math.pi/180., 0.], 'd')

aircraft.update_control(control)
aircraft.update_conditions(120.0, velocity, omega, 1.225, 295., 300. )


forces, moments = aircraft.get_forces()

print("forces: ", forces)
print("moments: ", moments)
print("y: ", moments[1])

print("alpha:", aircraft.get_alpha() )
print("beta:", aircraft.get_beta())
print("grid fins:", aircraft.calculate_grid_fin_forces())


C_XYlutX = np.array(modelparam['C_XYlutX'],'d')
C_XlutY  = np.array(modelparam['C_XlutY'], 'd')
C_YlutY  = np.array(modelparam['C_YlutY'], 'd')


top_drag  = np.interp(0.0,  C_XYlutX, C_YlutY)
print('drag test: ', top_drag)
