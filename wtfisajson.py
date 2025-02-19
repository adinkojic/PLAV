import json
import math

with open('openvspmodel.json', 'r') as file:
    modelparam = json.load(file)

file.close()


maxld = 1/2/math.sqrt(modelparam['k2']*modelparam['C_D0p'])
clldmax = math.sqrt(modelparam['C_D0p']/modelparam['k2'])
print(maxld)
print(clldmax)