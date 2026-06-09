"""Atmosphere/enviroment object, used to simulate a 
dynamic enviroment with wind
Implements only USSA1976 for now

https://www.ngdc.noaa.gov/stp/space-weather/online-publications/miscellaneous/us-standard-atmosphere-1976/us-standard-atmosphere_st76-1562_noaa.pdf"""

import numpy as np
from numba import jit, float64, boolean, int64    # import the types
from numba.experimental import jitclass


@jit(float64[:](float64), cache=True)
def get_pressure_density_temp(altitude):
    """Gets pressure from altitude, page 11 of document

    Parameters
    altitude

    Returns
    pressure, density, temperature array
    """

    start_heights  = np.array([0, 11000, 20000, 32000, 47000, 51000, 71000, 84852])
    start_pressures= np.array([101325, 22632.1, 5474.89, 868.019, 110.906, 66.9389, 3.95642 ])
    start_temp = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.95])
    temp_lapse = np.array([-0.0065, 0, 0.001, 0.0028, 0, -0.0028, -0.002 ])
    g = 9.80665
    Rs = 287.052874

    #probably should implement this
    if altitude < 0:
        altitude = 0

    if altitude > 84852:
        altitude = 84851.9

    #find constant subscript
    b = 0
    while altitude >= start_heights[b]:     #will raise an exeception if alt>84852
        b = b + 1
    b = b - 1
    #two equations depending on if the temperature changes with altitude, found on page 11
    if temp_lapse[b] == 0:
        pressure = start_pressures[b] * np.exp(-(altitude-start_heights[b]) * g / Rs /start_temp[b])
        temperature = start_temp[b]
        density = pressure / Rs / temperature
    else:
        pressure = start_pressures[b] * \
              ( start_temp[b]/(start_temp[b] + temp_lapse[b] * (altitude-start_heights[b])) ) \
                **(g / Rs / temp_lapse[b])
        temperature = start_temp[b] + temp_lapse[b] * (altitude-start_heights[b])
        density = pressure / Rs / temperature

    return np.array([pressure, density, temperature])

@jit(float64(float64), cache=True)
def get_speed_of_sound(temperature_K):
    """Speed of sounds [m/s]"""
    gamma_air = 1.4
    r_air = 287.052874

    return np.sqrt(gamma_air*r_air*temperature_K)

@jit(float64(float64), cache=True)
def get_dynamic_viscosity(temperature):
    """Equation 51 of USSA1976"""
    beta = 1.458e-6
    sutherlands = 110.4

    mu = (beta * temperature**1.5)/(temperature+sutherlands)

    return mu


spec = [
    #wind profile input
    ('wind_alt', float64[:]),
    ('wind_speed', float64[:]),
    ('wind_direction', float64[:]),

    #current values
    ('current_alt', float64),
    ('current_time', float64),
    ('current_density', float64),
    ('current_temperature', float64),
    ('current_pressure', float64),

    #Dryden turbulence state
    ('turbulence_on', boolean),     #master enable
    ('turb_seed', int64),           #RNG seed derived from the wind profile
    ('turb_init', boolean),         #have the gust states been primed yet
    ('turb_last_time', float64),    #time [s] of the previous turbulence update
    ('turb_u', float64),            #longitudinal gust velocity [m/s]
    ('turb_v', float64),            #lateral gust velocity [m/s]
    ('turb_w', float64)             #vertical gust velocity [m/s]
]

@jitclass(spec)
class Atmosphere(object):
    "Atmosphere jit'd object, storing the wind profile and atmosphere model"

    def __init__(self, wind_alt_profile = np.array([0.0,0.0]),
                 wind_speed_profile = np.array([0.0,0.0]),
                 wind_direction_profile = np.array([0.0,0.0]),
                 turbulence = False):
        self.wind_alt       = wind_alt_profile
        self.wind_speed     = wind_speed_profile
        self.wind_direction = wind_direction_profile

        self.current_alt = 0.0
        self.current_time = 0.0
        self.current_pressure = 0.0
        self.current_density = 0.0
        self.current_temperature = 0.0

        #Dryden turbulence: optional, off by default so existing behaviour is unchanged
        self.turbulence_on = turbulence
        self.turb_init     = False
        self.turb_last_time= 0.0
        self.turb_u = 0.0
        self.turb_v = 0.0
        self.turb_w = 0.0

        #The wind profile arrays "seed" the noise: a deterministic integer is built
        #from them so a given scenario always produces the same gust realisation
        #(reproducible Monte-Carlo), while different wind setups give different gusts.
        self.turb_seed = int64(np.abs(np.sum(wind_speed_profile))     * 1000.0
                             + np.abs(np.sum(wind_direction_profile)) * 13.0
                             + np.abs(np.sum(wind_alt_profile))       * 7.0) + 1
        if self.turbulence_on:
            np.random.seed(self.turb_seed)

    def change_wind_profile(self, wind_alt_profile, wind_speed_profile, wind_direction_profile):
        """updates the wind profile"""
        self.wind_alt       = wind_alt_profile
        self.wind_speed     = wind_speed_profile
        self.wind_direction = wind_direction_profile

    def update_conditions(self, altitude, time = 0.0, airspeed = 0.0):
        """Updates experienced atmosphere conditons
        must be called every timestep

        airspeed [m/s] is the true airspeed and is only needed when the Dryden
        turbulence model is active (it converts the spatial gust field into a
        temporal one).  If it is left at the default the local mean wind speed is
        used as a fall-back so old call sites keep working unchanged."""
        self.current_alt = altitude
        self.current_time = time

        pdt = get_pressure_density_temp(altitude)
        self.current_pressure    = pdt[0]
        self.current_density     = pdt[1]
        self.current_temperature = pdt[2]

        if self.turbulence_on:
            dt = time - self.turb_last_time
            self.turb_last_time = time
            #only step the filters forward on a real, positive time increment
            if dt > 0.0:
                self.update_turbulence(altitude, dt, airspeed)

    def update_turbulence(self, altitude, dt, airspeed):
        """Advances the Dryden turbulence gust velocities by one time step.

        Implements the low-altitude Dryden continuous-gust model from the U.S.
        military specifications MIL-F-8785C and MIL-HDBK-1797 (and as documented
        in the MathWorks Aerospace Blockset "Dryden Wind Turbulence Model").
        Each of the three gust components is the output of a first-order forming
        filter driven by unit-variance Gaussian white noise; we use the *exact*
        discretisation of that filter (an Ornstein-Uhlenbeck update) so the model
        is unconditionally stable and the steady-state variance is exactly the
        commanded turbulence intensity for any time step.

        All quantities are SI.  The empirical scale-length / intensity fits below
        were calibrated with altitude in feet, so the altitude is converted to
        feet for those formulas only and the resulting lengths converted back to
        metres."""

        FT_PER_M = 3.280839895013123   # 1 / 0.3048, foot<->metre conversion

        # --- driving wind speed -------------------------------------------------
        # MIL-HDBK-1797 parameterises low-altitude turbulence by W_20, the mean
        # wind speed at 20 ft (6.096 m).  We read it from the supplied wind
        # profile so the turbulence intensity scales with the modelled wind.
        w_20 = np.interp(6.096, self.wind_alt, self.wind_speed)

        # Airspeed V converts the spatial Dryden spectrum into a temporal signal
        # ("frozen-field"/Taylor hypothesis).  Fall back to the local mean wind
        # speed if no airspeed was supplied; clamp to avoid a zero time constant.
        v_tas = airspeed
        if v_tas <= 0.0:
            v_tas = np.interp(altitude, self.wind_alt, self.wind_speed)
        if v_tas < 0.1:
            v_tas = 0.1

        # --- scale lengths L and intensities sigma -----------------------------
        # Altitude is clamped to [10 ft, 1000 ft]: below 10 ft the empirical fit
        # is not defined, and the low-altitude model is only specified up to
        # 1000 ft, above which we hold the 1000 ft values (a documented
        # simplification; the high-altitude model would otherwise need a separate
        # turbulence-severity input).
        h_ft = altitude * FT_PER_M
        if h_ft < 10.0:
            h_ft = 10.0
        if h_ft > 1000.0:
            h_ft = 1000.0

        # 0.177 and 0.000823 are the empirical constants of the MIL-HDBK-1797
        # low-altitude fit; the 1.2 and 0.4 exponents come from the same fit.
        denom = 0.177 + 0.000823 * h_ft

        l_w = (h_ft) / FT_PER_M                       # L_w = h  (vertical scale length)
        l_u = (h_ft / denom**1.2) / FT_PER_M          # L_u = L_v = h / (...)^1.2
        l_v = l_u

        sigma_w = 0.1 * w_20                           # sigma_w = 0.1 * W_20
        sigma_u = sigma_w / denom**0.4                 # sigma_u = sigma_v = sigma_w/(...)^0.4
        sigma_v = sigma_u

        # --- prime the gust states on the first step ----------------------------
        # Draw the initial gusts directly from the stationary distribution
        # N(0, sigma^2) so the field starts already "developed" rather than from
        # zero.
        if not self.turb_init:
            self.turb_u = sigma_u * np.random.normal(0.0, 1.0)
            self.turb_v = sigma_v * np.random.normal(0.0, 1.0)
            self.turb_w = sigma_w * np.random.normal(0.0, 1.0)
            self.turb_init = True
            return

        # --- exact first-order (Ornstein-Uhlenbeck) update ----------------------
        # The first-order Dryden filter has correlation time tau = L / V, i.e. a
        # pole at beta = V / L.  Exact discretisation of  x' = -beta x + noise
        # over a step dt is   x[k] = phi*x[k-1] + sigma*sqrt(1-phi^2)*N(0,1)
        # with phi = exp(-beta*dt); this preserves the variance sigma^2 exactly.
        beta_u = v_tas / l_u
        beta_v = v_tas / l_v
        beta_w = v_tas / l_w

        phi_u = np.exp(-beta_u * dt)
        phi_v = np.exp(-beta_v * dt)
        phi_w = np.exp(-beta_w * dt)

        self.turb_u = phi_u * self.turb_u + sigma_u * np.sqrt(1.0 - phi_u*phi_u) * np.random.normal(0.0, 1.0)
        self.turb_v = phi_v * self.turb_v + sigma_v * np.sqrt(1.0 - phi_v*phi_v) * np.random.normal(0.0, 1.0)
        self.turb_w = phi_w * self.turb_w + sigma_w * np.sqrt(1.0 - phi_w*phi_w) * np.random.normal(0.0, 1.0)

    def get_density(self):
        """Returns density [kg/m^3]"""
        return self.current_density

    def get_temperature(self):
        """Returns temperature [K]"""
        return self.current_temperature

    def get_pressure(self):
        """Returns pressure [Pa]"""
        return self.current_pressure

    def get_speed_of_sound(self):
        """Returns speed of sound [m/s]"""
        return get_speed_of_sound(self.current_temperature)

    def get_wind_ned(self):
        """Returns wind speed NED [m/s] (mean wind plus Dryden turbulence)"""
        wind_speed = np.interp(self.current_alt, self.wind_alt, self.wind_speed)
        wind_direction = np.interp(self.current_alt, self.wind_alt, self.wind_direction)

        dir_rad = wind_direction * 0.017453292519943295
        wind_east  = np.sin(dir_rad) * wind_speed
        wind_north = np.cos(dir_rad) * wind_speed
        wind_down = 0.0

        if self.turbulence_on:
            # The Dryden gusts are body/wind-axis quantities: u is longitudinal
            # (along the mean wind), v is lateral (horizontal, 90 deg to it) and
            # w is vertical.  Project them into NED.  (A full implementation would
            # rotate u,v into the aircraft body axes; aligning with the mean wind
            # is a reasonable approximation when no heading is available here.)
            along_n  =  np.cos(dir_rad)   # unit vector along the mean wind
            along_e  =  np.sin(dir_rad)
            cross_n  = -np.sin(dir_rad)   # horizontal unit vector 90 deg to it
            cross_e  =  np.cos(dir_rad)

            wind_north += self.turb_u * along_n + self.turb_v * cross_n
            wind_east  += self.turb_u * along_e + self.turb_v * cross_e
            wind_down  += self.turb_w

        return np.array([wind_north, wind_east, wind_down], 'd')

    def get_turbulence_ned(self):
        """Returns the current Dryden gust velocities resolved into NED [m/s].
        Zero when turbulence is disabled. Useful for logging/diagnostics."""
        if not self.turbulence_on:
            return np.array([0.0, 0.0, 0.0], 'd')

        wind_direction = np.interp(self.current_alt, self.wind_alt, self.wind_direction)
        dir_rad = wind_direction * 0.017453292519943295

        turb_north = self.turb_u * np.cos(dir_rad) - self.turb_v * np.sin(dir_rad)
        turb_east  = self.turb_u * np.sin(dir_rad) + self.turb_v * np.cos(dir_rad)
        turb_down  = self.turb_w

        return np.array([turb_north, turb_east, turb_down], 'd')
