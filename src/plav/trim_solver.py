"""Trim solver for glider
For a given flight speed and altitude, computes AoA, control for p=q=r=hddot
(Bascially steady glider for that speed)"""


from typing import Tuple, Optional, Dict

import numpy as np
from scipy.optimize import least_squares

from plav.vehicle_models.generic_aircraft_config import AircraftConfig, init_aircraft, init_dummy_aircraft, alpha_beta_to_velocity
from plav.atmosphere_models.ussa1976 import get_pressure_density_temp
from plav.simulator import get_gravity
from plav.plav import Plav

def trim_glider_hddot0(
    airspeed: float,
    altitude: float,
    model: AircraftConfig,
    bank_phi: float = 0.0,                 # rad
    guess: Optional[Dict[str, float]] = None,
    angle_bounds_deg: Optional[Dict[str, Tuple[float, float]]] = None,
):
    """
    Solve for [alpha, beta, de, da, dr] at given V, h, φ with p=q=r=0 and h_ddot=0.

    Inputs:
      V (m/s), h (m), bank_phi (rad)
      model: AircraftConfig with get_forces() defined
      guess: dict with seeds in degrees for alpha_deg, beta_deg, de_deg, da_deg, dr_deg
      angle_bounds_deg: optional bounds per angle in degrees

    Returns: dict with trim solution, including derived gamma, sink rate, forces/moments.
    """
    gravity = get_gravity(phi = 0.70, h = altitude) #NJ
    pdt = get_pressure_density_temp(altitude)
    air_density = pdt[1] 

    dynamic_pressure = 0.5 * air_density * airspeed**2
    weight = gravity * model.get_mass()

    cos_phi = np.cos(bank_phi)
    tan_phi = np.tan(bank_phi)

    lift_req = (weight * cos_phi)

    bounds_default = {
        "alpha": (-12.0, 18.0),
        "beta":  (-10.0, 10.0),
        "de":    (-25.0, 25.0),
        "da":    (-20.0, 20.0),
        "dr":    (-25.0, 25.0),
    }
    if angle_bounds_deg:
        bounds_default.update(angle_bounds_deg)

    x0 = np.zeros(5) #guesses zero for now

    lower_bound = np.array([
        np.deg2rad(bounds_default["alpha"][0]),
        np.deg2rad(bounds_default["beta"][0]),
        np.deg2rad(bounds_default["de"][0]),
        np.deg2rad(bounds_default["da"][0]),
        np.deg2rad(bounds_default["dr"][0]),
    ])
    upper_bound = np.array([
        np.deg2rad(bounds_default["alpha"][1]),
        np.deg2rad(bounds_default["beta"][1]),
        np.deg2rad(bounds_default["de"][1]),
        np.deg2rad(bounds_default["da"][1]),
        np.deg2rad(bounds_default["dr"][1]),
    ])

    # Residuals
    def resid(x):
        alpha, beta, de, da, dr = x

        velocity = alpha_beta_to_velocity(np.float64(airspeed),np.float64( alpha), np.float64(beta))

        #update model, then get forces
        model.update_conditions(
            np.float64(altitude),
            np.array(velocity,'d'),
            np.array([0.0,0.0,0.0],'d'),
            np.float64(air_density),
            np.float64(pdt[2]),
            np.float64(343.0)
        )

        control_vec = np.array([dr, da, de, 0.0],'d')
        model.update_control(control_vec)

        body_forces, body_moments = model.get_forces()
        Fx = body_forces[0]
        Fy = body_forces[1]
        Fz = body_forces[2]

        Mx = body_moments[0]
        My = body_moments[1]
        Mz = body_moments[2]

        #we want moments to be zero but lift must equal weight
        lift = -np.cos(alpha) * Fz + np.sin(alpha) * Fx
        r_lift = lift - weight

        return np.array([Mx, My, Mz, r_lift, Fy], 'd')
    
    sol = least_squares(resid, x0, bounds=(lower_bound, upper_bound))
    alpha, beta, de, da, dr = sol.x

    alpha, beta, de, da, dr = sol.x

    #update model, then get forces
    velocity = alpha_beta_to_velocity(np.float64(airspeed),np.float64( alpha), np.float64(beta))
    model.update_conditions(
            np.float64(altitude),
            np.array(velocity,'d'),
            np.array([0.0,0.0,0.0],'d'),
            np.float64(air_density),
            np.float64(pdt[2]),
            np.float64(343.0)
        )

    body_forces, body_moments = model.get_forces()
    lift = -np.cos(alpha) * body_forces[2] + np.sin(alpha) * body_forces[0]
    drag = -np.sin(alpha) * body_forces[2] - np.cos(alpha) * body_forces[0]

    glide_ratio = lift/drag

    return {
        "converged": bool(sol.success),
        "message": sol.message,
        "iterations": int(sol.nfev),
        # Trim state
        "alpha_deg": float(np.rad2deg(alpha)),
        "beta_deg":  float(np.rad2deg(beta)),
        "de_deg":    float(de),
        "da_deg":    float(da),
        "dr_deg":    float(dr),
        "bank_phi_deg": float(np.rad2deg(bank_phi)),
        # Derived kinematics
        "glide_ratio": float(glide_ratio),
        # Aero / forces / moments
    }
