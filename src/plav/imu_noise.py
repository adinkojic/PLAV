"""Test file to add noise to the IMU"""\

import numpy as np
from numba.experimental import jitclass
from numba import jit, float64

def deg_to_rad(x): return np.deg2rad(x)
def dph_to_radps(x):  # deg/hour -> rad/s
    return np.deg2rad(x) / 3600.0
def ug_to_mps2_per_sqrtHz(x_ug_per_sqrtHz):  # µg/√Hz -> m/s^2/√Hz
    return x_ug_per_sqrtHz * 1e-6 * 9.80665

spec_gm = (
    ('tau', float64),
    ('sigma_ss', float64),
    ('dt', float64),
    ('b', float64[:]),

    ('phi', float64),
    ('q', float64),
)
@jitclass(spec_gm)
class GaussMarkovBias:
    """Basically IMU random walk"""
    def __init__(self, tau, sigma_ss, dt):
        self.tau = tau
        self.sigma_ss = sigma_ss
        self.dt = dt
        self.b = np.zeros(3)

        self.phi = np.exp(-self.dt / self.tau) if self.tau > 0 else 0.0
        self.q = (1.0 - self.phi**2) * self.sigma_ss**2

    def step(self):
        """Return bias"""
        self.b = self.phi * self.b + np.random.normal(0.0, np.sqrt(self.q), size=3)
        return self.b

spec_imu = (
    ('fs', float64),
    ('gyro_nd', float64),
    ('accel_nd', float64),
    ('gyro_bias_sigma', float64),
    ('gyro_bias_tau', float64),
    ('accel_bias_sigma', float64),
    ('accel_bias_tau', float64),
    ('scale_err_g', float64[:]),
    ('scale_err_a', float64[:]),
    ('cross_g', float64[:,:]),
    ('cross_a', float64[:,:]),
    ('g_sens', float64[:,:]),
    ('lsb_gyro', float64),
    ('lsb_accel', float64),

    ('dt', float64),
    ('sigma_g', float64),
    ('sigma_a', float64),
    ('bg', GaussMarkovBias.class_type.instance_type),
    ('ba', GaussMarkovBias.class_type.instance_type),
    ('Sa', float64[:, :]),
    ('Sg', float64[:, :]),
)

@jitclass(spec_imu)
class IMUNoise:
    """IMU Noise model"""
    def __init__(self, fs, gyro_nd, accel_nd, gyro_bias_sigma, gyro_bias_tau, \
                 accel_bias_sigma, accel_bias_tau):
        self.fs = fs  # sample rate [Hz]
        # White noise densities (per √Hz) in SI units:
        self.gyro_nd = gyro_nd      # [rad/s/√Hz]
        self.accel_nd = accel_nd     # [m/s^2/√Hz]
        # Bias models (steady-state sigma in SI units, tau in seconds):
        self.gyro_bias_sigma = gyro_bias_sigma
        self.gyro_bias_tau = gyro_bias_tau
        self.accel_bias_sigma = accel_bias_sigma
        self.accel_bias_tau = accel_bias_tau
        # Optional small systematic terms:
        self.scale_err_g = np.zeros(3)   # fractional (e.g., 200 ppm -> 200e-6)
        self.scale_err_a = np.zeros(3)
        self.cross_g = np.zeros((3,3))   # off-diagonal small couplings
        self.cross_a = np.zeros((3,3))
        self.g_sens = np.zeros((3,3))    # gyro g-sensitivity [ (rad/s) / (m/s^2) ]
        self.lsb_gyro = 0.0                   # set to 0 to disable quantization
        self.lsb_accel = 0.0

        self.dt = 1.0 / self.fs
        # per-sample white noise std (assuming one-sided density to Nyquist)
        self.sigma_g = self.gyro_nd * np.sqrt(self.fs / 2.0)
        self.sigma_a = self.accel_nd * np.sqrt(self.fs / 2.0)
        self.bg = GaussMarkovBias(self.gyro_bias_tau, self.gyro_bias_sigma, self.dt)
        self.ba = GaussMarkovBias(self.accel_bias_tau, self.accel_bias_sigma, self.dt)
        self.Sg = np.diag(1.0 + self.scale_err_g) + self.cross_g
        self.Sa = np.diag(1.0 + self.scale_err_a) + self.cross_a

    def step(self, omega_true, accel_true):
        """Give true values, return noisy measurements"""
        # Update biases
        b_g = self.bg.step()
        b_a = self.ba.step()
        # White noise
        n_g = np.random.normal(0.0, self.sigma_g, size=3)
        n_a = np.random.normal(0.0, self.sigma_a, size=3)
        # Apply systematic terms
        omega_meas = self.Sg @ omega_true + b_g + n_g + (self.g_sens @ accel_true)
        accel_meas = self.Sa @ accel_true + b_a + n_a
        # Quantization (optional)
        if self.lsb_gyro > 0:
            omega_meas = np.round(omega_meas / self.lsb_gyro) * self.lsb_gyro
        if self.lsb_accel > 0:
            accel_meas = np.round(accel_meas / self.lsb_accel) * self.lsb_accel
        return omega_meas, accel_meas

#imu = IMUNoise(
#    fs=1000.0,
#    gyro_nd=deg_to_rad(0.0038),
#    accel_nd=ug_to_mps2_per_sqrtHz(70.0),
#    gyro_bias_sigma=dph_to_radps(10.0),
#    gyro_bias_tau=500.0,
#    accel_bias_sigma=1.0e-2*9.80665,
#    accel_bias_tau=500.0,
#)

# true_accel = np.array([0.0, 0.0, -9.81])
# true_gyro = np.array([0.0, 0.0, 0.0])

# print(imu.step(true_gyro, true_accel))
