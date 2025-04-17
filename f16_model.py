"""F16 File from F16_S119
it's all lookup tables"""
import math

import numpy as np
from numba import float64
from numba.experimental import jitclass
from numba import jit
import quaternion_math as quat

from aircraftconfig import get_dynamic_viscosity, velocity_to_alpha_beta, get_wind_to_body_axis


spec = [
    #deflections, d prefix is normalized -1 to 1
    ('rdr', float64),
    ('ail', float64),
    ('el', float64),

    #geometrics
    ('mass', float64),
    ('cmac', float64),
    ('Sref', float64),
    ('bref', float64),
    ('inertiamatrix', float64[:,:]),
    ('cp_wrt_cm', float64[:]),

    #enviromentals
    ('altitude', float64),
    ('velocity', float64[:]),
    ('airspeed', float64),
    ('alpha', float64),
    ('beta', float64),
    ('reynolds', float64),
    ('omega', float64[:]),
    ('density', float64),
    ('temperature', float64),
    ('mach', float64),
]

@jit(float64(float64, float64, float64[:], float64[:], float64[:,:]))
def bilinear_interp(x, y, x_grid, y_grid, z_grid):
    """Billinear interpolation with flat extrapolation
    https://en.wikipedia.org/wiki/Bilinear_interpolation"""
    if x < x_grid[0]:
        x = x_grid[0]
    if x > x_grid[-1]:
        x = x_grid[-1]
    if y < y_grid[0]:
        y = y_grid[0]
    if y > y_grid[-1]:
        y = y_grid[-1]

    x_index = np.searchsorted(x_grid, x) -1
    y_index = np.searchsorted(y_grid, y) -1

    x1 = x_grid[x_index]
    x2 = x_grid[x_index+1]
    y1 = y_grid[y_index]
    y2 = y_grid[y_index+1]

    q11 = z_grid[y_index,   x_index  ]
    q12 = z_grid[y_index+1, x_index  ]
    q21 = z_grid[y_index,   x_index+1]
    q22 = z_grid[y_index+1, x_index+1]
    
    denom = (x2-x1)*(y2-y1)
    numer = (q11*(x2-x)*(y2-y) + q12*(x2-x)*(y-y1) + q21*(x-x1)*(y2-y) + q22*(x-x1)*(y-y1))

    return numer/denom

@jitclass(spec)
class F16_aircraft(object): #TODO: fix possible beta issue
    """Object used to lookup coefficients for F16 jet"""
    def __init__(self):
        self.rdr  = 0.0
        self.ail  = 0.0
        self.el   = -3.25


        self.mass = 637.1595 * 14.594 #kg
        self.cmac = 11.32 / (39.37/12) #m
        self.Sref = 300 / (39.37/12) **2 #m^2
        self.bref = 30.0 / (39.37/12) #m

        self.inertiamatrix = 1.3558179619 * np.array([
            [9496.0, 0,982.0 ],
            [0, 55814.0, 0],
            [982.0, 0, 63100.0]
        ]) #kg m^2

        self.altitude = 0.0
        self.velocity = np.zeros(3)
        self.omega = np.zeros(3)
        self.airspeed = 0
        self.alpha = 0
        self.beta = 0
        self.reynolds = 0
        self.density = 0
        self.temperature = 0
        self.mach = 0

        self.cp_wrt_cm = self.get_aero_center_wrt_cm()

    def update_conditions(self, altitude, velocity, omega, density, temperature, speed_of_sound):
        """Update altitude and velocity it thinks it's at
        Call this before every get_forces()"""
        self.altitude = altitude
        self.velocity = velocity
        self.omega = omega

        self.density = density
        self.temperature = temperature


        self.airspeed, self.alpha, self.beta = velocity_to_alpha_beta(velocity)

        dynamic_viscosity = get_dynamic_viscosity(temperature)
        self.reynolds = self.get_Re(density, dynamic_viscosity)

        self.mach = self.airspeed/speed_of_sound

    def get_forces(self):
        """Gets forces on aircraft from state and known derivatives"""

        rtd = 180/math.pi


        C_X,C_Y,C_Z, C_l, C_m, C_n = self.get_coeff()

        qbar = 0.5 * self.density *self.airspeed**2

        #body_lift = C_L * qbar * self.Sref
        #body_drag = C_D * qbar * self.Sref
        body_x = C_X * qbar * self.Sref
        body_y = C_Y * qbar * self.Sref
        body_z = C_Z * qbar * self.Sref
        body_pitching_moment = C_m * qbar * self.Sref * self.cmac
        body_yawing_moment   = C_n * qbar * self.Sref * self.bref
        body_rolling_moment  = C_l * qbar * self.Sref * self.bref

        #wind_to_body = get_wind_to_body_axis(self.alpha, self.beta)

        #body_forces_wind = np.array([-body_drag, body_side, -body_lift])
        body_forces_body = np.array([body_x, body_y, body_z])

        moments = np.array([body_rolling_moment, body_pitching_moment, body_yawing_moment])


        return body_forces_body, moments

    def get_coeff(self):
        """Gets the aerodynamic coefficients of the F16
        alpha and beta are in degrees"""


        bspan=30.0
        cbar=11.32
        dele=self.el/25.0
        dail=self.ail/20.0
        drdr=self.rdr/30.0
        
        rtd = 180/math.pi

        alpha = self.alpha * rtd #object alpha is rad, this one is degrees
        beta = self.beta * rtd

        cxt=cx_lookup(alpha,self.el) # implement table lookup
        cy=-0.02*beta+0.021*dail+0.086*drdr
        czt=cz_lookup(alpha)
        cz=czt*(1.-(beta/rtd)**2)-0.19*dele

        clt=cl_lookup(alpha,beta)
        dclda=dlda_lookup(alpha,beta)
        dcldr=dldr_lookup(alpha,beta)
        cl=clt + dclda*dail + dcldr*drdr
        cmt=cm_lookup(alpha,self.el)
        cnt=cn_lookup(alpha,beta)
        dcnda=dnda_lookup(alpha,beta)
        dcndr=dndr_lookup(alpha,beta)
        cn=cnt + dcnda*dail + dcndr*drdr
        
        # Add damping derivative contributions
        #  and cg position terms.

        p = self.omega[0]
        q = self.omega[1]
        r = self.omega[2]
        tvt=2*self.airspeed
        b2v=bspan/tvt
        cq2v=cbar*q/tvt

        #note dependence of alpha and body rates on some of these
        cx=cxt + cq2v* cxq_lookup(alpha)
        cy=cy + b2v*(cyp_lookup(alpha)*p + cyr_lookup(alpha)*r)
        cz=cz + cq2v*czq_lookup(alpha)
        cl=cl + b2v*(clp_lookup(alpha)*p + clr_lookup(alpha)*r)
        cm=cmt + cq2v*cmq_lookup(alpha)
        cn=cn + b2v*(cnp_lookup(alpha)*p + cnr_lookup(alpha)*r)

        return cx, cy, cz, cl, cm, cn

    def get_aero_center_wrt_cm(self):
        """get the position of the aerodynamic center
        with respect to center of mass
        returns meters"""
        CG_PCT_MAC = 25 #percent
        CBAR = 11.32

        aero_pos_ft = (35 - CG_PCT_MAC)*CBAR/100

        return np.array([-aero_pos_ft/ (39.37/12), 0, 0])
    
    def get_xcp(self):
        """returns x_cp with respect to CM"""
        return self.cp_wrt_cm

    def get_inertia_matrix(self):
        """Returns inertia matrix as 2d np array"""
        return np.ascontiguousarray(self.inertiamatrix)

    def get_mass(self):
        """Returns mass in kg"""
        return self.mass

    def get_Re(self, density, viscosity):
        """Gets reynolds number from given conditions"""
        return self.airspeed * self.cmac * density/viscosity
    
    def get_alpha(self):
        """Returns alpha in rad"""
        return self.alpha
    
    def get_beta(self):
        """Returns beta in rad"""
        return self.beta
    
    def get_mach(self):
        """Returns mach [nd]"""
        return self.mach
    
    def get_qbar(self):
        """Returns dyanmic pressure [Pa]"""
        return 0.5 * self.density *self.airspeed**2
    
    def get_airspeed(self):
        """Returns airspeed [m/s]"""
        return self.airspeed
    
    def get_reynolds(self):
        """Returns Reynolds Number"""
        return self.reynolds

    def input_control(self, elevator, aileron, rudder):
        """Input the control deflections in degrees of the elevator, aileron, and rudder"""
        self.el = elevator
        self.ail = aileron
        self.rdr = rudder


@jit(float64[:]())
def get_alpha_table():
    """Get Alpha for lookups"""
    alpha_table = np.array([-10., -5., 0., 5., 10., 15., 20., 25., 30., 35., 40., 45.], 'd')
    return alpha_table

@jit(float64[:]())
def get_el_table():
    """get elevator deflection table for lookups"""
    elevator_table = np.array([-24., -12., 0., 12., 24.], 'd')
    return elevator_table

@jit(float64[:]())
def get_beta_table():
    """gets beta range for lookups"""
    beta_table = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    return beta_table

@jit(float64[:]())
def get_beta2_table():
    """other beta table for control stuff"""
    beta_table = np.array([-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0], 'd')
    return beta_table

@jit(float64(float64, float64))
def cx_lookup(alpha, el):
    """Lookup for C_X, based on Alpha and Elevator Deflections [deg], [deg]"""
    alpha_table = get_alpha_table()
    elevator_table = get_el_table()
    table = np.array([
        [-.099,-.081,-.081,-.063,-.025,.044,.097,.113,.145,.167,.174,.166],
        [-.048,-.038,-.040,-.021,.016,.083,.127,.137,.162,.177,.179,.167],
        [-.022,-.020,-.021,-.004,.032,.094,.128,.130,.154,.161,.155,.138],
        [-.040,-.038,-.039,-.025,.006,.062,.087,.085,.100,.110,.104,.091],
        [-.083,-.073,-.076,-.072,-.046,.012,.024,.025,.043,.053,.047,.040]
    ], 'd')
    C_X = bilinear_interp(alpha, el, alpha_table, elevator_table, table)
    return C_X

@jit(float64(float64, float64))
def cm_lookup(alpha, el):
    """Lookup for C_m, based on Alpha and Elevator Deflections [deg], [deg]"""
    alpha_table = get_alpha_table()
    elevator_table = get_el_table()
    table = np.array([
        [.205,.168,.186,.196,.213,.251,.245,.238,.252,.231,.198,.192],
        [.081,.077,.107,.110,.110,.141,.127,.119,.133,.108,.081,.093],
        [-.046,-.020,-.009,-.005,-.006,.010,.006,-.001,.014,.000,-.013,.032],
        [-.174,-.145,-.121,-.127,-.129,-.102,-.097,-.113,-.087,-.084,-.069,-.006],
        [-.259,-.202,-.184,-.193,-.199,-.150,-.160,-.167,-.104,-.076,-.041,-.005]
    ], 'd')
    C_m = bilinear_interp(alpha, el, alpha_table, elevator_table, table)
    return C_m

@jit(float64(float64, float64))
def cl_lookup(alpha, beta):
    """Lookup for C_l, based on Alpha and Beta [deg], [deg]"""
    alpha_table = get_alpha_table()
    beta_table = get_beta_table()
    table = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.0010, -0.0040, -0.0080, -0.012, -0.016, -0.022, -0.022, -0.021, -0.015, -0.0080, -0.013, -0.015],
        [-0.0030, -0.0090, -0.017, -0.024, -0.03, -0.041, -0.045, -0.04, -0.016, -0.0020, -0.01, -0.019],
        [-0.0010, -0.01, -0.02, -0.03, -0.039, -0.054, -0.057, -0.054, -0.023, -0.0060, -0.014, -0.027],
        [0.0, -0.01, -0.022, -0.034, -0.047, -0.06, -0.069, -0.067, -0.033, -0.036, -0.035, -0.035],
        [0.0070, -0.01, -0.023, -0.034, -0.049, -0.063, -0.081, -0.079, -0.06, -0.058, -0.062, -0.059],
        [0.0090, -0.011, -0.023, -0.037, -0.05, -0.068, -0.089, -0.088, -0.091, -0.076, -0.077, -0.076]
    ], 'd')
    C_l = bilinear_interp(alpha, beta, alpha_table, beta_table, table)
    return C_l

@jit(float64(float64, float64))
def cn_lookup(alpha, beta):
    """Lookup for C_l, based on Alpha and Beta [deg], [deg]"""
    alpha_table = get_alpha_table()
    beta_table = get_beta_table()
    table = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.018, 0.019, 0.018, 0.019, 0.019, 0.018, 0.013, 0.0070, 0.0040, -0.014, -0.017, -0.033],
        [0.038, 0.042, 0.042, 0.042, 0.043, 0.039, 0.03, 0.017, 0.0040, -0.035, -0.047, -0.057],
        [0.056, 0.057, 0.059, 0.058, 0.058, 0.053, 0.032, 0.012, 0.0020, -0.046, -0.071, -0.073],
        [0.064, 0.077, 0.076, 0.074, 0.073, 0.057, 0.029, 0.0070, 0.012, -0.034, -0.065, -0.041],
        [0.074, 0.086, 0.093, 0.089, 0.08, 0.062, 0.049, 0.022, 0.028, -0.012, -0.0020, -0.013],
        [0.079, 0.09, 0.106, 0.106, 0.096, 0.08, 0.068, 0.03, 0.064, 0.015, 0.011, -0.0010]
    ], 'd')
    C_n = bilinear_interp(alpha, beta, alpha_table, beta_table, table)
    return C_n

@jit(float64(float64))
def cz_lookup(alpha):
    """Lookup for C_Z, based on Alpha and Elevator Deflections [deg]"""
    alpha_table = get_alpha_table()
    table = np.array([.770,.241,-.100,-.416,-.731,-1.053, -1.366,\
                        -1.646,-1.917,-2.120,-2.248,-2.229], 'd')
    C_Z = np.interp(alpha, alpha_table, table)
    return C_Z

@jit(float64(float64))
def cxq_lookup(alpha):
    """Lookup for C_Xq, based on Alpha and Elevator Deflections [deg]"""
    alpha_table = get_alpha_table()
    table = np.array([-.267, -.110, .308, 1.34, 2.08, 2.91, 2.76, 2.05, 1.50, 1.49, 1.83, 1.21], 'd')
    C_Xq = np.interp(alpha, alpha_table, table)
    return C_Xq

@jit(float64(float64))
def cyr_lookup(alpha):
    """Lookup for C_Yr, based on Alpha and Elevator Deflections [deg]"""
    alpha_table = get_alpha_table()
    table = np.array([.882, .852, .876, .958, .962, .974, .819, .483, .590, 1.21, -.493, -1.04], 'd')
    C_Yr = np.interp(alpha, alpha_table, table)
    return C_Yr

@jit(float64(float64))
def cyp_lookup(alpha):
    """Lookup for C_Yp, based on Alpha and Elevator Deflections [deg]"""
    alpha_table = get_alpha_table()
    table = np.array([-.108, -.108, -.188, .110, .258, .226, .344, .362, .611, .529, .298, -.227], 'd')
    C_Yp = np.interp(alpha, alpha_table, table)
    return C_Yp

@jit(float64(float64))
def czq_lookup(alpha):
    """Lookup for C_Zq, based on Alpha and Elevator Deflections [deg]"""
    alpha_table = get_alpha_table()
    table = np.array([-8.80, -25.8, -28.9, -31.4, -31.2, -30.7, -27.7, -28.2, -29.0, -29.8, -38.3, -35.3], 'd')
    C_Zq = np.interp(alpha, alpha_table, table)
    return C_Zq

@jit(float64(float64))
def clr_lookup(alpha):
    """Lookup for C_lr, based on Alpha and Elevator Deflections [deg]"""
    alpha_table = get_alpha_table()
    table = np.array([-.126, -.026, .063, .113, .208, .230, .319, .437, .680, .100, .447, -.330], 'd')
    C_lr = np.interp(alpha, alpha_table, table)
    return C_lr

@jit(float64(float64))
def clp_lookup(alpha):
    """Lookup for C_lp, based on Alpha and Elevator Deflections [deg]"""
    alpha_table = get_alpha_table()
    table = np.array([-.360, -.359, -.443, -.420, -.383, -.375, -.329, -.294, -.230, -.210, -.120, -.100], 'd')
    C_lp = np.interp(alpha, alpha_table, table)
    return C_lp

@jit(float64(float64))
def cmq_lookup(alpha):
    """Lookup for C_mq, based on Alpha and Elevator Deflections [deg]"""
    alpha_table = get_alpha_table()
    table = np.array([-7.21, -5.40, -5.23, -5.26, -6.11, -6.64, -5.69, -6.00, -6.20, -6.40, -6.60, -6.00], 'd')
    C_mq = np.interp(alpha, alpha_table, table)
    return C_mq

@jit(float64(float64))
def cnr_lookup(alpha):
    """Lookup for C_mq, based on Alpha and Elevator Deflections [deg]"""
    alpha_table = get_alpha_table()
    table = np.array([-.380, -.363, -.378, -.386, -.370, -.453, -.550, -.582, -.595, -.637, -1.02, -.840], 'd')
    C_nr = np.interp(alpha, alpha_table, table)
    return C_nr

@jit(float64(float64))
def cnp_lookup(alpha):
    """Lookup for C_mq, based on Alpha and Elevator Deflections [deg]"""
    alpha_table = get_alpha_table()
    table = np.array([.061, .052, .052, -.012, -.013, -.024, .050, .150, .130, .158, .240, .150], 'd')
    C_np = np.interp(alpha, alpha_table, table)
    return C_np

@jit(float64(float64,float64))
def dlda_lookup(alpha, beta):
    """Lookup for C_dlda, based on Alpha and Beta [deg], [deg]"""
    alpha_table = get_alpha_table()
    beta_table = get_beta2_table()
    table = np.array([
        [-0.041, -0.052, -0.053, -0.056, -0.05, -0.056, -0.082, -0.059, -0.042, -0.038, -0.027, -0.017],
        [-0.041, -0.053, -0.053, -0.053, -0.05, -0.051, -0.066, -0.043, -0.038, -0.027, -0.023, -0.016],
        [-0.042, -0.053, -0.052, -0.051, -0.049, -0.049, -0.043, -0.035, -0.026, -0.016, -0.018, -0.014],
        [-0.04, -0.052, -0.051, -0.052, -0.048, -0.048, -0.042, -0.037, -0.031, -0.026, -0.017, -0.012],
        [-0.043, -0.049, -0.048, -0.049, -0.043, -0.042, -0.042, -0.036, -0.025, -0.021, -0.016, -0.011],
        [-0.044, -0.048, -0.048, -0.047, -0.042, -0.041, -0.02, -0.028, -0.013, -0.014, -0.011, -0.01],
        [-0.043, -0.049, -0.047, -0.045, -0.042, -0.037, -0.0030, -0.013, -0.01, -0.0030, -0.0070, -0.0080]
    ], 'd')
    dlda = bilinear_interp(alpha, beta, alpha_table, beta_table, table)
    return dlda

@jit(float64(float64,float64))
def dldr_lookup(alpha, beta):
    """Lookup for C_dldr, based on Alpha and Beta [deg], [deg]"""
    alpha_table = get_alpha_table()
    beta_table = get_beta2_table()
    table = np.array([
        [0.0050, 0.017, 0.014, 0.01, -0.0050, 0.0090, 0.019, 0.0050, 0.0, -0.0050, -0.011, 0.0080],
        [0.0070, 0.016, 0.014, 0.014, 0.013, 0.0090, 0.012, 0.0050, 0.0, 0.0040, 0.0090, 0.0070],
        [0.013, 0.013, 0.011, 0.012, 0.011, 0.0090, 0.0080, 0.0050, 0.0, 0.0050, 0.0030, 0.0050],
        [0.018, 0.015, 0.015, 0.014, 0.014, 0.014, 0.014, 0.015, 0.013, 0.011, 0.0060, 0.0010],
        [0.015, 0.014, 0.013, 0.013, 0.012, 0.011, 0.011, 0.01, 0.0080, 0.0080, 0.0070, 0.0030],
        [0.021, 0.011, 0.01, 0.011, 0.01, 0.0090, 0.0080, 0.01, 0.0060, 0.0050, 0.0, 0.0010],
        [0.023, 0.01, 0.011, 0.011, 0.011, 0.01, 0.0080, 0.01, 0.0060, 0.014, 0.02, 0.0]
    ], 'd')
    dldr = bilinear_interp(alpha, beta, alpha_table, beta_table, table)
    return dldr

@jit(float64(float64,float64))
def dnda_lookup(alpha, beta):
    """Lookup for C_dldr, based on Alpha and Beta [deg], [deg]"""
    alpha_table = get_alpha_table()
    beta_table = get_beta2_table()
    table = np.array([
        [0.0010, -0.027, -0.017, -0.013, -0.012, -0.016, 0.0010, 0.017, 0.011, 0.017, 0.0080, 0.016],
        [0.0020, -0.014, -0.016, -0.016, -0.014, -0.019, -0.021, 0.0020, 0.012, 0.016, 0.015, 0.011],
        [-0.0060, -0.0080, -0.0060, -0.0060, -0.0050, -0.0080, -0.0050, 0.0070, 0.0040, 0.0070, 0.0060, 0.0060],
        [-0.011, -0.011, -0.01, -0.0090, -0.0080, -0.0060, 0.0, 0.0040, 0.0070, 0.01, 0.0040, 0.01],
        [-0.015, -0.015, -0.014, -0.012, -0.011, -0.0080, -0.0020, 0.0020, 0.0060, 0.012, 0.011, 0.011],
        [-0.024, -0.01, -0.0040, -0.0020, -0.0010, 0.0030, 0.014, 0.0060, -0.0010, 0.0040, 0.0040, 0.0060],
        [-0.022, 0.0020, -0.0030, -0.0050, -0.0030, -0.0010, -0.0090, -0.0090, -0.0010, 0.0030, -0.0020, 0.0010]
    ], 'd')
    dnda = bilinear_interp(alpha, beta, alpha_table, beta_table, table)
    return dnda

@jit(float64(float64,float64))
def dndr_lookup(alpha, beta):
    """Lookup for C_dldr, based on Alpha and Beta [deg], [deg]"""
    alpha_table = get_alpha_table()
    beta_table = get_beta2_table()
    table = np.array([
        [-0.018, -0.052, -0.052, -0.052, -0.054, -0.049, -0.059, -0.051, -0.03, -0.037, -0.026, -0.013],
        [-0.028, -0.051, -0.043, -0.046, -0.045, -0.049, -0.057, -0.052, -0.03, -0.033, -0.03, -0.0080],
        [-0.037, -0.041, -0.038, -0.04, -0.04, -0.038, -0.037, -0.03, -0.027, -0.024, -0.019, -0.013],
        [-0.048, -0.045, -0.045, -0.045, -0.044, -0.045, -0.047, -0.048, -0.049, -0.045, -0.033, -0.016],
        [-0.043, -0.044, -0.041, -0.041, -0.04, -0.038, -0.034, -0.035, -0.035, -0.029, -0.022, -0.0090],
        [-0.052, -0.034, -0.036, -0.036, -0.035, -0.028, -0.024, -0.023, -0.02, -0.016, -0.01, -0.014],
        [-0.062, -0.034, -0.027, -0.028, -0.027, -0.027, -0.023, -0.023, -0.019, -0.0090, -0.025, -0.01]
    ], 'd')
    dndr = bilinear_interp(alpha, beta, alpha_table, beta_table, table)
    return dndr