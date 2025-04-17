"""F16 File from F16_S119"""
import math

import numpy as np
from numba import float64
from numba.experimental import jitclass
from numba import jit


spec = [
    #deflections, d prefix is normalized -1 to 1
    ('rdr', float64),
    ('ail', float64),
    ('el', float64),
]

@jit
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
class F16_Lookup(object):
    """Object used to lookup coefficients for F16 jet"""
    def __init__(self):
        self.rdr  = 0.0
        self.ail  = 0.0
        self.el   = 0.0

    def input_control(self, elevator, aileron, rudder):
        """Input the control deflections in degrees of the elevator, aileron, and rudder"""
        self.el = elevator
        self.ail = aileron
        self.rdr = rudder

    def get_coeff(self, alpha, beta, el, ail, rdr, airspeed, pqr):
        """Gets the aerodynamic coefficients of the F16
        alpha and beta are in degrees"""

        bspan = 30.0
        cbar=11.32
        dele=self.el/25.0
        dail=self.ail/20.0
        drdr=self.rdr/30.0
        
        rtd = 180/math.pi

        cxt=self.cx_lookup(alpha,el) # implement table lookup
        cy=-0.02*beta+0.021*dail+0.086*drdr
        czt=self.cz_lookup(alpha)
        cz=czt*(1.-(beta/rtd)^2)-0.19*dele

        clt=self.cl_lookup(alpha,beta)
        dclda=self.dlda_lookup(alpha,beta)
        dcldr=self.dldr_lookup(alpha,beta)
        cl=clt + dclda*dail + dcldr*drdr
        cmt=self.cm_lookup(alpha,el)
        cnt=self.cn_lookup(alpha,beta)
        dcnda=self.dnda_lookup(alpha,beta)
        dcndr=self.dndr_lookup(alpha,beta);
        cn=cnt + dcnda*dail + dcndr*drdr;
        
        # Add damping derivative contributions
        #  and cg position terms.

        p = pqr[0]
        q = pqr[1]
        r = pqr[2]
        
        tvt=2*airspeed
        b2v=bspan/tvt
        cq2v=cbar*q/tvt

        #note dependence of alpha and body rates on some of these
        cx=cxt + cq2v* self.cxq_lookup(alpha)
        cy=cy + b2v*(self.cyp_lookup(alpha)*p + self.cyr_lookup(alpha)*r)
        cz=cz + cq2v*self.czq_lookup(alpha)
        cl=cl + b2v*(self.clp_lookup(alpha)*p + self.clr_lookup(alpha)*r)
        cm=cmt + cq2v*self.cmq_lookup(alpha)
        cn=cn + b2v*(self.cnp_lookup(alpha)*p + self.cnr_lookup(alpha)*r)

        return cx, cy, cz, cl, cm, cn

    def get_alpha_table(self):
        """Get Alpha for lookups"""
        alpha_table = np.array([-10., -5., 0., 5., 10., 15., 20., 25., 30., 35., 40., 45.])
        return alpha_table
    
    def get_el_table(self):
        """get elevator deflection table for lookups"""
        elevator_table = np.array([-24., -12., 0., 12., 24.])
        return elevator_table
    
    def get_beta_table(self):
        """gets beta range for lookups"""
        beta_table = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
        return beta_table

    def get_beta2_table(self):
        """other beta table for control stuff"""
        beta_table = np.array([-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0])
        return beta_table

    def cx_lookup(self, alpha, el):
        """Lookup for C_X, based on Alpha and Elevator Deflections [deg], [deg]"""
        alpha_table = self.get_alpha_table()
        elevator_table = self.get_el_table()
        table = np.array([
            [-.099,-.081,-.081,-.063,-.025,.044,.097,.113,.145,.167,.174,.166],
            [-.048,-.038,-.040,-.021,.016,.083,.127,.137,.162,.177,.179,.167],
            [-.022,-.020,-.021,-.004,.032,.094,.128,.130,.154,.161,.155,.138],
            [-.040,-.038,-.039,-.025,.006,.062,.087,.085,.100,.110,.104,.091],
            [-.083,-.073,-.076,-.072,-.046,.012,.024,.025,.043,.053,.047,.040]
        ])
        C_X = bilinear_interp(alpha, el, alpha_table, elevator_table, table)
        return C_X
    
    def cm_lookup(self, alpha, el):
        """Lookup for C_m, based on Alpha and Elevator Deflections [deg], [deg]"""
        alpha_table = self.get_alpha_table()
        elevator_table = self.get_el_table()
        table = np.array([
            [.205,.168,.186,.196,.213,.251,.245,.238,.252,.231,.198,.192],
            [.081,.077,.107,.110,.110,.141,.127,.119,.133,.108,.081,.093],
            [-.046,-.020,-.009,-.005,-.006,.010,.006,-.001,.014,.000,-.013,.032],
            [-.174,-.145,-.121,-.127,-.129,-.102,-.097,-.113,-.087,-.084,-.069,-.006],
            [-.259,-.202,-.184,-.193,-.199,-.150,-.160,-.167,-.104,-.076,-.041,-.005]
        ])
        C_m = bilinear_interp(alpha, el, alpha_table, elevator_table, table)
        return C_m
    
    def cl_lookup(self, alpha, beta):
        """Lookup for C_l, based on Alpha and Beta [deg], [deg]"""
        alpha_table = self.get_alpha_table()
        beta_table = self.get_beta_table()
        table = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.0010, -0.0040, -0.0080, -0.012, -0.016, -0.022, -0.022, -0.021, -0.015, -0.0080, -0.013, -0.015],
            [-0.0030, -0.0090, -0.017, -0.024, -0.03, -0.041, -0.045, -0.04, -0.016, -0.0020, -0.01, -0.019],
            [-0.0010, -0.01, -0.02, -0.03, -0.039, -0.054, -0.057, -0.054, -0.023, -0.0060, -0.014, -0.027],
            [0.0, -0.01, -0.022, -0.034, -0.047, -0.06, -0.069, -0.067, -0.033, -0.036, -0.035, -0.035],
            [0.0070, -0.01, -0.023, -0.034, -0.049, -0.063, -0.081, -0.079, -0.06, -0.058, -0.062, -0.059],
            [0.0090, -0.011, -0.023, -0.037, -0.05, -0.068, -0.089, -0.088, -0.091, -0.076, -0.077, -0.076]
        ])
        C_l = bilinear_interp(alpha, beta, alpha_table, beta_table, table)
        return C_l
    
    def cn_lookup(self, alpha, beta):
        """Lookup for C_l, based on Alpha and Beta [deg], [deg]"""
        alpha_table = self.get_alpha_table()
        beta_table = self.get_beta_table()
        table = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.018, 0.019, 0.018, 0.019, 0.019, 0.018, 0.013, 0.0070, 0.0040, -0.014, -0.017, -0.033],
            [0.038, 0.042, 0.042, 0.042, 0.043, 0.039, 0.03, 0.017, 0.0040, -0.035, -0.047, -0.057],
            [0.056, 0.057, 0.059, 0.058, 0.058, 0.053, 0.032, 0.012, 0.0020, -0.046, -0.071, -0.073],
            [0.064, 0.077, 0.076, 0.074, 0.073, 0.057, 0.029, 0.0070, 0.012, -0.034, -0.065, -0.041],
            [0.074, 0.086, 0.093, 0.089, 0.08, 0.062, 0.049, 0.022, 0.028, -0.012, -0.0020, -0.013],
            [0.079, 0.09, 0.106, 0.106, 0.096, 0.08, 0.068, 0.03, 0.064, 0.015, 0.011, -0.0010]
        ])
        C_n = bilinear_interp(alpha, beta, alpha_table, beta_table, table)
        return C_n
    
    def cz_lookup(self, alpha):
        """Lookup for C_Z, based on Alpha and Elevator Deflections [deg]"""
        alpha_table = self.get_alpha_table()
        table = np.array([.770,.241,-.100,-.416,-.731,-1.053, -1.366,\
                          -1.646,-1.917,-2.120,-2.248,-2.229])
        C_Z = np.interp(alpha, alpha_table, table)
        return C_Z
    
    def cxq_lookup(self, alpha):
        """Lookup for C_Xq, based on Alpha and Elevator Deflections [deg]"""
        alpha_table = self.get_alpha_table()
        table = np.array([-.267, -.110, .308, 1.34, 2.08, 2.91, 2.76, 2.05, 1.50, 1.49, 1.83, 1.21])
        C_Xq = np.interp(alpha, alpha_table, table)
        return C_Xq
    
    def cyr_lookup(self, alpha):
        """Lookup for C_Yr, based on Alpha and Elevator Deflections [deg]"""
        alpha_table = self.get_alpha_table()
        table = np.array([.882, .852, .876, .958, .962, .974, .819, .483, .590, 1.21, -.493, -1.04])
        C_Yr = np.interp(alpha, alpha_table, table)
        return C_Yr
    
    def cyp_lookup(self, alpha):
        """Lookup for C_Yp, based on Alpha and Elevator Deflections [deg]"""
        alpha_table = self.get_alpha_table()
        table = np.array([-.108, -.108, -.188, .110, .258, .226, .344, .362, .611, .529, .298, -.227])
        C_Yp = np.interp(alpha, alpha_table, table)
        return C_Yp

    def czq_lookup(self, alpha):
        """Lookup for C_Zq, based on Alpha and Elevator Deflections [deg]"""
        alpha_table = self.get_alpha_table()
        table = np.array([-8.80, -25.8, -28.9, -31.4, -31.2, -30.7, -27.7, -28.2, -29.0, -29.8, -38.3, -35.3])
        C_Zq = np.interp(alpha, alpha_table, table)
        return C_Zq
    
    def clr_lookup(self, alpha):
        """Lookup for C_lr, based on Alpha and Elevator Deflections [deg]"""
        alpha_table = self.get_alpha_table()
        table = np.array([-.126, -.026, .063, .113, .208, .230, .319, .437, .680, .100, .447, -.330])
        C_lr = np.interp(alpha, alpha_table, table)
        return C_lr

    def clp_lookup(self, alpha):
        """Lookup for C_lp, based on Alpha and Elevator Deflections [deg]"""
        alpha_table = self.get_alpha_table()
        table = np.array([-.360, -.359, -.443, -.420, -.383, -.375, -.329, -.294, -.230, -.210, -.120, -.100])
        C_lp = np.interp(alpha, alpha_table, table)
        return C_lp
    
    def cmq_lookup(self, alpha):
        """Lookup for C_mq, based on Alpha and Elevator Deflections [deg]"""
        alpha_table = self.get_alpha_table()
        table = np.array([-7.21, -5.40, -5.23, -5.26, -6.11, -6.64, -5.69, -6.00, -6.20, -6.40, -6.60, -6.00])
        C_mq = np.interp(alpha, alpha_table, table)
        return C_mq
    
    def cnr_lookup(self, alpha):
        """Lookup for C_mq, based on Alpha and Elevator Deflections [deg]"""
        alpha_table = self.get_alpha_table()
        table = np.array([-.380, -.363, -.378, -.386, -.370, -.453, -.550, -.582, -.595, -.637, -1.02, -.840])
        C_nr = np.interp(alpha, alpha_table, table)
        return C_nr
    
    def cnp_lookup(self, alpha):
        """Lookup for C_mq, based on Alpha and Elevator Deflections [deg]"""
        alpha_table = self.get_alpha_table()
        table = np.array([.061, .052, .052, -.012, -.013, -.024, .050, .150, .130, .158, .240, .150])
        C_np = np.interp(alpha, alpha_table, table)
        return C_np
    
    def dlda_lookup(self, alpha, beta):
        """Lookup for C_dlda, based on Alpha and Beta [deg], [deg]"""
        alpha_table = self.get_alpha_table()
        beta_table = self.get_beta2_table()
        table = np.array([
            [-0.041, -0.052, -0.053, -0.056, -0.05, -0.056, -0.082, -0.059, -0.042, -0.038, -0.027, -0.017],
            [-0.041, -0.053, -0.053, -0.053, -0.05, -0.051, -0.066, -0.043, -0.038, -0.027, -0.023, -0.016],
            [-0.042, -0.053, -0.052, -0.051, -0.049, -0.049, -0.043, -0.035, -0.026, -0.016, -0.018, -0.014],
            [-0.04, -0.052, -0.051, -0.052, -0.048, -0.048, -0.042, -0.037, -0.031, -0.026, -0.017, -0.012],
            [-0.043, -0.049, -0.048, -0.049, -0.043, -0.042, -0.042, -0.036, -0.025, -0.021, -0.016, -0.011],
            [-0.044, -0.048, -0.048, -0.047, -0.042, -0.041, -0.02, -0.028, -0.013, -0.014, -0.011, -0.01],
            [-0.043, -0.049, -0.047, -0.045, -0.042, -0.037, -0.0030, -0.013, -0.01, -0.0030, -0.0070, -0.0080]
        ])
        dlda = bilinear_interp(alpha, beta, alpha_table, beta_table, table)
        return dlda
    
    def dldr_lookup(self, alpha, beta):
        """Lookup for C_dldr, based on Alpha and Beta [deg], [deg]"""
        alpha_table = self.get_alpha_table()
        beta_table = self.get_beta2_table()
        table = np.array([
            [0.0050, 0.017, 0.014, 0.01, -0.0050, 0.0090, 0.019, 0.0050, 0.0, -0.0050, -0.011, 0.0080],
            [0.0070, 0.016, 0.014, 0.014, 0.013, 0.0090, 0.012, 0.0050, 0.0, 0.0040, 0.0090, 0.0070],
            [0.013, 0.013, 0.011, 0.012, 0.011, 0.0090, 0.0080, 0.0050, 0.0, 0.0050, 0.0030, 0.0050],
            [0.018, 0.015, 0.015, 0.014, 0.014, 0.014, 0.014, 0.015, 0.013, 0.011, 0.0060, 0.0010],
            [0.015, 0.014, 0.013, 0.013, 0.012, 0.011, 0.011, 0.01, 0.0080, 0.0080, 0.0070, 0.0030],
            [0.021, 0.011, 0.01, 0.011, 0.01, 0.0090, 0.0080, 0.01, 0.0060, 0.0050, 0.0, 0.0010],
            [0.023, 0.01, 0.011, 0.011, 0.011, 0.01, 0.0080, 0.01, 0.0060, 0.014, 0.02, 0.0]
        ])
        dldr = bilinear_interp(alpha, beta, alpha_table, beta_table, table)
        return dldr
    
    def dnda_lookup(self, alpha, beta):
        """Lookup for C_dldr, based on Alpha and Beta [deg], [deg]"""
        alpha_table = self.get_alpha_table()
        beta_table = self.get_beta2_table()
        table = np.array([
            [0.0010, -0.027, -0.017, -0.013, -0.012, -0.016, 0.0010, 0.017, 0.011, 0.017, 0.0080, 0.016],
            [0.0020, -0.014, -0.016, -0.016, -0.014, -0.019, -0.021, 0.0020, 0.012, 0.016, 0.015, 0.011],
            [-0.0060, -0.0080, -0.0060, -0.0060, -0.0050, -0.0080, -0.0050, 0.0070, 0.0040, 0.0070, 0.0060, 0.0060],
            [-0.011, -0.011, -0.01, -0.0090, -0.0080, -0.0060, 0.0, 0.0040, 0.0070, 0.01, 0.0040, 0.01],
            [-0.015, -0.015, -0.014, -0.012, -0.011, -0.0080, -0.0020, 0.0020, 0.0060, 0.012, 0.011, 0.011],
            [-0.024, -0.01, -0.0040, -0.0020, -0.0010, 0.0030, 0.014, 0.0060, -0.0010, 0.0040, 0.0040, 0.0060],
            [-0.022, 0.0020, -0.0030, -0.0050, -0.0030, -0.0010, -0.0090, -0.0090, -0.0010, 0.0030, -0.0020, 0.0010]
        ])
        dnda = bilinear_interp(alpha, beta, alpha_table, beta_table, table)
        return dnda
    
    def dndr_lookup(self, alpha, beta):
        """Lookup for C_dldr, based on Alpha and Beta [deg], [deg]"""
        alpha_table = self.get_alpha_table()
        beta_table = self.get_beta2_table()
        table = np.array([
            [-0.018, -0.052, -0.052, -0.052, -0.054, -0.049, -0.059, -0.051, -0.03, -0.037, -0.026, -0.013],
            [-0.028, -0.051, -0.043, -0.046, -0.045, -0.049, -0.057, -0.052, -0.03, -0.033, -0.03, -0.0080],
            [-0.037, -0.041, -0.038, -0.04, -0.04, -0.038, -0.037, -0.03, -0.027, -0.024, -0.019, -0.013],
            [-0.048, -0.045, -0.045, -0.045, -0.044, -0.045, -0.047, -0.048, -0.049, -0.045, -0.033, -0.016],
            [-0.043, -0.044, -0.041, -0.041, -0.04, -0.038, -0.034, -0.035, -0.035, -0.029, -0.022, -0.0090],
            [-0.052, -0.034, -0.036, -0.036, -0.035, -0.028, -0.024, -0.023, -0.02, -0.016, -0.01, -0.014],
            [-0.062, -0.034, -0.027, -0.028, -0.027, -0.027, -0.023, -0.023, -0.019, -0.0090, -0.025, -0.01]
        ])
        dndr = bilinear_interp(alpha, beta, alpha_table, beta_table, table)
        return dndr