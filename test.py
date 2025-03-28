import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numba import jit
from numba import float64, int64    # import the types
from ivp_logger import IVPLogger
from numba.experimental import jitclass


# Define the RHS of the ODE, including position, velocity, and forces

@jit
def rhs(t, y):
    # Unpack position (x, y) and velocity (vx, vy)
    x, y, vx, vy = y

    g = 9.81
    #y force gravity and drag
    force_y = -g  - 0.1 * vy 
    
    dxdt = vx
    dydt = vy
    dvxdt = 0.0  # no horizontal forces
    dvydt = force_y

    return np.array([dxdt, dydt, dvxdt, dvydt])


spec = [
    ('state', float64[:]),
    ('t_span', float64[:]),
    ('time', float64),
    ('data', float64[:,:]),
    ('data_columns', int64),
    ('valid_data_size', int64)
]

@jitclass(spec)
class Simulator(object):
    """A sim object is required to store all the required data nicely."""
    def __init__(self, init_state, time_span):
        self.state = init_state
        self.t_span = time_span
        self.time = time_span[0]
        self.logger = IVPLogger(5)

        self.logger.append_data(np.append(time_span[0], [init_state]))

    def advance_timestep(self, t_step = 0.01):
        """advance timestep function, updates timestep and saves values"""

        t_span=np.array([self.time, self.time + t_step])
        new_state = solve_ivp(fun = rhs, t_span = t_span, y0=self.state)
        self.state = new_state.y[:,-1]
        self.time = new_state.t[-1]


        print(new_state.y[-1])

        time = np.array([new_state.t[-1]])
        forces = np.array([0.0])
 
        print(new_state.y[:,-1])
        data_to_append = np.append(time, [new_state.y[:,-1]])

        self.logger.append_data(data_to_append)

    def run_sim(self):
        """runs the sim, could also include control inputs"""
        while self.time < self.t_span[1]:
            self.advance_timestep()

    def return_results(self):
        """logger"""
        return self.logger.return_data()



# Initial conditions: x0, y0, vx0, vy0
y0 = np.array([0.0, 0.0, 0.0, 10.0])  # Initial position (x, y) and velocity (vx, vy)

t_span = np.array([0.0, 2.0]) 

sol = solve_ivp(fun = rhs, t_span=t_span, y0=y0, max_step=0.1)

sim_object = Simulator(y0, t_span)
sim_object.run_sim()

data = sim_object.return_results()
y2 = data[:,2] 
t2 = data[:,0]

# Extract positions and velocities from the solution
x, y, vx, vy = sol.y


# Plot the trajectory of the ball
plt.plot(sol.t, y)
plt.plot(t2, y2)
plt.xlabel('Position x (m)')
plt.ylabel('Position y (m)')
plt.title('Projectile motion of the ball')
plt.grid(True)
plt.show()
