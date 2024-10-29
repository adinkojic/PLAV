import numpy as np
import scipy
import quaternion
import matplotlib.pyplot as plt
import scipy.integrate
'''
Implenting physics simulation by using an RungeKutta integrator at a fixed timestep 

'''
def get_forces(state):
    #TODO: implement gravity model
    g = 9.81
    gravity = np.array([0, 0, -g])
    orientation = np.quaternion(state[6], state[7], state[8], state[9])
    grotatedq = quaternion.rotate_vectors(orientation, gravity)
    grotatedf= quaternion.as_float_array(grotatedq)

    result = np.array([grotatedf[0][0], grotatedf[1][0], grotatedf[2][0]])
    return result




# x_dot = f(x,u)
# this is the f(x,u)
# returns [x,y,z (NED) , vx, vy, vz, q1 q2 q3 q4, p, q, r]
# TODO: Change to numpyarray
def x_dot(t, y):
    

    #TODO: get real glider values
    mass = 1
    interia = np.identity(3)

    x_dot = np.zeros(13)

    #xdot ydot zdot are vx, vy, vz
    #TODO: convert this from inertial velocity to body velocity
    v_body = np.array([ y[3], y[4], y[5] ])
    orientation = np.quaternion(y[6], y[7], y[8], y[9])
    v_inertialq = quaternion.rotate_vectors(orientation, v_body)
    omega = np.array([ y[10], y[11], y[12]])

    v_inertialf = quaternion.as_float_array(v_inertialq)
    x_dot[0] = v_inertialf[0][0]
    x_dot[1] = v_inertialf[1][0]
    x_dot[2] = v_inertialf[2][0]

    #solving for acceleration, which is velocity_dot
    force_body = get_forces(state = y)
    

    v_b_dot = force_body/mass + np.cross(omega, v_body)
    x_dot[3] = v_b_dot[0]
    x_dot[4] = v_b_dot[1]
    x_dot[5] = v_b_dot[2]

    # integrating roll rates to quaternion
    #  r*q2 -q*q3 +p*q4      *0.5 everything
    # -r*q1 +p*q3 +q*q4
    #  q*q1 -p*q2 +r*q4
    # -p*q1 -q*q2 -r*q3
    q1dot= 0.5*( y[12]*y[7] - y[11]*y[8] + y[10]*y[9] )
    q2dot= 0.5*(-y[12]*y[6] + y[10]*y[8] + y[11]*y[9] )
    q3dot= 0.5*( y[11]*y[6] - y[10]*y[7] + y[12]*y[9] )
    q4dot= 0.5*(-y[10]*y[6] - y[11]*y[7] - y[12]*y[8] )
    x_dot[6] = q1dot
    x_dot[7] = q2dot
    x_dot[9] = q3dot
    x_dot[10]= q4dot

    #to solve for angular velocity change
    #w_B = II_B^-1( M_B - w_Bconj * II_B * w_B )
    #moments_body = get_moments(t)  TODO: implent
    moments_body = np.zeros(3)
    omega_dot = np.linalg.inv(interia) * (moments_body - np.cross(omega, interia*omega))
    #above is probbably wrong due to issues dimensions and stuff, likely needs a transpose
    x_dot[10] = omega_dot[0][0]
    x_dot[11] = omega_dot[1][0]
    x_dot[12] = omega_dot[2][0]

    return x_dot

int_pos = np.array([0, 0, 15])
int_vel = np.array([0, 0, 0])
int_ori = np.array([1, 0, 0, 0])
int_rte = np.array([0, 0, 0])

y0 = np.append(int_pos, np.append(int_vel, np.append(int_ori, int_rte) ))


results = scipy.integrate.solve_ivp(fun = x_dot, t_span = [0, 1], y0=y0, max_step = 0.01)

plt.plot(results.t, results.y[2])
plt.show()