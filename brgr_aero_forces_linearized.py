import math
import numpy as np
from numba import jit
import quaternion_math as quat


'''BRGR Glide Vehicle Aero Sim
TODO: simulation for high altitude'''

#gives in body frame
def from_alpha_beta(airspeed, alpha, beta):
    x = airspeed * math.cos(alpha * math.pi/180) * math.cos(beta * math.pi/180)
    y = airspeed * math.sin(beta  * math.pi/180)
    z = airspeed * math.sin(-alpha * math.pi/180) * math.cos(beta * math.pi/180)

    return np.array([x,y,z])

@jit #velocity, sans wind
def velocity_to_alpha_beta(velocity_body):
    airspeed = math.sqrt(velocity_body[0]**2 + velocity_body[1]**2 + velocity_body[2]**2)
    temp = math.sqrt(velocity_body[0]**2 + velocity_body[2]**2)
    beta = math.atan2(velocity_body[1], temp)
    alpha = -math.atan2(velocity_body[2], velocity_body[0])

    return airspeed, alpha, beta

#TODO: implement atmosphere model
@jit
def get_atmosphere(altitude):
    density = 1.225
    viscosity = 0.00001789
    return density, viscosity

@jit
def get_Re(airspeed, density, viscosity):
    length = 0.1778 #characteristic length
    Re = airspeed * length * density/viscosity
    return Re

#TODO: implement based on better data and with Re
@jit
def get_coeff(alpha, beta, Re, p, q ,r):
    aa = np.array([alpha]).clip(-15*math.pi/180, 15*math.pi/180)
    alpha = aa[0]

    C_L0 = 0.384
    C_La = 4.446
    C_D0 = 0.0088
    epsilon = 0.055
    C_M0 = 0.055
    C_Ma = 0.1432
    C_Mq = -0.01 #not sure about this value

    C_L = C_L0 + C_La * alpha
    C_D = C_D0 + epsilon * C_L**2
    C_M = C_M0 + C_Ma * alpha #+ C_Mq * q

    C_Y = 0 #yaw
    C_R = 0 #roll
    C_Z = 0 #side force

    return C_L,C_D,C_M, C_Z, C_Y, C_R

@jit
def get_wind_to_stability_axis(alpha, beta):
    beta_rot  = quat.from_angle_axis(beta, np.array([0, 0, 1]))
    alpha_rot = quat.from_angle_axis(alpha, np.array([0, -1, 0]))

    return quat.mulitply(alpha_rot, beta_rot)

@jit
def gridfin_control_mixing(pitch_command, roll_command, yaw_command):
    """basic gridfin control mixing matrix
    for testing/manual control"""
    top_gf_cmd  = yaw_command - roll_command
    star_gf_cmd = -pitch_command + 0.5*yaw_command - roll_command
    left_gf_cmd =  pitch_command - 0.5*yaw_command - roll_command

    return np.array([top_gf_cmd, star_gf_cmd, left_gf_cmd])

@jit
def get_grid_fin_forces(command_vector, velocity, p, q, r, Re):
    """get grid fin forces from airspeed
    input 1x3 command_vector for angle control, right hand rule, radians
    start from top use right hand rule
    moment based on lever arm, gridfins have low moments
    
    return forces, moment
    
    #TODO: implement 6DoF, this is only 3"""
    
    omega = np.array(p,q,r)
    #pos q brings nose down
    arm = 0.42 #meters
    radial = 0.095

    #i really don't feel like typing out math.cos(30)
    cos30 = 0.8660254037844386

    top_pos  = np.array([-arm, radial, 0])
    port_pos = np.array([-arm, cos30*radial, -0.5*radial])
    star_pos = np.array([-arm, -cos30*radial, -0.5*radial])

    local_velocity_top  = velocity + np.cross(omega,top_pos)
    local_velocity_port = velocity + np.cross(omega,port_pos)
    local_velocity_star = velocity + np.cross(omega,star_pos)


    top_radial  = np.array([0, 0, 1])
    port_radial = np.array([0, -cos30, 0.5])
    star_radial = np.array([0, cos30, 0.5])

    
    top_norm  = np.array([math.cos(command_vector[0]), math.sin(command_vector[0]), 0])
    port_norm = np.array([math.cos(command_vector[1]), -0.5*math.sin(command_vector[1]), -cos30*math.sin(command_vector[1])])
    star_norm = np.array([math.cos(command_vector[2]), -0.5*math.sin(command_vector[2]), cos30*math.sin(command_vector[2])])

    #top_tangential  = np.array([math.sin(command_vector[0]), -math.cos(command_vector[0]), 0])
    #port_tangential = np.array([math.sin(command_vector[1]), 0.5*math.cos(command_vector[1]), cos30*math.sin(command_vector[1])])
    #star_tangential = np.array([math.cos(command_vector[2]), 0.5*math.cos(command_vector[2]), -cos30*math.sin(command_vector[2])])

    #change aircraft velocity basis to grid fin basis
    top_tangential_transform  = quat.from_angle_axis(command_vector[0], top_radial)
    port_tangential_transform = quat.from_angle_axis(command_vector[1], port_radial)
    star_tangential_transform = quat.from_angle_axis(command_vector[2], star_radial)

    #rotate it so it aligns nicely
    top_radial_transform  = quat.from_angle_axis(-math.pi/2, np.array([1,0,0]))
    port_radial_transform = quat.from_angle_axis(math.pi/6, np.array([1,0,0]))
    star_radial_transform = quat.from_angle_axis(5*math.pi/6, np.array([1,0,0]))

    top_total_tansform  = quat.mulitply(top_radial_transform, top_tangential_transform)
    port_total_tansform = quat.mulitply(port_radial_transform, port_tangential_transform)
    star_total_tansform = quat.mulitply(star_radial_transform, star_tangential_transform)

    top_frame_velocity  = quat.rotateFrameQ(top_total_tansform, local_velocity_top)
    port_frame_velocity = quat.rotateFrameQ(port_total_tansform, local_velocity_port)
    star_frame_velocity = quat.rotateFrameQ(star_total_tansform, local_velocity_star)

    top_speed, top_alpha, top_beta = velocity_to_alpha_beta(top_frame_velocity)
    port_speed, port_alpha, port_beta = velocity_to_alpha_beta(port_frame_velocity)
    star_speed, star_alpha, star_beta = velocity_to_alpha_beta(star_frame_velocity)

    #ok now that we have alpha beta we can get actual forces
    gfC_D0 = 0.00384
    gfC_Ta = 0.057
    gfC_Ra = 0.04
    gfEpT  = 0.057
    gfEpR  = 0.025
    Aref   = 0.2274 #this is how I found out I don't have half the control auth I need

    #TODO: implement based on Re
    density = 1.225

    topTan = 0.5 * density * top_speed**2 *Aref * gfC_Ta * top_alpha
    topRad = 0.5 * density * top_speed**2 *Aref * gfC_Ra * top_beta
    topDrag = 0.5 * density * top_speed**2 *Aref * (gfC_D0 +  gfEpT * topTan**2 + gfEpR * topRad**2)

    portTan = 0.5 * density * port_speed**2 *Aref * gfC_Ta * port_alpha
    portRad = 0.5 * density * port_speed**2 *Aref * gfC_Ra * port_beta
    portDrag = 0.5 * density * port_speed**2 *Aref * (gfC_D0 +  gfEpT * portTan**2 + gfEpR * portRad**2)

    starTan = 0.5 * density * star_speed**2 * Aref * gfC_Ta * star_alpha
    starRad = 0.5 * density * star_speed**2 * Aref * gfC_Ra * star_beta
    starDrag = 0.5 * density * star_speed**2 * Aref * (gfC_D0 +  gfEpT * starTan**2 + gfEpR * starRad**2)

    #god what a terrible configuration
    total_drag_force = topDrag + portDrag + starDrag
    total_side_force = topTan + 0.5 * portTan + 
    total_vert_froce = 







#TODO: test and implement grid fins
@jit
def get_aero_forces(state):
    Aref = 0.2274 #m**2
    mac = 0.1778 #m

    velocity = np.array([state[3],state[4],state[5]])
    airspeed, alpha, beta = velocity_to_alpha_beta(velocity)

    altitude = state[2]

    density, viscosity = get_atmosphere(altitude)
    Re = get_Re(airspeed, density, viscosity)

    p, q, r = state[10], state[11], state[12]



    C_L, C_D, C_M, C_Z, C_Y, C_R = get_coeff(alpha, beta, Re, p, q ,r)

    qbar = 0.5 * density *airspeed**2

    body_lift = C_L * qbar * Aref
    body_drag = C_D * qbar * Aref
    body_side = C_Z * qbar * Aref
    body_pitching_moment = C_M * qbar * Aref * mac
    body_yawing_moment   = C_Y * qbar * Aref #Coeff definition might be wrong
    body_rolling_moment  = C_R * qbar * Aref

    wind_to_stab = get_wind_to_stability_axis(alpha,beta)

    body_forces_wind = np.array([-body_drag, body_side, body_lift])
    body_forces_stab = quat.rotateVectorQ(wind_to_stab, body_forces_wind)

    moments = np.array([body_rolling_moment, -body_pitching_moment, body_yawing_moment])

    return body_forces_stab, moments
