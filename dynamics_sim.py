'''
Implenting physics simulation by using an RungeKutta integrator at a fixed timestep 

'''



# x_dot = f(x,u)
# this is the f(x,u)
# returns [x,y,z (NED) , vx, vy, vz, q1 q2 q3 q4, p, q, r]
def x_dot(x, forcesBody, momentsBody, mass, interia):
    x_dot = []
    #xdot ydot zdot are vx, vy, vz
    x_dot.append(x[0])
    x_dot.append(x[1])
    x_dot.append(x[2])

    #solving for acceleration, which is velocity_dot
    #TODO: adapt for body
    xaccel = forceInertial[0]/mass
    yaccel = forceInertial[1]/mass
    zaccel = forceInertial[2]/mass
    x_dot.append(xaccel)
    x_dot.append(yaccel)
    x_dot.append(zaccel)

    # integrating roll rates to quaternion
    #  r*q2 -q*q3 +p*q4      *0.5 everything
    # -r*q1 +p*q3 +q*q4
    #  q*q1 -p*q2 +r*q4
    # -p*q1 -q*q2 -r*q3
    q1dot= 0.5*( momentsBody[2]*x[7] - momentsBody[1]*x[8] + momentsBody[0]*x[9] )
    q2dot= 0.5*(-momentsBody[2]*x[6] + momentsBody[0]*x[8] + momentsBody[1]*x[9] )
    q3dot= 0.5*( momentsBody[1]*x[6] - momentsBody[0]*x[7] + momentsBody[2]*x[9] )
    q4dot= 0.5*(-momentsBody[0]*x[6] - momentsBody[1]*x[7] - momentsBody[2]*x[8] )
    x_dot.append(q1dot)
    x_dot.append(q2dot)
    x_dot.append(q3dot)
    x_dot.append(q4dot)

    #to solve for angular velocity change
    #w_B = II_B^-1( M_B - w_Bconj * II_B * w_B )
    #TODO: implement above
    #p_dot = ()/()

    return x_dot