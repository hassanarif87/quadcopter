#Quad rotor simulator
#
#by Hassan Arif 20180204
#
# Top view
# 1(Cw)2
#   \ /
#    X
#   / \
# 4     3
#
# x
# |
# |
# *----y
import numpy as np
from numpy import linalg as LA
from numpy import sin as sin
from numpy import cos as cos
from numpy import tan as tan
from navpy import angle2quat
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

pi = np.pi
PLOT = False
PLOT_TRAJ = True

def normalize( vector ):
    # normalise vector
    if (LA.norm(vector) == 0 ):
        normalizedVector = vector
    else:
        normalizedVector = vector/LA.norm(vector)
    return normalizedVector

def pid_con(sp, fb, Kp, sat, Kd):
    #TODO: optional argumanets and add PI
# PID
    dt = 1. / 1000.
    err = sp - fb
    pgain = Kp * (err)

    output = pgain

    dgain = Kd * (err / dt)
    output = output + dgain


    if (sat > 0.):
        # set Saturation
        output = min(output, sat)
        output = max(output, -sat)
    return output

def quar_axis_error(eulAngSP, eulAng):
    # Compute the error in quaternions from the setpoints and robot state in the body frame aligned with x, y, z axis

    # Euler to quart
    #print(eulAngSP)
    setpoints_q0, setpoint_qvec = angle2quat(eulAngSP[0,0], eulAngSP[1,0], eulAngSP[2,0])
    state_q0, state_qvec = angle2quat(eulAng[0,0], eulAng[1,0], eulAng[2,0])

    a1 = setpoints_q0
    b1 = setpoint_qvec[0]
    c1 = setpoint_qvec[1]
    d1 = setpoint_qvec[2]
    a2 = state_q0
    b2 = state_qvec[0] # Conjugate  minus
    c2 = state_qvec[1] # conjugate   minus
    d2 = state_qvec[2] # Conjugate  minus
    state_quat_conjugate = np.matrix([a2, -b2, -c2, -d2])

    # Quaternion multiplication q_set * (q_state)'
    quaternion_error_W = np.zeros([4,1])
    quaternion_error_W[0] = a1 * (a2) - b1 * (-b2) - c1 * (-c2) - d1 * (-d2)
    quaternion_error_W[1] = a1 * (-b2) + b1 * a2 + c1 * (-d2) - d1 * (-c2)
    quaternion_error_W[2] = a1 * (-c2) - b1 * (-d2) + c1 * a2 + d1 * (-b2)
    quaternion_error_W[3] = a1 * (-d2) + b1 * (-c2) - c1 * (-b2) + d1 * a2

    # Translate the error into the body frame
    if (quaternion_error_W[0] < 0):
        quaternion_error_W[1: 4] = -1. * quaternion_error_W[1: 4]
    state_q = np.hstack((state_q0, state_qvec))
    Rmatrixq = quat2rotm(state_q)
    axis_error = np.dot( np.transpose(Rmatrixq),(quaternion_error_W[1:4]) )

    return axis_error

def T(eulAng ):
#  Transformes omega in body frame to NED body centered frame
    phi = eulAng[0,0]
    theta = eulAng[1,0]
    psi =  eulAng[2,0]

    t = np.matrix([[1., sin(phi)* tan(theta), cos(phi)*tan(theta)],
         [0., cos(phi), -sin(phi)],
         [0., sin(phi)/cos(theta), cos(phi)/cos(theta)]])
    return t

def Tinv(eulAng ):
#  Transformes omega in body frame to NED body centered frame

    phi = eulAng[0,0]
    theta = eulAng[1,0]
    psi =  eulAng[2,0]
    t = np.matrix([[1., sin(phi)* tan(theta), cos(phi)*tan(theta)],
         [0., cos(phi), -sin(phi)],
         [0., sin(phi)/cos(theta), cos(phi)/cos(theta)]])
    return LA.inv(t)

def quat2rotm(q):
    s = 1.
    rmatrix =np.matrix([[1.-2.*s*(q[2]*q[2]+q[3]*q[3]), 2.*s*(q[1]*q[2]-q[3]*q[0]), 2.*s*(q[1]*q[3]+q[2]*q[0])],
                        [2.*s*(q[1]*q[2]+q[3]*q[0]), 1.-2.*s*(q[1]*q[1]+q[3]*q[3]), 2.*s*(q[2]*q[3]-q[1]*q[0])],
                        [2.*s*(q[1]*q[3]-q[2]*q[0]), 2.*s*(q[2]*q[3]+q[1]*q[0]), 1.-2.*s*(q[1]*q[1]+q[2]*q[2])]])
    return rmatrix

def  rot_matrix3d(eulAng):
    #  Eular angle transformation using Z Y X convention
    phi = eulAng[0,0]
    theta = eulAng[1,0]
    psi =  eulAng[2,0]

    Rx = np.matrix([ [1., 0., 0.], [0., cos(phi), -sin(phi)], [0.,  sin(phi), cos(phi)]]) #Roll
    Ry = np.matrix([ [cos(theta), 0., sin(theta)], [0., 1., 0.], [-sin(theta), 0. , cos(theta)]]) #Pitch
    Rz = np.matrix([ [cos(psi), -sin(psi), 0.], [sin(psi), cos(psi), 0.], [0., 0., 1.] ]) #Yaw

    return LA.multi_dot([Rz,Ry,Rx])

def thrust_tilt(eulAng,PWM_hover):

    phi = eulAng[0,0] # Roll
    theta = eulAng[1,0] # Pitch
    psi =  eulAng[2,0]
    scaling = 1./(abs(np.sqrt(cos(phi)*cos(theta))))
    scaling = min (scaling, 1.3)
    return PWM_hover*scaling

def theta_wraper(theta):
    # keeps theta + / - pi
    if (theta >= (pi)):
        theta = theta - 2 * pi
    elif(theta < -(pi)):
        theta = theta + 2 * pi
    return theta

# Quad physical parameters
armLength = 95.E-3 # mm
lp = armLength * sin(pi / 4.)

g = 9.81
tau = 35.E-3
m = 1.157

Ixx = 0.014
Iyy = 0.014
Izz = 2. * 0.014
I = np.matrix([[Ixx, 0, 0],
              [0, Iyy, 0],
              [0, 0, Izz]])

Cl = 7.4e-05 # Thrust to RPS ^ 2

Cd = 6.5e-07
Fd = 0.5694 # drag force coeff of drone assuming it moves at 2 m / s  at 0 and angle of 0.1 %

prop2f_trqMatrix = np.matrix([[Cl, Cl, Cl, Cl],
                             [Cl * lp, -Cl * lp, -Cl * lp, Cl * lp],
                             [Cl * lp, Cl * lp, -Cl * lp, -Cl * lp],
                             [-Cd, Cd, -Cd, Cd]])

# Initialization
X = np.matrix('0; 0; 0', dtype = float)
eulAng = np.matrix('0; 0; 0', dtype = float)
Xdot = np.matrix('0; 0; 0', dtype = float) # zeros(3, 1);
Q = np.matrix('1; 0; 0; 0', dtype = float)
omega = np.zeros([3, 1])
state =np.vstack([X, eulAng, Xdot, omega])
# hovering condition
wHover = np.matrix('1; 1; 1; 1', dtype = float)* np.sqrt(m * g / 4. / Cl)
u = np.matrix('0; 0; 0; 0', dtype = float)
PWM_hover = wHover / 4.
w = wHover
PWM = PWM_hover

# Initial Setpoint
eulAngSP = np.matrix('0; -0.2; 0', dtype = float)
rateSP = np.matrix('0; 0; 0', dtype = float)
vzSP = 0.

# Sim time and Sampling
f = 1000.
dt = 1. / f

tstart = 0.
tend = 6.
time = np.linspace(tstart,tend,(int)(tend*f))

n = len(time)
i = 0
F_b = np.zeros([3,n])
Trq_b = np.zeros([3,n])
# Controller settings
Kp_p = 19.
Kp_q = 19.
Kp_r = 19.
W_SAT = 35.

Kp_roll = 15.
Kp_pitch = 15.
PR_SAT = pi

Kp_vz = 40.
Kp_vzSAT = 12.
#u2motor = LA.inv([[1, 1, 1, 1],
#                [1, -1, -1, 1],
#                [1, 1, -1, -1],
#                [-1, 1, -1, 1]]) / 0.25
u2motor = np.matrix( ' 1  1  1 -1;'
                     ' 1  -1 1  1;'
                     ' 1 -1 -1 -1;'
                     ' 1  1  -1  1', dtype=float)

log = np.zeros([30,n])

for t in time:
#for t in range(0,2000):
# loop

    # no small angle assumptions, nonlinear model
    # TODO, include motor inertias effects
    if(t > 3):
        eulAngSP = np.matrix('-0.1; 0.1; 0', dtype=float)
    eulAng = state[3:6]
    vel = state[6:9]
    omega = state[9:12]



    # Observer
    C = np.diag([0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.])

    y = np.dot(C,state)

    # vertcal velocity from barometer and rate in bodyframe
    # Baro
    vz = state[8]

    # Gyro
    p = state[9] # y(10)
    q = state[10] # y(11)
    r = state[11] # y(12)

    # Euler angles
    # eulAng = np.zeros([3, 1])
    ## controller

    # Calculating quartenion error
    axis_error = quar_axis_error(eulAngSP,eulAng)

    # PID


    #rateSP[0] = pid_con(eulAngSP[0],eulAng[0], Kp_roll,PR_SAT,0.) # pid_con(axis_error[2], 0., Kp_roll, PR_SAT, 0.)
    #rateSP[1] = pid_con(eulAngSP[1],eulAng[1], Kp_pitch,PR_SAT,0.) #pid_con(axis_error[1], 0., Kp_pitch, PR_SAT, 0.)
    #rateSP[2] = 0.

    rateSP[0] = pid_con(axis_error[2], 0., Kp_roll, PR_SAT, 0.)
    rateSP[1] = pid_con(axis_error[1], 0., Kp_pitch, PR_SAT, 0.)
    rateSP[2] = 0.

    u[0] = thrust_tilt(eulAng, PWM_hover[0]) -pid_con(vzSP, vz, Kp_vz, Kp_vzSAT, 0.)
    u[1] = pid_con(rateSP[0], p, Kp_p, W_SAT, 0.)
    u[2] = pid_con(rateSP[1], q, Kp_q, W_SAT, 0.)
    u[3] = pid_con(rateSP[2], r, Kp_r, 0., 0.)
    # ** *TEST VAR ** * %
    # testvar = axis_error; % pid_con(vzSP, vz, Kp_vz, 20);

    PWM = np.dot(u2motor,u)


# ESC Saturation

    for val in range(0, 4):
        PWM[val] = min(PWM[val], 100)
        PWM[val] = max(PWM[val], 10)
    #print(PWM)

    # Motor dynamics
    wCmd = PWM * 4 # mapping ESC to rps experimentally
    # w = wCmd;
    dw = (wCmd - w) / tau
    w = w + dw * dt

    f_trq = np.dot(prop2f_trqMatrix , np.power(w,2))

    # Euler Newton equations - quad.dynamics

    # Quart formulation
    # Rotation  matrix for transforming body coord to ground coord
    # RmatrixQ = quat2rotm(roro.Q');
    #
    # s = Q(1);
    # v = [Q(1); Q(2);Q(3)];
    # sdot = -0.5 * (dot(omega, v));
    # vdot = 0.5 * (s * omega + cross(omega, v));
    # Qdot = [sdot; vdot]
    #
    trans = T(eulAng)
    Xdot = vel
    eulAng_dot = np.dot(trans, omega)
    fb_temp = np.reshape(F_b[:, i], (3, 1))

    velDot = np.matrix([[0.], [0.], [g]]) + np.dot(rot_matrix3d(eulAng),np.matrix([[0.], [0.], [-f_trq[0]]]) ) / m - Fd * vel +  fb_temp/ m
    Iomega = np.dot(I, omega)
    invI =  LA.inv(I) #<<< ---

    #print("Test")
    x_product = np.transpose(np.cross(np.transpose(omega), np.transpose(Iomega)))
    trqb_temp = np.reshape(Trq_b[:, i],(3,1))
    #print(np.dot(invI, f_trq[1:4] ))
    omegaDot = np.dot(invI,(-1.*x_product)) + np.dot(invI, f_trq[1:4] )  + np.dot(invI,trqb_temp)

    stateDot =np.vstack([Xdot, eulAng_dot, velDot, omegaDot])


    # Update
    state = state + stateDot * dt

    # Accel Sensor
    accel = velDot
    ## Logging sim data
    log_temp =  np.vstack([state, t, u, PWM,velDot, eulAngSP, rateSP])
    #print(log_temp)
    log[:,i] = np.reshape(log_temp,(30))
    i = i + 1


# Extracting Sim data
X = log[0:3,:]
eulAng = log[3:6,:]
vel = log[6:9,:]
angvel = log[9:12,:] # omega
#velDot = log[12:16,:]
u = log[13:17,:]
PWM = log[17:21,:]
velDot = log[21:24,:]
eulAngSP = log[24:27,:]
rateSP = log[27:30,:]

if (PLOT == True):
    plt.figure(1)
    plt.plot(time, X[0,:],'r',label='X',linewidth= 0.5)
    plt.plot(time, X[1,:],'g',label='Y',linewidth= 0.5)
    plt.plot(time, X[2,:],'b',label='Z',linewidth= 0.5)
    plt.legend()
    plt.title('Pos')


    plt.figure(2)
    plt.plot(time, vel[0,:],'r',label='X',linewidth= 0.5)
    plt.plot(time, vel[1,:],'g',label='Y',linewidth= 0.5)
    plt.plot(time, vel[2,:],'b',label='Z',linewidth= 0.5)
    plt.legend()
    plt.ylabel("Velocity(m/s)")
    plt.xlabel("Time(s)")
    plt.title('Vel')

    plt.figure(3)
    plt.plot(time, eulAng[0,:],'r',label='Roll',linewidth= 0.5)
    plt.plot(time, eulAng[1,:],'g',label='Pitch',linewidth= 0.5)
    plt.plot(time, eulAng[2,:],'b',label='Yaw',linewidth= 0.5)
    plt.plot(time, eulAngSP[0,:],'r--',label='Roll SP',linewidth= 0.5)
    plt.plot(time, eulAngSP[1,:],'g--',label='Pitch SP',linewidth= 0.5)
    plt.plot(time, eulAngSP[2,:],'b--',label='Yaw SP',linewidth= 0.5)
    plt.ylabel("Angle(rad)")
    plt.xlabel("Time(s)")
    plt.legend(loc='upper right')
    plt.title('Euler Angles')


    plt.figure(4)
    plt.plot(time, u[0,:],label='Z',linewidth= 0.5)
    plt.plot(time, u[1,:],label='R',linewidth= 0.5)
    plt.plot(time, u[2,:],label='P',linewidth= 0.5)
    plt.plot(time, u[3,:],label='Y',linewidth= 0.5)
    plt.legend()
    plt.title('Cmd')

    plt.figure(5)
    plt.plot(time, PWM[0,:])
    plt.plot(time, PWM[1,:])
    plt.plot(time, PWM[2,:])
    plt.plot(time, PWM[3,:])
    plt.title('PWM')

    plt.figure(6)
    plt.plot(time, velDot[0,:])
    plt.plot(time, velDot[1,:])
    plt.plot(time, velDot[2,:])
    plt.title('Linear Accel')
    plt.figure(7)
    plt.plot(time, rateSP[0,:],label='Roll',linewidth= 0.5)
    plt.plot(time, rateSP[1,:],label='Pitch',linewidth= 0.5)
    plt.plot(time, rateSP[2,:],label='Yaw',linewidth= 0.5)
    plt.legend()
    plt.title('Rate Sp')

if (PLOT_TRAJ == True):
    ## 3d path
    wing1o = np.matrix('0; 0; 0', dtype = float)
    wing1edge =  np.array([[.5],[-.5],[0.]])
    wing2o = np.matrix('0; 0; 0', dtype = float)
    wing2edge = np.array([[.5],[.5],[0.]])  #np.matrix('1; 1; 0', dtype = float)
    wing3o = np.matrix('0; 0; 0', dtype = float)
    wing3edge = np.array([[-.5],[.5],[0.]]) # np.matrix('-1; 1; 0', dtype = float)
    wing4o = np.matrix('0; 0; 0', dtype = float)
    wing4edge = np.array([[-.5],[-.5],[0.]]) # np.matrix('-1; -1; 0', dtype = float)

    def update_traj(num, dat, lines):
        for line in lines:
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data(dat[0:2, :num])
            line.set_3d_properties(dat[2, :num])
        return lines

    def update_wings(num, dat, lines):
        for line in lines:
            # NOTE: there is no .set_data() for 3 dim data...
            x = np.array(dat[0:2,0,num])
            y = np.array(dat[0:2,1,num])
            line.set_data(x, y)
            line.set_3d_properties(dat[0:2,2,num])
        return lines
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)


    # Fifty lines of random 3-D lines
    data = X

    #data = [Gen_RandLine(25, 3) for index in range(1)]
    data = np.array(data)
    #data1  = np.array([data + wing1edge, data + wing3edge])
    #data2  = np.array([data + wing2edge, data + wing4edge])
    #print(data1.shape)

    n_steps = (tend* f)
    plot_frames = (int)(n_steps/50)
    traj_data = np.zeros([3,plot_frames])
    wing_data1 = np.zeros([2,3,plot_frames])
    wing_data2 = np.zeros([2,3,plot_frames])
    i = 0
    for j in range(0,plot_frames):
        traj_data[:,j] = data[:,i]
        trag_date_n = np.reshape(traj_data[:,j],(3,1))
        eul = np.reshape(eulAng[:,i],(3,1))
        wing1 = trag_date_n + np.dot(rot_matrix3d(eul),wing1edge)
        wing2 = trag_date_n + np.dot(rot_matrix3d(eul),wing2edge)
        wing3 = trag_date_n + np.dot(rot_matrix3d(eul),wing3edge)
        wing4 = trag_date_n + np.dot(rot_matrix3d(eul),wing4edge)
        wing_data1[0,:, j] = wing1.reshape(3)
        wing_data1[1,:, j] = wing3.reshape(3)
        wing_data2[0,:, j] = wing2.reshape(3)
        wing_data2[1,:, j] = wing4.reshape(3)
        i = i + 50

    lines = ax.plot(traj_data[0, 0:1], traj_data[1, 0:1], traj_data[2, 0:1], 'b--')
    lines1 = ax.plot(wing_data1[0:2, 0, 0], wing_data1[0:2, 1, 0], wing_data1[0:2, 2, 0], 'r')
    lines2 = ax.plot(wing_data1[0:2, 0, 0], wing_data1[0:2, 1, 0], wing_data1[0:2, 2, 0], 'g')

    # Setting the axes properties
    ax.set_xlim3d([-1.0, 8.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-4.0, 4.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-4.0, 4.0])
    ax.set_zlabel('Z')

    ax.set_title('Trajectory')

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_traj, plot_frames, fargs=(traj_data, lines),
                                       interval=1, blit=False)
    line_ani1 = animation.FuncAnimation(fig, update_wings, plot_frames, fargs=(wing_data1, lines1),
                                       interval=1, blit=False)
    line_ani2 = animation.FuncAnimation(fig, update_wings, plot_frames, fargs=(wing_data2, lines2),
                                       interval=1, blit=False)
#########################

plt.show()