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
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from Controller import PIDController, quar_axis_error, thrust_tilt
from Helper import Logger, normalize, rot_matrix3d, quat2rotm
from transforms3d import euler

euler = euler.EulerFuncs('rxyz')
pi = np.pi
PLOT = True
PLOT_TRAJ = True
#log_variables = ['x_NED', 'e_xyz', 'angle_error', 'x_dot_NED', 'omega', ' u', 'PWM', 'x_ddot_NED', 'e_xyz_sp_NED', 'rate_sp']
log_variables = ['X', 'eulAng', 'Xdot', 'omega', 't', 'u', 'PWM','velDot', 'eulAngSP', 'rateSP']

logger = Logger(log_variables)
# Quad physical parameters
armLength = 95.E-3 # mm
lp = armLength * sin(pi / 4.)

g = 9.81
tau = 35.E-3
m = 1.157

Ixx = 0.014
Iyy = 0.014
Izz = 2. * 0.014
I = np.array([[Ixx, 0, 0],
              [0, Iyy, 0],
              [0, 0, Izz]])

Cl = 7.4e-05 # Thrust to RPS ^ 2

Cd = 6.5e-07
Fd = 0.5694 # drag force coeff of drone assuming it moves at 2 m / s  at 0 and angle of 0.1 %

prop2f_trqMatrix = np.array([[Cl, Cl, Cl, Cl],
                             [Cl * lp, -Cl * lp, -Cl * lp, Cl * lp],
                             [Cl * lp, Cl * lp, -Cl * lp, -Cl * lp],
                             [-Cd, Cd, -Cd, Cd]])

# Initialization
X = np.array([0., 0., 0.]).reshape(3,1)
eulAng = np.array([0., 0., 0.]).reshape(3,1)
Xdot = np.array([0., 0., 0.]).reshape(3,1) # zeros(3, 1);
Q = np.array([1., 0., 0., 0.])
omega = np.array([0., 0., 0.]).reshape(3,1)
state =np.vstack([X, eulAng, Xdot, omega])
# hovering condition
wHover = np.array([1, 1, 1, 1]).reshape(4,1) * np.sqrt(m * g / 4. / Cl)

u = np.array([0, 0, 0, 0]).reshape(4,1)
PWM_hover = wHover / 4.
w = wHover
PWM = PWM_hover


# Initial Setpoint
eulAngSP = np.array([0., -0.2, 0.]).reshape(3,1)
rateSP = np.array([0., 0., 0.]).reshape(3,1)
vzSP = 0.

# Sim time and Sampling
f = 1000.
dt = 1. / f

tstart = 0.
tend = 6.
time = np.linspace(tstart,tend,(int)(tend*f))

n = len(time)
i = 0
F_b = np.zeros([3,1])
Trq_b = np.zeros([3,1])
# Controller settings
Kp_p = 19.
Kp_q = 19.
Kp_r = 19.
W_SAT = 35.

Kp_roll = 15.
Kp_pitch = 15.
Kp_yaw = 15.
PR_SAT = pi

Kp_vz = 40.
Kp_vzSAT = 12.

roll_controller = PIDController(dt, Kp_roll, sat=PR_SAT, name='Roll')
pitch_controller = PIDController(dt, Kp_pitch, sat = PR_SAT, name='Pitch')
yaw_controller = PIDController(dt, Kp_yaw, name='Roll')

alt_controller = PIDController(dt, Kp_vzSAT, sat = Kp_vzSAT, name='Alt')

p_controller = PIDController(dt, Kp_p, name='p')
q_controller = PIDController(dt, Kp_q, name='q')
r_controller = PIDController(dt, Kp_r, name='r')

u2motor = np.array( [[1.,  1.,  1., -1.],
                     [1.,  -1., 1.,  1.],
                     [1., -1., -1., -1.],
                     [1.,  1.,  -1.,  1.]])

q_state = euler.euler2quat(eulAng[0,0], eulAng[1,0], eulAng[2,0])
for t in time:


    if(t > 3):
        eulAngSP = np.array([-0.1, 0.1, 0]).reshape(3,1)

    X = np.array(state[0:3])
    eulAng = np.array(state[3:6])
    vel = np.array(state[6:9])
    omega = np.array(state[9:12])
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

    ## controller

    # Calculating quartenion error
    q_sp = euler.euler2quat(eulAngSP[0,0], eulAngSP[1,0], eulAngSP[2,0]) # XYZ
    axis_error = quar_axis_error(q_sp,q_state)
    # Cascade PID

    rateSP[0,0] = roll_controller.update(axis_error[0], 0.)
    rateSP[1,0] = pitch_controller.update(axis_error[1], 0.)
    rateSP[2,0] = yaw_controller.update(axis_error[2], 0.)

    u[0] = thrust_tilt(eulAng, PWM_hover[0]) -alt_controller.update(vzSP, vz)
    u[1] = p_controller.update(rateSP[0,0], p)
    u[2] = q_controller.update(rateSP[1,0], q)
    u[3] = r_controller.update(rateSP[2,0], r)
    # ** *TEST VAR ** * %
    # testvar = axis_error; % pid_con(vzSP, vz, Kp_vz, 20);
    ## ESC ##
    PWM = np.dot(u2motor,u)
    # ESC Saturation

    for val in range(0, 4):
        PWM[val] = min(PWM[val], 100)
        PWM[val] = max(PWM[val], 10)

    ## Motor dynamics ##
    wCmd = PWM * 4 # mapping ESC to rps experimentally
    # w = wCmd;
    dw = (wCmd - w) / tau
    w = w + dw * dt

    f_trq = np.dot(prop2f_trqMatrix , np.power(w,2))

    ## Euler Newton equations - quad.dynamics ##
    # Quart formulation
    # dcm transforming body coord to ground coord
    dcm_body2frame =  quat2rotm( q_state )

    s = q_state[0]
    v = np.array([q_state[1], q_state[2], q_state[3]])

    sdot = -0.5 * (np.dot(v, omega))
    vdot = 0.5 * (s * omega.reshape(3) + np.cross(omega.reshape(3), v))
    Qdot = np.append(sdot, vdot)

    Xdot = vel
    eulAng_dot = np.dot(dcm_body2frame, omega)
    fb_temp = F_b#  np.reshape(F_b[:, i], (3, 1))
    velDot = np.array([[0.], [0.], [g]]) + np.dot(dcm_body2frame,np.array([[0.], [0.], -f_trq[0]]) ) / m - Fd * vel +  fb_temp/ m
    Iomega = np.dot(I, omega)
    invI =  LA.inv(I) # <<< ---

    x_product = np.transpose(np.cross(np.transpose(omega), np.transpose(Iomega)))
    trqb_temp =Trq_b # np.reshape(Trq_b[:, i],(3,1))
    omegaDot = np.dot(invI,(-1.*x_product)) + np.dot(invI, f_trq[1:4] )  + np.dot(invI,trqb_temp)

    stateDot =np.vstack([Xdot, eulAng_dot, velDot, omegaDot])

    # Update
    state = state + stateDot * dt
    q_next = normalize(q_state + Qdot * dt)
    q_state = q_next

    # Accel Sensor
    accel = velDot
    ## Logging sim data
    var_list = [X , eulAng, vel, omega, t, u, PWM, velDot, eulAngSP, rateSP]
    logger.update_log(var_list)

# Extracting Sim data
logger.postprocess()
X = logger.log['X']
eulAng = logger.log['eulAng']
vel = logger.log['Xdot']
angvel = logger.log['omega']
u = logger.log['u']
PWM = logger.log['PWM']
velDot = logger.log['velDot']
eulAngSP = logger.log['eulAngSP']
rateSP = logger.log['rateSP']

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
    wing1o = np.array([0, 0, 0])
    wing1edge =  np.array([[.5],[-.5],[0.]])
    wing2o = np.array([0, 0, 0])
    wing2edge = np.array([[.5],[.5],[0.]])
    wing3o = np.array([0, 0, 0])
    wing3edge = np.array([[-.5],[.5],[0.]])
    wing4o = np.array([0, 0, 0])
    wing4edge = np.array([[-.5],[-.5],[0.]])

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

    data = X

    data = np.array(data)


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