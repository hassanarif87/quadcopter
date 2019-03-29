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
import matplotlib as plt
pi = np.pi


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
    dt = 1 / 1000
    err = sp - fb
    pgain = Kp * (err)

    output = pgain

    dgain = Kp * (err / dt)
    output = output + dgain


    if (sat > 0):
        # set Saturation
        output = min(output, sat)
        output = max(output, -sat)
    return output

def quar_axis_error(eulAngSP, eulAng):
    # Compute the error in quaternions from the setpoints and robot state in the body frame aligned with x, y, z axis

    # Euler to quart

    setpoints_q = angle2quat(eulAngSP[1], eulAngSP[2], eulAngSP[3])
    state_q = angle2quat(eulAng[1], eulAng[2], eulAng[3])

    a1 = setpoints_q[1]
    b1 = setpoints_q[2]
    c1 = setpoints_q[3]
    d1 = setpoints_q[4]
    a2 = state_q[1]
    b2 = state_q[2] # Conjugate  minus
    c2 = state_q[3] # conjugate   minus
    d2 = state_q[4] # Conjugate  minus
    state_quat_conjugate = np.matrix([a2, -b2, -c2, -d2])

    # Quaternion multiplication q_set * (q_state)'
    quaternion_error_W = np.zeros([4,1])
    quaternion_error_W[1] = a1 * (a2) - b1 * (-b2) - c1 * (-c2) - d1 * (-d2)
    quaternion_error_W[2] = a1 * (-b2) + b1 * a2 + c1 * (-d2) - d1 * (-c2)
    quaternion_error_W[3] = a1 * (-c2) - b1 * (-d2) + c1 * a2 + d1 * (-b2)
    quaternion_error_W[4] = a1 * (-d2) + b1 * (-c2) - c1 * (-b2) + d1 * a2

    # Translate the error into the body frame
    if (quaternion_error_W[1] < 0):
        quaternion_error_W[2: 4] = -1 * quaternion_error_W[2: 4]
    Rmatrixq = quat2rotm(state_q)
    axis_error = np.transpose(Rmatrixq) * np.transpose(quaternion_error_W[2:4])

    return axis_error

def T(eulAng ):
#  Transformes omega in body frame to NED body centered frame

    phi = eulAng[1]
    theta = eulAng[2]
    psi = eulAng[3]
    t = np.matrix([[1, sin(phi)* tan(theta), cos(phi)*tan(theta)]
         [0, cos(phi), -sin(phi)]
         [0, sin(phi)/cos(theta), cos(phi)/cos(theta)]])
    return t

def Tinv(eulAng ):
#  Transformes omega in body frame to NED body centered frame

    phi = eulAng[1]
    theta = eulAng[2]
    psi = eulAng[3]
    t = np.matrix([[1, sin(phi)* tan(theta), cos(phi)*tan(theta)],
         [0, cos(phi), -sin(phi)],
         [0, sin(phi)/cos(theta), cos(phi)/cos(theta)]])
    return LA.inv(t)

def quat2rotm(q):
    s = 1
    rmatrix =np.matrix([[1-2*s*(q[2]*q[2]+q[3]*q[3]), 2*s*(q[1]*q[2]-q[3]*q[0]), 2*s*(q[1]*q[3]+q[2]*q[0])],
                        [2*s*(q[1]*q[2]+q[3]*q[0]), 1-2*s*(q[1]*q[1]+q[3]*q[3]), 2*s*(q[2]*q[3]-q[1]*q[0])],
                        [2*s*(q[1]*q[3]-q[2]*q[0]), 2*s*(q[2]*q[3]+q[1]*q[0]), 1-2*s*(q[1]*q[1]+q[2]*q[2])]])
    return rmatrix

def  rot_matrix3d(eulAng):
    #  Eular angle transformation using Z Y X convention
    phi = eulAng[1]
    theta = eulAng[2]
    psi = eulAng[3]

    Rx = [ [1, 0, 0], [0, cos(phi), -sin(phi)], [0,  sin(phi), cos(phi)]] #Roll
    Ry = [ [cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0 , cos(theta)]] #Pitch
    Rz = [ [cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1] ] #Yaw

    return Rz*Ry*Rx
def thrust_tilt(eulAng,PWM_hover):

    phi = eulAng[1] # Roll
    theta = eulAng[2] # Pitch
    #psi = eulAng[3]
    scaling = 1/(abs(np.sqrt(cos(phi)*cos(theta))))
    scaling = min (scaling, 1.3)
    return PWM_hover*scaling

# Quad physical parameters
armLength = 95e-3 # mm
lp = armLength * np.sin(pi / 4)

g = 9.81
tau = 35E-3
m = 1.157

Ixx = 0.014
Iyy = 0.014
Izz = 2 * 0.014
I = np.matrix([[Ixx, 0, 0],
              [0, Iyy, 0],
              [0, 0, Izz]])

Cl = 7.4e-05 # Thrust to RPS ^ 2

Cd = 6.5e-07
# Fd = 0.5694; # drag force coeff of drone assuming it moves at 2 m / s  at 0 and angle of 0.1 %
#Fd = 0.5 * pi * (cageR) ^ 2 * 1.225 * 0.5;
prop2f_trqMatrix = np.array([[Cl, Cl, Cl, Cl],
                             [Cl * lp, -Cl * lp, -Cl * lp, Cl * lp],
                             [Cl * lp, Cl * lp, -Cl * lp, -Cl * lp],
                             [-Cd, Cd, -Cd, Cd]])

# Initialization
X = np.zeros([3, 1])
eulAng = np.zeros([3, 1])
Xdot = np.matrix('1.5; 0; 0') # zeros(3, 1);
Q = np.matrix('1; 0; 0; 0')
omega = np.zeros([3, 1])
state = np.transpose(np.array([[X, eulAng, Xdot, omega]]))
# hovering condition
wHover = np.matrix('1; 1; 1; 1')* np.sqrt(m * g / 4 / Cl)
u = np.matrix('0; 0; 0; 0')
PWM_hover = wHover / 4
w = wHover
PWM = PWM_hover
log = []

# Initial Setpoint
eulAngSP = np.array('0; -0.25; 0')
rateSP = np.array('3; 1')
vzSP = 0

# Sim time and Sampling
f = 1000
dt = 1 / f

tstart = 0
tend = 6
time = np.linspace(0,tend,(int)(tend*f))

n = len(time)
i = 0

# Controller settings
Kp_p = 19
Kp_q = 19
Kp_r = 19
W_SAT = 35

Kp_roll = 15
Kp_pitch = 15
PR_SAT = pi

Kp_vz = 40
Kp_vzSAT = 12
#u2motor = LA.inv([1, 1, 1, 1;
#1, -1, -1, 1;
#1, 1, -1, -1;
#-1, 1, -1, 1]) / 0.25;
u2motor = np.array( ' 1  1  1 -1;'
                     ' 1  1 -1  1;'
                     ' 1 -1 -1 -1;'
                     ' 1 -1  1  1')

#for t in time:
for t in range(0,1):
# loop
    i = i + 1
    # no small angle assumptions, nonlinear model
    # TODO, include motor inertias effects

    eulAng = state[4:6]
    vel = state[7:9]
    omega = state[10:12]



    # Observer
    C = np.diag([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

    y = C * state

    # vertcal velocity from barometer and rate in bodyframe
    # Baro
    vz = state[9]

    # Gyro
    p = state[10] # y(10)
    q = state[11] # y(11)
    r = state[12] # y(12)

    # Euler angles
    # eulAng = np.zeros([3, 1])
    ## controller

    # Calculating quartenion error

    axis_error = quar_axis_error(eulAngSP, eulAng)

    # PID
    rateSP[1] = pid_con(axis_error[3], 0, Kp_roll, PR_SAT, 0)
    rateSP[2] = pid_con(axis_error[2], 0, Kp_pitch, PR_SAT, 0)
    rateSP[3] = 0
    u[1] = thrust_tilt(eulAng, PWM_hover[1]) - pid_con(vzSP, vz, Kp_vz, Kp_vzSAT, 0)
    u[2] = pid_con(rateSP[1], p, Kp_p, W_SAT, 0)
    u[3] = pid_con(rateSP[2], q, Kp_q, W_SAT, 0)
    u[4] = pid_con(rateSP[3], r, Kp_r, 0, 0)
    # ** *TEST VAR ** * %
    # testvar = axis_error; % pid_con(vzSP, vz, Kp_vz, 20);

    PWM = u2motor * np.transpose(u)


# ESC Saturation

    for val in range(0, 4):
        PWM[val] = min(PWM[val], 100)
        PWM[val] = max(PWM[val], 10)

    # Motor dynamics
    wCmd = PWM * 4 # mapping ESC to rps experimentally
    # w = wCmd;
    dw = (wCmd - w) / tau
    w = w + dw * dt

    f_trq = prop2f_trqMatrix * w*w

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
    eulAng_dot = trans * omega
'''
    velDot = np.matrix([[0.], [0.], [g]]) + Rmatrix3(eulAng)*[0,0,-f_trq(1)]' / m - Fd * vel. ^ 2 +  Fb[:, i]./ m;
    Iomega = I * omega;
    invI = 1.\LA.inv(I);
    omegaDot = invI * (-cross(omega, Iomega)) + invI * [f_trq(2), f_trq(3), f_trq(4)]'+ invI * Trqb[:, i];



    stateDot = [Xdot; eulAng_dot; velDot; omegaDot];

    # Update
    state = state + stateDot. * dt;

    # Accel Sensor
    accel = velDot;

    log = [log, [state; t; velDot; u'; w; testvar]];


X = log(1:3,:);
eulAng = log(4:6,:);
vel = log(7:9,:);
angvel = log(10:12,:);
velDot = log(14:16,:);
u = log(17:20,:);
w = log(21:24,:);
testVar = log(25:28,:);
maxangle = max(sqrt(eulAng(1,:).^ 2 + eulAng(2,:).^ 2 + eulAng(3,:).^ 2))
maxaccel = max(sqrt(velDot(1,:).^ 2 + velDot(2,:).^ 2 + velDot(3,:).^ 2))
maxvel = max(sqrt(vel(1,:).^ 2 + vel(2,:).^ 2 + vel(3,:).^ 2))
figure
plot(time, X(1,:))
hold
on
plot(time, X(2,:))
plot(time, X(3,:))
legend('X', 'Y', 'Z')
title('Pos')

% %
% figure
subplot(3, 1, 1)
plot(time, velDot(1,:))
hold
on
plot(time, velDot(2,:))
plot(time, velDot(3,:))
legend('X', 'Y', 'Z')
title('Accel')
hold
off

subplot(3, 1, 2)
plot(time, vel(1,:))
hold
on
plot(time, vel(2,:))
plot(time, vel(3,:))
title('Vel')
legend('x', 'y', 'z');
hold
off

subplot(3, 1, 3)

plot(time, eulAng(1,:))
hold
on
plot(time, eulAng(2,:))
plot(time, eulAng(3,:))
title('Euler Angles')
legend('roll', 'pitch', 'yaw');
hold
off
% %

figure
plot(time, u(1,:))
hold
on
plot(time, u(2,:))
plot(time, u(3,:))
plot(time, u(4,:))
legend('1', '2', '3', '4')
title('Cmd')

figure
plot(time, w(1,:))
hold
on
plot(time, w(2,:))
plot(time, w(3,:))
plot(time, w(4,:))
legend('1', '2', '3', '4')
title('motor speed ')

figure
plot(time, vel(1,:))
hold
on
plot(time, vel(2,:))
plot(time, vel(3,:))
title('Vel')
legend('x', 'y', 'z');

figure
plot(time, eulAng(1,:))
hold
on
plot(time, eulAng(2,:))
plot(time, eulAng(3,:))
title('Euler Angles')
legend('roll', 'pitch', 'yaw');
% axis(-2 * pi
2 * pi )
% figure
  % plot(time, testVar)
  % title('test')

  % %
  figure
plot(time, testVar(1,:))
hold
on
plot(time, testVar(2,:))
plot(time, testVar(3,:))
plot(time, testVar(4,:))
legend('1', '2', '3', '4')
title('PWM')

% %
visualization3(log, 0)
'''