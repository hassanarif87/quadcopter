#Quad rotor simulator
#
#by Hassan Arif
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
import json
import numpy as np
from numpy import linalg as LA
from numpy import sin as sin
from numpy import cos as cos
from numpy import tan as tan
from transforms3d import euler
from copy import deepcopy

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

# Simulation Objects
from Controller import FlightComputer
from SimObjects.MassObj import MassObj
from SimObjects.DynamicObj import DynamicObj, SixDofState, ned_gravity
from SimObjects.Motor import Motor
from SimObjects.AeroForces import AeroForces

from plotter import plot_logs
from Helper import Logger, normalize, rot_matrix3d

euler = euler.EulerFuncs('rxyz')
# Constants
pi = np.pi
g = 9.81 # m/s
rho = 1.1839  # kg/m/m/m at 25 C

PLOT = True
PLOT_TRAJ = True
#log_variables = ['x_NED', 'e_xyz', 'angle_error', 'x_dot_NED', 'omega', ' u', 'PWM', 'x_ddot_NED', 'e_xyz_sp_NED', 'rate_sp']
log_variables = ['X', 'eulAng', 'Xdot', 'omega', 't', 'u', 'PWM','velDot', 'eulAngSP', 'rateSP', 'aero_force']
logger = Logger(log_variables)

## Loading configs 
with open('config.json') as f:
    config = json.load(f)

drone_params = config['drone_params']
fc_config =  config['fc_config']

# Quad physical parameters
m = drone_params['mass']
I = np.array(drone_params['Inertia'])
Fd = drone_params['Fd'] # drag force coeff of drone assuming it moves at 2 m / s  at 0 and angle of 0.1 %
points = drone_params['points']

# Propulsion Parameters
prop_config = drone_params['propulsion']
motors = prop_config['motors']
tau = prop_config['tau']
Cl = prop_config['Cl'] # Thrust to RPS ^ 2
Cd = prop_config['Cd']
pwm2rpm  = prop_config['pwm2rpm']# mapping ESC to rps experimentally



# Sim time and Sampling
f = 1000.
dt = 1. / f
logger.f =f
tstart = 0.
tend = 6.
time = np.linspace(tstart,tend,(int)(tend*f))

# Temp calculated here, should done in fc config
# hovering condition
wHover = np.array([1, 1, 1, 1]) * np.sqrt(m * g / 4. / Cl)
PWM_hover = wHover / 4.
w = wHover

# Initial Setpoint
eulAngSP = np.array([0., -0.06, 0.])
## Initialize simulation objects ##

# Drone Dynamic object
drone_mass_body = MassObj(name='Drone', mass=m, I = I)
drone_dyn_body = DynamicObj(massobject=drone_mass_body)

# Adding attachment points
for point_name in points.keys():
    point = points[point_name]
    drone_dyn_body.massobj.add_point(name=point_name, loc= point['location'], dcm = point['dcm_obj2body'])
drone_dyn_body.massobj.add_point(name='Aero', loc= np.zeros(3))

# Adding Motors (force objects)
motor_objs = []
for idx, motor_name in enumerate( motors.keys()):
    motor = Motor(Cl, Cd, tau, pwm2rpm)
    motor.w= wHover[idx]
    motor.direction = motors[motor_name]['direction']
    drone_dyn_body.add_forceobj(
        point_name = motor_name,
        force_obj = motor
    )
    motor_objs.append(motor)

# Adding AeroForces (force object)
drone_dyn_body.add_forceobj(
    point_name = 'Aero',
    force_obj = AeroForces(Fd)
)
# Flight computer

fc = FlightComputer(dt, fc_config)
fc.PWM_hover = PWM_hover

state_ = SixDofState.zero_states('ground_ned')
state_.g_fun = ned_gravity
state = np.hstack([state_.vector, w])
for t in time:

    if(t > 3):
        eulAngSP = np.array([-0.1, 0.1, 0])

    x = np.array(state[0:3])
    q_s = normalize(state[3:7])
    x_dot = np.array(state[7:10])
    omega = np.array(state[10:13])
    w = state[13:17]
    state_.update(state[0:13])

    # Model Updates
    drone_dyn_body.force_obj_dict['Aero'].update(x_dot)
    # Derivatives
    drone_dyn_body.force_obj_dict.keys()
    w_dot = []
    for idx, motor in enumerate(motor_objs):
        w_dot.append(motor.derivative(w[idx]))
    w_dot = np.array(w_dot)
    sixdofstate_dot = drone_dyn_body.derivative(state_)

    # vertcal velocity from barometer and rate in bodyframe
    vz = state[9]

    # Gyro
    # TODO: convert to body frame
    p = state[10]
    q = state[11]
    r = state[12]

    sensor_data = dict(
        vz = vz,
        q_state = q_s,
        p = p,
        q = q,
        r = r,
    )
    ## controller
    PWM, log = fc.update(sensor_data, eulAngSP)

    for PWMcmd, motor in zip(PWM, motor_objs):
        motor.set_command(PWMcmd)

    eulAng, eulAngSP, u, PWM, rateSP = log

    state_dot = np.hstack([sixdofstate_dot, w_dot])

    state = state + state_dot*dt

    #logging
    aero_force = drone_dyn_body.force_obj_dict['Aero'].force

    var_list = deepcopy([x , eulAng, x_dot, omega, t, u, PWM, state_dot[7:10], eulAngSP, rateSP, aero_force])
    logger.update_log(var_list)

# Extracting Sim data
logger.postprocess()
plot_logs(time, logger, PLOT, PLOT_TRAJ)