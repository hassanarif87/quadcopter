import numpy as np
from transforms3d import quaternions

class PIDController:
    def __init__(self, dt, p_gain, i_gain = 0, d_gain = 0, sat=None, name = 'default'):
        self.name = name
        self.dt = dt
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.sat = sat
        self.integrated_error = 0.
        self.prev_error = 0.
    def update(self, sp, fb):
        # PID
        dt = self.dt
        error = sp - fb
        self.integrated_error = self.integrated_error + (self.prev_error + error) / 2 * dt
        p_term = self.p_gain * (error)
        i_term =self.i_gain * (self.integrated_error)
        d_term = self.d_gain * (error - self.prev_error) / dt
        output = p_term + i_term + d_term
        self.prev_error = error
        if (self.sat is not None):
            # set Saturation
            output = min(output, self.sat)
            output = max(output, -self.sat)

        return output

    @property
    def unwind_integral(self):
        self.integrated_error=0


def quar_axis_error(q_sp, q_state):
    # Compute the error in quaternions from the setpoints and robot state in the body frame aligned with x, y, z axis

    #state_quat_conjugate = np.array([a2, -b2, -c2, -d2])
    # Quaternion multiplication q_set * (q_state)' target - state
    q_state_conj = quaternions.qconjugate(q_state)
    q_error =  quaternions.qmult(q_sp,q_state_conj)

    # Nearest rotation
    if (q_error[0] < 0):
        q_error = -1. * q_error

    axis_error = quaternions.quat2axangle(q_error)
    return axis_error[0] * axis_error[1]

def thrust_tilt(eulAng,PWM_hover):

    phi = eulAng[0,0] # Roll
    theta = eulAng[1,0] # Pitch
    psi =  eulAng[2,0]
    scaling = 1./(abs(np.sqrt(np.cos(phi)*np.cos(theta))))
    scaling = min (scaling, 1.3)
    return PWM_hover*scaling
