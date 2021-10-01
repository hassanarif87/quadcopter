import typing
import numpy as np
from dataclasses import dataclass, field
from transforms3d import quaternions
from numpy import linalg as LA

def zero_gravity():
    return np.array([0.0,0.0,0.0])
def ned_gravity():
    return np.array([0.0,0.0,9.81])
@dataclass
class SixDofState:
    """Class keeping track of the 6dof object states"""
    frame_name: str
    x: np.array
    q: np.array
    x_dot: np.array
    omega: np.array
    inertial_frame: bool = False
    g_fun: typing.Any = field(repr=False, default=zero_gravity)
    _g: np.array = field(repr=False, default=None)

    @property
    def dcm_body2frame(self):
        """ Integration frame of reference"""
        return quaternions.quat2mat(self.q)
    @property
    def g(self):
        """Gravity function"""
        if self._g is None:
            return self.g_fun()
        else:
            return self._g
    @property
    def vector(self):
        return np.hstack([self.x, self.q, self.x_dot, self.omega])

    def update(self, state):
        self.x = np.array(state[0:3])
        self.q = np.array(state[3:7])
        self.x_dot = np.array(state[7:10])
        self.omega = np.array(state[10:13])
    @classmethod
    def zero_states(cls, frame_name):
        """Class method to craete obejct wiht zero states
        Args:
            frame_name (String): Name of integraition frame
        """
        return cls(
            frame_name = frame_name,
            x = np.zeros(3),
            q = np.array([1,0,0,0]),
            x_dot=np.zeros(3),
            omega = np.zeros(3)
        )


class DynamicObj:
    def __init__(self, massobject):
        self.massobj = massobject
        self.force_obj_dict = dict()
        # TODO: Add abllity to integrate in more frames?
        self.state = None # should the states be added here?

    def init_state(self,state_frame):
        self.state = state_frame

    def add_forceobj(self, point_name, force_obj):
        """Attaches a force torque proucing object to a point
        Args:
            Point Name
            Force Object
        """
        # TODO: Check if point exists if not print list of points
        self.force_obj_dict[point_name] = force_obj

    def force_collector_update(self):
        """Iterates through force objects attached to the DynObj and
        collects and calculates the forces and moments about the DynObj
        center of mass
        """
        sum_force_body = np.zeros(3)
        sum_torque_body = np.zeros(3)
        point_list = self.massobj.points
        com = point_list['com']

        for name in self.force_obj_dict.keys():
            force_obj = self.force_obj_dict[name]
            dcm_obj2body = np.array(point_list[name].dcm_obj2body)
            force_body = force_obj.force
            torque_body = force_obj.moment
            moment_arm = com.location - point_list[name].location
            torque_body += np.cross(moment_arm, force_body)
            # Transform from force obj to body frame
            sum_force_body += np.dot(dcm_obj2body, force_body)
            sum_torque_body += np.dot(dcm_obj2body, torque_body)
        return sum_force_body, sum_torque_body

    def derivative(self, state):
        """Calculates the derivaties of the 6dof Dynamic object
        Args:
            state (SixDofState): DynObj states, integration frame of reference and gravity fucntion
        Returns:
            state_dot (np.array): Derivatives of the state vector
        """
        # Extracting forces and torques
        sum_forces__body, sum_torque__body = self.force_collector_update()

        # Extracting States
        x = state.x
        q = state.q
        x_dot = state.x_dot
        omega = state.omega
        dcm_body2frame = state.dcm_body2frame

        m = self.massobj.mass
        I = self.massobj.inertia

        # Translational dynamics
        sum_forces = np.dot(dcm_body2frame, sum_forces__body)
        x_ddot = state.g + sum_forces / m

        # Rotational Dynamics
        #sum_torque = np.dot(dcm_body2frame, sum_torque__body)
        sum_torque =  sum_torque__body

        s = q[0]
        v = np.array([q[1], q[2], q[3]])
        sdot = -0.5 * (np.dot(v, omega))
        vdot = 0.5 * (s * omega + np.cross(omega, v))
        q_dot = np.append(sdot, vdot)
        Iomega = np.dot(I, omega)

        invI =  LA.inv(I)
        x_product = np.cross(omega, Iomega)
        omega_dot = np.dot(invI,(-1.*x_product)) + np.dot(invI, sum_torque)
        state_dot = np.hstack([x_dot, q_dot, x_ddot, omega_dot])


        return state_dot