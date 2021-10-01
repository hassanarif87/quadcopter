import numpy as np
#import quaternion
from numpy import linalg as LA
from numpy import sin as sin
from numpy import cos as cos
from numpy import tan as tan

def normalize( vector ):
    # normalise vector
    if (LA.norm(vector) == 0 ):
        normalizedVector = vector
    else:
        normalizedVector = vector/LA.norm(vector)
    return normalizedVector

def quat2rotm(q):
    s = 1.
    rmatrix =np.array([[1.-2.*s*(q[2]*q[2]+q[3]*q[3]), 2.*s*(q[1]*q[2]-q[3]*q[0]), 2.*s*(q[1]*q[3]+q[2]*q[0])],
                        [2.*s*(q[1]*q[2]+q[3]*q[0]), 1.-2.*s*(q[1]*q[1]+q[3]*q[3]), 2.*s*(q[2]*q[3]-q[1]*q[0])],
                        [2.*s*(q[1]*q[3]-q[2]*q[0]), 2.*s*(q[2]*q[3]+q[1]*q[0]), 1.-2.*s*(q[1]*q[1]+q[2]*q[2])]])
    return rmatrix

def  rot_matrix3d(eulAng):
    # Eular angle transformation using X Y Z convention
    # rotates vector in a frame
    phi = eulAng[0,0]
    theta = eulAng[1,0]
    psi =  eulAng[2,0]

    Rx = np.array([ [1., 0., 0.], [0., cos(phi), -sin(phi)], [0.,  sin(phi), cos(phi)]]) #Roll
    Ry = np.array([ [cos(theta), 0., sin(theta)], [0., 1., 0.], [-sin(theta), 0. , cos(theta)]]) #Pitch
    Rz = np.array([ [cos(psi), -sin(psi), 0.], [sin(psi), cos(psi), 0.], [0., 0., 1.] ]) #Yaw

    return LA.multi_dot([Rz,Ry,Rx])


def theta_wraper(theta):
    # keeps theta + / - pi
    if (theta >= (np.pi)):
        theta = theta - 2 * np.pi
    elif(theta < -(np.pi)):
        theta = theta + 2 * np.pi
    return theta

class Logger:
    def __init__(self, log_name_lst) -> None:
        self.log_name_lst = log_name_lst
        self.log = dict()
        self._init_log()
    def _init_log (self):
        for var in self.log_name_lst:
            self.log[var]= []
    def update_log(self,var_lst):
        if len(var_lst) is not len(self.log_name_lst):
            raise IndexError ('Logged variables not the same as defined list: ', self.log_variables)
        for log_name , val in zip(self.log_name_lst, var_lst):
            l = np.size(val)# Flatten vector
            self.log[log_name].append(val.reshape(l))
    def postprocess(self):
        for key in self.log.keys():
            self.log[key] = np.transpose(np.array(self.log[key]))