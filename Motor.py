import numpy as np
from TemplateObj import ForceObjTemplate, DynamicObjTemplate

class Motor(ForceObjTemplate, DynamicObjTemplate):
    def __init__(self, Cl, Cd, tau, pwn2rpm):
        self.Cl = Cl
        self.Cd = Cd
        self.tau = tau
        self.pwm2rpm = pwn2rpm
        self.direction = 1
        self.wCmd = 0 # Should states be members?

    def set_command(self, cmdPWM):
        self.wCmd = cmdPWM * self.pwm2rpm

    def derivative(self,w):
        dw = (self.wCmd - w) / self.tau
        return dw

    @property
    def force(self):
        """Force in motor frame of reference"""
        return np.array([self.Cl * self.w, 0,0])

    @property
    def moment(self):
        """Moment in motor frame of reference"""
        return np.array([self.Cd * self.w, 0,0])