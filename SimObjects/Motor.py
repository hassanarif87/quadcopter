import numpy as np
from SimObjects.TemplateObj import ForceObjTemplate, DynamicObjTemplate

class Motor(ForceObjTemplate, DynamicObjTemplate):
    def __init__(self, Cl, Cd, tau, pwn2rpm):
        self.Cl = Cl
        self.Cd = Cd
        self.tau = tau
        self.pwm2rpm = pwn2rpm
        self.direction = 1
        self.w = 0
        self.wCmd = 0 # Should states be members?

    def set_command(self, cmdPWM):
        self.wCmd = cmdPWM * self.pwm2rpm

    def derivative(self,w):
        self.w = w
        dw = (self.wCmd - w) / self.tau
        return dw

    @property
    def force(self):
        """Force in motor frame of reference"""
        return np.array([ 0.0, 0.0, self.Cl * self.w**2])

    @property
    def moment(self):
        """Moment in motor frame of reference"""
        return np.array([ 0.0, 0.0, self.direction  * self.Cd * self.w**2])