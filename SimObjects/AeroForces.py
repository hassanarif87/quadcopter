import numpy as np
from SimObjects.TemplateObj import ForceObjTemplate

class AeroForces(ForceObjTemplate):
    def __init__(self, Fd):
        self.Fd = Fd
        self._x_dot = None # pointer to state??

    def update(self, x_dot):
        self._x_dot = x_dot

    @property
    def force(self):
        """Force inmotor frame of reference"""
        return self._x_dot * -self.Fd

    @property
    def moment(self):
        """Moment in motor frame of reference"""
        return np.zeros(3)