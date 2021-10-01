import numpy as np
from dataclasses import dataclass
_zero3 = np.zeros(3)
_zero3x3 = np.zeros([3,3])

@dataclass
class Point:
    location: np.array
    dcm_obj2body: np.array


class MassObj:
    def __init__ (self, mass, name, com = _zero3, I = _zero3x3):
        """
        Object discribing  the mass properties of a body

        Args:
            mass (float):mass of the body
            name (string): discriptive name
            com (np.array, optional): Center  of Mass w.r.t Body CS. Defaults to _zero3.
            I (np.array, optional): Inertia about CoM. Defaults to _zero3x3.
        """
        self.name = name
        self._mass  = mass
        self._com = com
        self._rg_sq = I / self._mass
        self.points = {
            "com": Point(
                location = self._com,
                dcm_obj2body=_zero3x3
                )
        }

    def __repr__(self):
        info= dict(
            name=self.name,
            mass=self.mass,
            com = self.com,
            inertia = self._rg_sq * self._mass
        )
        return str(info)

    @property
    def mass(self):
        return self._mass

    @property
    def com(self):
        return self._com

    @property
    def inertia(self):
        return self._mass * self._rg_sq

    def add_point(self, name, loc, dcm = _zero3x3):
        ''' Points can be used to attach other objects or apply forces in case of a DynObj
            Args:
                name of point
                location wrt mass obj coord system
        '''
        self.points[name] = Point(location= loc, dcm_obj2body=dcm)
