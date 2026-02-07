import casadi as cs
import moto
import numpy as np


class var(cs.SX):
    def __init__(self, s: moto.sym):
        super().__init__(s.sx)
        self.__sym__ = s

    @property
    def sym(self) -> moto.sym:
        return self.__sym__

    def symbolic_integrate(self, x: cs.SX, dx: cs.SX) -> cs.SX:
        """integrate from x with dx, i.e., x + dx"""
        return self.__sym__.integrate(x, dx)

    def symbolic_difference(self, x1: cs.SX, x0: cs.SX) -> cs.SX:
        """difference from x0 to x1, i.e., x1 - x0"""
        return self.__sym__.difference(x1, x0)

    def integrate(
        self, x: np.ndarray, dx: np.ndarray, alpha: float = 1.0
    ) -> np.ndarray:
        """integrate from x with dx, i.e., x + alpha * dx"""
        return self.__sym__.integrate(x, dx, alpha)

    def difference(self, x1: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """difference from x0 to x1, i.e., x1 - x0"""
        return self.__sym__.difference(x1, x0)

    def finalize(self):
        self.__sym__.finalize()

    @property
    def name(self):
        return self.__sym__.name

    @property
    def dim(self):
        return self.__sym__.dim

    @property
    def tdim(self):
        return self.__sym__.tdim

    @tdim.setter
    def tdim(self, val):
        self.__sym__.tdim = val

    @property
    def default_value(self):
        return self.__sym__.default_value

    @default_value.setter
    def default_value(self, val):
        self.__sym__.default_value = val

    @property
    def uid(self):
        return self.__sym__.uid

    @property
    def sx(self):
        return self.__sym__.sx
