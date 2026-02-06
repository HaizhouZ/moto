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


    def integrate(self, x: np.ndarray, dx: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """integrate from x with dx, i.e., x + alpha * dx"""
        return self.__sym__.integrate(x, dx, alpha)

    def difference(self, x1: np.ndarray, x0: np.ndarray) -> np.ndarray: 
        """difference from x0 to x1, i.e., x1 - x0"""
        return self.__sym__.difference(x1, x0)
    
    def finalize(self):
        self.__sym__.finalize()