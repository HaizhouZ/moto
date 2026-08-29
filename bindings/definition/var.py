import casadi as cs
import moto
import numpy as np


class var(cs.SX):
    """A CasADi expression carrying a registered moto symbol."""

    def __init__(self, s: moto.sym):
        """Construct a symbolic variable from its registered moto symbol."""
        super().__init__(s.sx)
        self.__sym__ = s

    @property
    def sym(self) -> moto.sym:
        """The registered moto symbol represented by this expression."""
        return self.__sym__

    def symbolic_integrate(self, x: cs.SX, dx: cs.SX) -> cs.SX:
        """integrate from x with dx, i.e., x + dx"""
        return self.__sym__.symbolic_integrate(x, dx)

    def symbolic_difference(self, x1: cs.SX, x0: cs.SX) -> cs.SX:
        """difference from x0 to x1, i.e., x1 - x0"""
        return self.__sym__.symbolic_difference(x1, x0)

    def clone(self, name: str) -> "var":
        """Clone into an independent symbol with a fresh identity."""
        return self.__sym__.clone(name)

    def integrate(
        self, x: np.ndarray, dx: np.ndarray, alpha: float = 1.0
    ) -> np.ndarray:
        """integrate from x with dx, i.e., x + alpha * dx"""
        return self.__sym__.integrate(x, dx, alpha)

    def difference(self, x1: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """difference from x0 to x1, i.e., x1 - x0"""
        return self.__sym__.difference(x1, x0)

    def finalize(self):
        """Finalize the underlying symbol."""
        self.__sym__.finalize()

    @property
    def name(self):
        """Symbol name."""
        return self.__sym__.name

    @property
    def dim(self):
        """Storage dimension."""
        return self.__sym__.dim

    @property
    def tdim(self):
        """Tangent-space dimension."""
        return self.__sym__.tdim

    @property
    def default_value(self):
        """Default numeric value."""
        return self.__sym__.default_value

    @default_value.setter
    def default_value(self, val):
        """Set the default numeric value."""
        self.__sym__.default_value = val

    @property
    def uid(self):
        """Stable symbol identity."""
        return self.__sym__.uid

    @property
    def sx(self):
        """Underlying CasADi SX expression."""
        return self.__sym__.sx
