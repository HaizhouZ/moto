import casadi as cs
import moto

class var(cs.SX):
    def __init__(self, s: moto.sym):
        super().__init__(s.sx)
        self.__sym__ = s
    
    @property
    def sym(self) -> moto.sym:
        return self.__sym__