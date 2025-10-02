import moto
import casadi as cs

class var(cs.SX):
    '''
    This is a placeholder class to represent a symbolic variable
    that combines the functionality of `moto.var` and `casadi.SX`.
    It is intended to be used in type annotations and does not
    implement any actual functionality.
    '''

    @property
    def sym_handle(self) -> moto.var_alias: ...