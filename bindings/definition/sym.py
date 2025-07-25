from moto import shared_expr, create_sym, get_sym_sx
from moto import field
import casadi as cs


class sym(cs.SX):
    """
    Symbolic variable class for CasADi expressions.
    (exists because nanobind does not support inheritance from multiple C++ classes directly)
    This class extends the `moto.sym_impl` and `cs::SX` classes to provide
    symbolic variables with additional functionality specific to the moto framework.
    """

    def __init__(self, name: str, dim: int, field: field = field.field_undefined, base: shared_expr = None):
        self.sym_base = base if base is not None else create_sym(name, dim, field)
        print("Creating sym with base:", self.sym_base)
        self.name = name
        self.dim = dim
        self.field = field
        cs.SX.__init__(self, get_sym_sx(self.sym_base))
        print(self)

    def __str__(self):
        return f"sym(name=\"{self.name}\", dim={self.dim}, field={self.field})"

    @staticmethod
    def inputs(name: str, dim: int = 1):
        return sym(name, dim, field=field.field_u)

    @staticmethod
    def params(name: str, dim: int = 1):
        return sym(name, dim, field=field.field_p)

    @staticmethod
    def states(name: str, dim: int = 1):
        return sym(name, dim, field=field.field_x), sym(name + "_nxt", dim, field=field.field_y)
