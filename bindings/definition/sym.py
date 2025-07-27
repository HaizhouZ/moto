from moto import var, create_sym, get_sym_sx, create_states
from moto import field
import casadi as cs


class sym(cs.SX):
    """
    Symbolic variable class for CasADi expressions.
    (exists because nanobind does not support inheritance from multiple C++ classes directly)
    This class extends the `moto.sym_impl` and `cs::SX` classes to provide
    symbolic variables with additional functionality specific to the moto framework.
    """

    def __init__(
        self, name: str = None, dim: int = None, field: field = field.field_undefined, base: var = None
    ):
        self.sym_base: var = base if base is not None else create_sym(name, dim, field)
        self.next: sym = None
        self.prev: sym = None
        cs.SX.__init__(self, get_sym_sx(self.sym_base))

    def __str__(self):
        return f'sym(uid={self.uid}, name="{self.name}", dim={self.dim}, field={self.field})'

    @property
    def name(self):
        return self.sym_base.name

    @name.setter
    def name(self, value):
        self.sym_base.name = value

    @property
    def dim(self):
        return self.sym_base.dim

    @dim.setter
    def dim(self, value):
        self.sym_base.dim = value

    @property
    def field(self):
        return self.sym_base.field

    @field.setter
    def field(self, value):
        self.sym_base.field = value

    @property
    def uid(self):
        return self.sym_base.uid

    @property
    def use_count(self):
        return self.sym_base.use_count

    @property
    def impl_use_count(self):
        return self.sym_base.impl_use_count

def inputs(name: str, dim: int = 1):
    return sym(name, dim, field=field.field_u)

def params(name: str, dim: int = 1):
    return sym(name, dim, field=field.field_p)

def states(name: str, dim: int = 1):
    x, y = create_states(name, dim)
    print(f"Created states: {x}, {y}")
    x = sym(base=x)
    y = sym(base=y)
    x.next = y
    y.prev = x
    return x, y