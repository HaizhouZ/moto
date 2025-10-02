from moto import var, create_sym, get_sym_sx, create_states
from moto import field
import casadi as cs
import numpy as np

def sym(name: str = None, dim: int = None, field: field = field.field_undefined, 
        base: var = None, default_val: np.ndarray | float | None = None):
    """
    Symbolic variable wrapper for CasADi expressions.
    (exists because nanobind does not support inheritance from multiple C++ classes directly)
    This class extends the `moto.sym_impl` and `cs::SX` classes to provide
    symbolic variables with additional functionality specific to the moto framework.
    """
    sym_handle: var = base if base is not None else create_sym(name, dim, field, default_val)
    s = cs.SX(get_sym_sx(sym_handle))
    s.sym_handle: var = sym_handle
    s.__str__ = lambda : f'sym(uid={s.uid}, name="{s.name}", dim={s.dim()}, field={s.field})'
    return s

def inputs(name: str, dim: int = 1, default_val: np.ndarray | float | None = None):
    return sym(name, dim, field=field.field_u, default_val=default_val)

def params(name: str, dim: int = 1, default_val: np.ndarray | float | None = None):
    return sym(name, dim, field=field.field_p, default_val=default_val)

def states(name: str, dim: int = 1, default_val: np.ndarray | float | None = None):
    x, y = create_states(name, dim, default_val)
    x = sym(base=x)
    y = sym(base=y)
    x.next = y
    y.prev = x
    return x, y