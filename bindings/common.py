import casadi as cs
import moto


def sym(name: str, dim: int, field: moto.field):
    s = cs.SX.sym(name, dim)
    s.name = name
    s.field = field
    return s


def make_next(name, dim):
    # this suffix is compulsory
    if not name.endswith("_nxt"):
        name = name + "_nxt"
    return sym(name, dim, moto.field_y)


def states(name, dim):
    x = sym(name, dim, moto.field_x)
    xn = make_next(name, dim)
    return x, xn


def inputs(name, dim):
    u = sym(name, dim, moto.field_u)
    return u


def params(name, dim=1):
    p = sym(name, dim, moto.field_p)
    return p
