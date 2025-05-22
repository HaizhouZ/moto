import casadi as cs
import atri

def sym(name: str, dim: int, field: atri.field):
    s = cs.SX.sym(name, dim)
    s.name = name
    s.field = field
    return s