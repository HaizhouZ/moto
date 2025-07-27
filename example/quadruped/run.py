import moto
import casadi as cs

r, rn = moto.states("r", 2)
v = moto.inputs("v", 2)
dt = moto.params("dt", 1)
r_d = moto.params("r_d", 2)

dint = moto.constr("dint", 2, field=moto.field_dyn)
dint.add_arguments([r, v, dt])

c = moto.cost("c", [r, r_d, v], 100 * cs.sumsqr(r - r_d) + cs.sumsqr(v))
# prob = moto.ocp.create()
# prob.add([dint, c])

# prob_term = prob.clone()
# c_term = c.as_terminal()
# prob_term.add(c_term)

