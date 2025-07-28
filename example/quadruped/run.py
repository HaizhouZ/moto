import moto
import casadi as cs

r, rn = moto.states("r", 2)
v = moto.inputs("v", 2)
dt = moto.params("dt", 1)
r_d = moto.params("r_d", 2)

dint = moto.constr("dint", 2, field=moto.field_dyn)
dint.add_arguments([r, v, dt])

c_state = moto.cost("c", [r, r_d], 100 * cs.sumsqr(r - r_d))
c_input = moto.cost("c_input", [v], 10 * cs.sumsqr(v))

prob = moto.ocp.create()
prob.add([dint, c_state, c_input])

prob_term = prob.clone()
print(prob.uid, prob_term.uid)
c_term = c_state.clone().as_terminal()
print(c_term.name)
prob_term.add(c_term)

sqp = moto.ns_sqp()
g = sqp.graph
g.add(sqp.create_node(prob))
# end_node = g.add(sqp.node_type(prob_term))
# g.add_edge(init_node, end_node, 50)