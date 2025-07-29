import moto
import casadi as cs
import numpy as np

r, rn = moto.states("r", 2)
v = moto.inputs("v", 2)
dt = moto.params("dt", 1)
r_d = moto.params("r_d", 2)

dint = moto.constr("dint", dim=2, field=moto.field_dyn)
dint.add_arguments([rn, r, v, dt])
# dt = 0.01
# dint = moto.constr("dint", [rn, r, v, dt], rn - (r + v * dt), field=moto.field_dyn)


def dyn_value(data: moto.func_approx_data):
    data.v = data[rn] - (data[r] + data[v] * data[dt])

def dyn_jac(data: moto.func_approx_data):
    data.jac(rn)
    data.set_jac(rn, np.eye(2))
    data.set_jac(r, -np.eye(2))
    data.set_jac(v, -data[dt] * np.eye(2))


dint.value = dyn_value
dint.jacobian = dyn_jac

c_state = moto.cost("c", [r, r_d, dt], 100 * cs.sumsqr(r - r_d) * dt)
c_input = moto.cost("c_input", [v, dt], 1 * cs.sumsqr(v) * dt)
c_state_terminal = moto.cost("c", [r, r_d], 100 * cs.sumsqr(r - r_d)).as_terminal()

prob = moto.ocp.create()
prob.add([dint, c_state, c_input])

prob_term = prob.clone()
print(prob.uid, prob_term.uid)
# c_term = c_state.clone().as_terminal()
prob_term.add(c_state_terminal)

sqp = moto.ns_sqp()
g = sqp.graph
init_node = g.set_head(g.add(sqp.create_node(prob)))
end_node = g.set_tail(g.add(sqp.create_node(prob_term)))
g.add_edge(init_node, end_node, 50)


def init(cur: moto.ns_sqp.data_type):
    cur.value[r_d] = np.array([1.0, 1.0])
    cur.value[dt] = 0.01


g.for_each_parallel(init)

sqp.update(10)
