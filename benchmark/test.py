import moto
import numpy as np

u = moto.sym.inputs("u", 3, np.array([0.1, 0.2, 0.3]))
b = moto.sym.params("b", 3, np.array([1.0, 1.0, 1.0]))
prob = moto.ocp.create()
prob2 = moto.ocp.create()
prob.add(u)
prob2.add(u)
prob.add([u, b])
prob2.add([u, b])
print([u.__sym__])
print(prob.exprs(moto.field_u))
# print(prob.exprs(moto.field_p))
# print(prob2.exprs(moto.field_u))
print("---")
u2 = moto.var(prob.exprs(moto.field_u)[0])
# del prob
del prob2

print([u.__sym__])
print(u2.__sym__)
print("-----")
z = moto.sym.params("z", 3, np.array([5.0, 10.0, 15.0]))
f = moto.constr.create("f", [u, b], z * (u + b), moto.approx_order_first)
f.add_argument(z)
print(f.uid)
print(f.in_args)

f = f.cast_ineq()

f.finalize()

print("------")
c = moto.cost.create("c", [u, b], (u - b).T @ (u - b))
print("Step1")
c.set_diag_hess()
print("Step2")
c.finalize()

prob.add([f, c])

print(f.uid)

print("-------------")

x, xn = moto.sym.states("x", 3)

d = moto.dense_dynamics.create("d", [x, u, xn], xn - (x + u), moto.approx_order_first)

prob.add(d)


prob.wait_until_ready()
print("Problem ready")